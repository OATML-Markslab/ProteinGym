import datetime
import os
import sys
from resource import getrusage, RUSAGE_SELF

import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from scipy.special import erfinv
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils.data_utils import one_hot_3D, get_dataloader
from . import VAE_encoder, VAE_decoder


class VAE_model(nn.Module):
    """
    Class for the VAE model with estimation of weights distribution parameters via Mean-Field VI.
    """

    def __init__(self,
                 model_name,
                 data,
                 encoder_parameters,
                 decoder_parameters,
                 random_seed,
                 seq_len=None,
                 alphabet_size=None,
                 Neff=None,
                 ):

        super().__init__()

        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.random_seed = random_seed
        torch.manual_seed(random_seed)

        self.seq_len = seq_len if seq_len is not None else data.seq_len
        self.alphabet_size = alphabet_size if alphabet_size is not None else data.alphabet_size
        self.Neff = Neff if Neff is not None else data.Neff

        self.encoder_parameters = encoder_parameters
        self.decoder_parameters = decoder_parameters

        encoder_parameters['seq_len'] = self.seq_len
        encoder_parameters['alphabet_size'] = self.alphabet_size
        decoder_parameters['seq_len'] = self.seq_len
        decoder_parameters['alphabet_size'] = self.alphabet_size

        self.encoder = VAE_encoder.VAE_MLP_encoder(params=encoder_parameters)
        if decoder_parameters['bayesian_decoder']:
            self.decoder = VAE_decoder.VAE_Bayesian_MLP_decoder(params=decoder_parameters)
        else:
            self.decoder = VAE_decoder.VAE_Standard_MLP_decoder(params=decoder_parameters)
        self.logit_sparsity_p = decoder_parameters['logit_sparsity_p']

    def sample_latent(self, mu, log_var):
        """
        Samples a latent vector via reparametrization trick
        """
        eps = torch.randn_like(mu).to(self.device)
        z = torch.exp(0.5 * log_var) * eps + mu
        return z

    def KLD_diag_gaussians(self, mu, logvar, p_mu, p_logvar):
        """
        KL divergence between diagonal gaussian with prior diagonal gaussian.
        """
        KLD = 0.5 * (p_logvar - logvar) + 0.5 * (torch.exp(logvar) + torch.pow(mu - p_mu, 2)) / (
                torch.exp(p_logvar) + 1e-20) - 0.5

        return torch.sum(KLD)

    def annealing_factor(self, annealing_warm_up, training_step):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        if training_step < annealing_warm_up:
            return training_step / annealing_warm_up
        else:
            return 1

    def KLD_global_parameters(self):
        """
        KL divergence between the variational distributions and the priors (for the decoder weights).
        """
        KLD_decoder_params = 0.0
        zero_tensor = torch.tensor(0.0).to(self.device)

        for layer_index in range(len(self.decoder.hidden_layers_sizes)):
            for param_type in ['weight', 'bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                    self.decoder.state_dict(keep_vars=True)[
                        'hidden_layers_mean.' + str(layer_index) + '.' + param_type].flatten(),
                    self.decoder.state_dict(keep_vars=True)[
                        'hidden_layers_log_var.' + str(layer_index) + '.' + param_type].flatten(),
                    zero_tensor,
                    zero_tensor
                )

        for param_type in ['weight', 'bias']:
            KLD_decoder_params += self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['last_hidden_layer_' + param_type + '_mean'].flatten(),
                self.decoder.state_dict(keep_vars=True)['last_hidden_layer_' + param_type + '_log_var'].flatten(),
                zero_tensor,
                zero_tensor
            )

        if self.decoder.include_sparsity:
            self.logit_scale_sigma = 4.0
            self.logit_scale_mu = 2.0 ** 0.5 * self.logit_scale_sigma * erfinv(2.0 * self.logit_sparsity_p - 1.0)

            sparsity_mu = torch.tensor(self.logit_scale_mu).to(self.device)
            sparsity_log_var = torch.log(torch.tensor(self.logit_scale_sigma ** 2)).to(self.device)
            KLD_decoder_params += self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['sparsity_weight_mean'].flatten(),
                self.decoder.state_dict(keep_vars=True)['sparsity_weight_log_var'].flatten(),
                sparsity_mu,
                sparsity_log_var
            )

        if self.decoder.convolve_output:
            for param_type in ['weight']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                    self.decoder.state_dict(keep_vars=True)['output_convolution_mean.' + param_type].flatten(),
                    self.decoder.state_dict(keep_vars=True)['output_convolution_log_var.' + param_type].flatten(),
                    zero_tensor,
                    zero_tensor
                )

        if self.decoder.include_temperature_scaler:
            KLD_decoder_params += self.KLD_diag_gaussians(
                self.decoder.state_dict(keep_vars=True)['temperature_scaler_mean'].flatten(),
                self.decoder.state_dict(keep_vars=True)['temperature_scaler_log_var'].flatten(),
                zero_tensor,
                zero_tensor
            )
        return KLD_decoder_params

    def loss_function(self, x_recon_log, x, mu, log_var, kl_latent_scale, kl_global_params_scale, annealing_warm_up,
                      training_step, Neff):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        """
        BCE = F.binary_cross_entropy_with_logits(x_recon_log, x, reduction='sum') / x.shape[0]
        KLD_latent = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.shape[0]
        if self.decoder.bayesian_decoder:
            KLD_decoder_params_normalized = self.KLD_global_parameters() / Neff
        else:
            KLD_decoder_params_normalized = 0.0
        warm_up_scale = self.annealing_factor(annealing_warm_up, training_step)
        neg_ELBO = BCE + warm_up_scale * (
                kl_latent_scale * KLD_latent + kl_global_params_scale * KLD_decoder_params_normalized)
        return neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized

    def all_likelihood_components(self, x):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        """
        mu, log_var = self.encoder(x)
        z = self.sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)

        recon_x_log = recon_x_log.view(-1, self.alphabet_size * self.seq_len)
        x = x.view(-1, self.alphabet_size * self.seq_len)

        BCE_batch_tensor = torch.sum(F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none'), dim=1)
        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        ELBO_batch_tensor = -(BCE_batch_tensor + KLD_batch_tensor)

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor

    def all_likelihood_components_z(self, x, mu, log_var):
        """Skip the encoder part and directly sample z"""
        # Need to run mu, log_var = self.encoder(x) first
        z = self.sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)

        recon_x_log = recon_x_log.view(-1, self.alphabet_size * self.seq_len)
        x = x.view(-1, self.alphabet_size * self.seq_len)

        BCE_batch_tensor = torch.sum(F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none'), dim=1)
        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        ELBO_batch_tensor = -(BCE_batch_tensor + KLD_batch_tensor)

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor

    def train_model(self, data, training_parameters, use_dataloader=False):
        """
        Training procedure for the VAE model.
        If use_validation_set is True then:
            - we split the alignment data in train/val sets.
            - we train up to num_training_steps steps but store the version of the model with lowest loss on validation set across training
        If not, then we train the model for num_training_steps and save the model at the end of training
        """
        if torch.cuda.is_available():
            cudnn.benchmark = True
        self.train()

        if training_parameters['log_training_info']:
            filename = training_parameters['training_logs_location'] + os.sep + self.model_name + "_losses.csv"
            with open(filename, "a") as logs:
                logs.write("Number of sequences in alignment file:\t" + str(data.num_sequences) + "\n")
                logs.write("Neff:\t" + str(self.Neff) + "\n")
                logs.write("Alignment sequence length:\t" + str(data.seq_len) + "\n")

        optimizer = optim.Adam(self.parameters(), lr=training_parameters['learning_rate'],
                               weight_decay=training_parameters['l2_regularization'])

        if training_parameters['use_lr_scheduler']:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_parameters['lr_scheduler_step_size'],
                                                  gamma=training_parameters['lr_scheduler_gamma'])

        if training_parameters['use_validation_set']:
            # TODO fix this for use with a dataloader
            x_train, x_val, weights_train, weights_val = train_test_split(data.one_hot_encoding, data.weights,
                                                                          test_size=training_parameters[
                                                                              'validation_set_pct'],
                                                                          random_state=self.random_seed)
            best_val_loss = float('inf')
            best_model_step_index = 0
        else:
            x_train = data.one_hot_encoding
            weights_train = data.weights
            best_val_loss = None
            best_model_step_index = training_parameters['num_training_steps']
        
        seq_sample_probs = weights_train / np.sum(weights_train)
        assert len(data.seq_name_to_sequence) == weights_train.shape[0]  # One weight per sequence
        
        # TMP TODO: Keep old behaviour for comparison
        if use_dataloader:
            # Stream one-hot encodings
            dataloader = get_dataloader(msa_data=data, batch_size=training_parameters['batch_size'], num_training_steps=training_parameters['num_training_steps'])
        else:
            batch_order = np.arange(x_train.shape[0])
            assert batch_order.shape == seq_sample_probs.shape, f"batch_order and seq_sample_probs must have the same shape. batch_order.shape={batch_order.shape}, seq_sample_probs.shape={seq_sample_probs.shape}"
            def get_mock_dataloader():
                while True:
                    # Sample a batch according to sequence weight
                    batch_index = np.random.choice(batch_order, training_parameters['batch_size'], p=seq_sample_probs).tolist()
                    batch = x_train[batch_index]
                    yield batch
            dataloader = get_mock_dataloader()

        self.Neff_training = np.sum(weights_train)

        start = time.time()
        train_loss = 0
        print("debug starting training here:")
        for training_step, batch in enumerate(tqdm(dataloader, desc="Training model", total=training_parameters['num_training_steps'], mininterval=2)):#mininterval=10)):
            
            if training_step >= training_parameters['num_training_steps']:
                print("debug Breaking at step", training_step)
                break
            x = batch.to(self.device, dtype=self.dtype)
            if training_step == 0:
                print("Got batch 1")
                
            optimizer.zero_grad()

            mu, log_var = self.encoder(x)
            z = self.sample_latent(mu, log_var)
            recon_x_log = self.decoder(z)

            neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = self.loss_function(
                recon_x_log, x, mu, log_var,
                training_parameters['kl_latent_scale'],
                training_parameters['kl_global_params_scale'],
                training_parameters['annealing_warm_up'],
                training_step,
                self.Neff_training)

            neg_ELBO.backward()
            optimizer.step()

            if training_parameters['use_lr_scheduler']:
                scheduler.step()

            if training_step % training_parameters['log_training_freq'] == 0:
                progress = "|Train : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, KLD_decoder_params_norm: {4:.3f}, Time: {5:.2f} |".format(
                    training_step, neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized, time.time() - start)
                print(progress)

                if training_parameters['log_training_info']:
                    with open(filename, "a+") as logs:
                        logs.write(progress + "\n")

            if training_step % training_parameters['save_model_params_freq'] == 0:
                self.save(model_checkpoint=training_parameters[
                                               'model_checkpoint_location'] + os.sep + self.model_name + "_step_" + str(
                    training_step),
                          encoder_parameters=self.encoder_parameters,
                          decoder_parameters=self.decoder_parameters,
                          training_parameters=training_parameters)

            if training_parameters['use_validation_set'] and training_step % training_parameters[
                'validation_freq'] == 0:
                x_val = torch.tensor(x_val, dtype=self.dtype).to(self.device)
                val_neg_ELBO, val_BCE, val_KLD_latent, val_KLD_global_parameters = self.test_model(x_val, weights_val,
                                                                                                   training_parameters[
                                                                                                       'batch_size'])

                progress_val = "\t\t\t|Val : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, KLD_decoder_params_norm: {4:.3f}, Time: {5:.2f} |".format(
                    training_step, val_neg_ELBO, val_BCE, val_KLD_latent, val_KLD_global_parameters,
                    time.time() - start)
                print(progress_val)
                if training_parameters['log_training_info']:
                    with open(filename, "a+") as logs:
                        logs.write(progress_val + "\n")

                if val_neg_ELBO < best_val_loss:
                    best_val_loss = val_neg_ELBO
                    best_model_step_index = training_step
                    self.save(model_checkpoint=training_parameters[
                                                   'model_checkpoint_location'] + os.sep + self.model_name + "_best",
                              encoder_parameters=self.encoder_parameters,
                              decoder_parameters=self.decoder_parameters,
                              training_parameters=training_parameters)
                self.train()
        print("TMP: Finished training, last training_step=", training_step)
        
        
    def test_model(self, x_val, weights_val, batch_size):
        self.eval()

        with torch.no_grad():
            val_batch_order = np.arange(x_val.shape[0])
            val_seq_sample_probs = weights_val / np.sum(weights_val)

            val_batch_index = np.random.choice(val_batch_order, batch_size, p=val_seq_sample_probs).tolist()
            x = torch.tensor(x_val[val_batch_index], dtype=self.dtype).to(self.device)
            mu, log_var = self.encoder(x)
            z = self.sample_latent(mu, log_var)
            recon_x_log = self.decoder(z)

            neg_ELBO, BCE, KLD_latent, KLD_global_parameters = self.loss_function(recon_x_log, x, mu, log_var,
                                                                                  kl_latent_scale=1.0,
                                                                                  kl_global_params_scale=1.0,
                                                                                  annealing_warm_up=0, training_step=1,
                                                                                  Neff=self.Neff_training)  # set annealing factor to 1

        return neg_ELBO.item(), BCE.item(), KLD_latent.item(), KLD_global_parameters.item()

    def save(self, model_checkpoint, encoder_parameters, decoder_parameters, training_parameters, batch_size=256):
        # Create intermediate dirs above this
        os.makedirs(os.path.dirname(model_checkpoint), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'encoder_parameters': encoder_parameters,
            'decoder_parameters': decoder_parameters,
            'training_parameters': training_parameters,
        }, model_checkpoint)

    def compute_evol_indices(self, msa_data, list_mutations_location, num_samples, batch_size=256,
                             mutant_column="mutations", num_chunks=1, aggregation_method="full"):
        list_valid_mutations = []
        evol_indices = []

        full_data = pd.read_csv(list_mutations_location, header=0)
        print("Full length: ", len(full_data))
        size_per_chunk = int(len(full_data) / num_chunks)
        print("size_per_chunk: " + str(size_per_chunk))

        for chunk in range(num_chunks):
            print("chunk #: " + str(chunk))
            data_chunk = full_data[chunk * size_per_chunk:(chunk + 1) * size_per_chunk]
            list_valid_mutations_chunk, evol_indices_chunk, _, _ = self.compute_evol_indices_chunk(msa_data=msa_data,
                                                                                                   list_mutations_location=data_chunk,
                                                                                                   num_samples=num_samples,
                                                                                                   batch_size=batch_size,
                                                                                                   mutant_column=mutant_column,
                                                                                                   aggregation_method=aggregation_method)
            list_valid_mutations.extend(list(list_valid_mutations_chunk))
            evol_indices.extend(list(evol_indices_chunk))
        return list_valid_mutations, evol_indices, '', ''

    def compute_evol_indices_chunk(self, msa_data, list_mutations_location, num_samples, batch_size=256,
                                   mutant_column="mutations", aggregation_method="full"):
        """
        The column in the list_mutations dataframe that contains the mutant(s) for a given variant should be called "mutations"
        """

        # Note: wt is added inside this function, so no need to add a row in csv/dataframe input with wt

        # Multiple mutations are to be passed colon-separated
        list_mutations = list_mutations_location  # pd.read_csv(list_mutations_location, header=0)

        # Remove (multiple) mutations that are invalid
        list_valid_mutations = ['wt']
        list_valid_mutated_sequences = {}
        list_valid_mutated_sequences['wt'] = msa_data.focus_seq_trimmed  # first sequence in the list is the wild_type

        if aggregation_method not in ["full", "batch", "online"]:
            raise ValueError("Invalid aggregation method: {}".format(aggregation_method))

        for mutation in list_mutations[mutant_column]:
            try:
                individual_substitutions = str(mutation).split(':')
            except Exception as e:
                print("Error with mutant {}".format(str(mutation)))
                print("Specific error: " + str(e))
                continue
            mutated_sequence = list(msa_data.focus_seq_trimmed)[:]
            fully_valid_mutation = True
            for mut in individual_substitutions:
                try:
                    wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
                    if wt_aa == mut_aa:
                        continue 
                    # Log specific invalid mutants
                    if pos not in msa_data.uniprot_focus_col_to_wt_aa_dict:
                        print("pos {} not in uniprot_focus_col_to_wt_aa_dict".format(pos))
                        fully_valid_mutation = False
                    # Given it's in the dict, check if it's a valid mutation
                    elif msa_data.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa:
                        print("wt_aa {} != uniprot_focus_col_to_wt_aa_dict[{}] {}".format(
                            wt_aa, pos, msa_data.uniprot_focus_col_to_wt_aa_dict[pos]))
                        fully_valid_mutation = False
                    if mut not in msa_data.mutant_to_letter_pos_idx_focus_list:
                        print("mut {} not in mutant_to_letter_pos_idx_focus_list".format(mut))
                        fully_valid_mutation = False

                    if fully_valid_mutation:
                        wt_aa, pos, idx_focus = msa_data.mutant_to_letter_pos_idx_focus_list[mut]
                        mutated_sequence[idx_focus] = mut_aa  # perform the corresponding AA substitution
                    else:
                        print("Not a valid mutant: " + mutation)
                        break

                except Exception as e:
                    print("Error processing mutation {} in mutant {}".format(str(mut), str(mutation)))
                    print("Specific error: " + str(e))
                    fully_valid_mutation = False
                    break

            if fully_valid_mutation:
                list_valid_mutations.append(mutation)
                list_valid_mutated_sequences[mutation] = ''.join(mutated_sequence)

        # One-hot encoding of mutated sequences
        mutated_sequences_one_hot = one_hot_3D(list_valid_mutations, list_valid_mutated_sequences,
                                               alphabet=msa_data.alphabet, seq_length=len(msa_data.focus_cols))

        # TODO for low memory might need to calculate one-hot on the fly, or fix chunked calculation with elbo - elbo_wt
        mutated_sequences_one_hot = torch.tensor(mutated_sequences_one_hot, dtype=torch.bool)
        print("One-hot encoding of mutated sequences complete")
        print(f"{datetime.datetime.now()} Peak memory in GB: {getrusage(RUSAGE_SELF).ru_maxrss / 1024 ** 2:.3f}")
        # https://stackoverflow.com/questions/54361763/pytorch-why-is-the-memory-occupied-by-the-tensor-variable-so-small/54365012#54365012
        print(
            f"tmp debug: storage size of mutated_sequences_one_hot: {sys.getsizeof(mutated_sequences_one_hot.storage()) / 1e9:.4f} GB")
        dataloader = torch.utils.data.DataLoader(mutated_sequences_one_hot, batch_size=batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True)

        if aggregation_method == "full":
            prediction_matrix = torch.zeros((len(list_valid_mutations), num_samples), dtype=self.dtype)
            print(
                f"tmp debug: storage size of mutated_sequences_one_hot: {sys.getsizeof(prediction_matrix.storage()) / 1e9:.4f} GB")
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, 'Looping through mutation batches')):
                    x = batch.type(self.dtype).to(self.device)
                    for j in tqdm(range(num_samples),
                                       'Looping through number of samples for batch #: ' + str(i + 1), mininterval=5):
                        seq_predictions, _, _ = self.all_likelihood_components(x)
                        prediction_matrix[i * batch_size:i * batch_size + len(x), j] = seq_predictions
                    tqdm.write('\n')
                mean_predictions = prediction_matrix.mean(dim=1, keepdim=False)
                std_predictions = prediction_matrix.std(dim=1, keepdim=False)
                delta_elbos = mean_predictions - mean_predictions[0]
                evol_indices = - delta_elbos.detach().cpu().numpy()

        elif aggregation_method == "batch":
            # Reduce memory by factor of num_batches (num_valid_mutations / batch_size)
            # Note: This will mean that higher memory GPU needs higher RAM because we store larger sample batches before aggregating
            mean_predictions = torch.zeros(len(list_valid_mutations))
            std_predictions = torch.zeros(len(list_valid_mutations))

            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, 'Looping through mutation batches')):
                    x = batch.type(self.dtype).to(self.device)

                    # Simplest: Aggregate mean and std each batch (instead of online per sample)
                    # Reduce memory by factor of num_batches (num_valid_mutations / batch_size)
                    batch_samples = torch.zeros(size=(len(x), 20_000),
                                                dtype=self.dtype)  # Store these on CPU to save GPU memory
                    for j in tqdm(range(num_samples),
                                       'Looping through number of samples for batch #: ' + str(i + 1), mininterval=1):
                        seq_predictions, _, _ = self.all_likelihood_components(x)
                        batch_samples[:, j] = seq_predictions.detach().cpu()

                    # Aggregate mean and std for this batch, this should be negligibly quick
                    mean_predictions[i * batch_size:i * batch_size + len(x)] = batch_samples.mean(dim=1, keepdim=False)
                    std_predictions[i * batch_size:i * batch_size + len(x)] = batch_samples.std(dim=1, keepdim=False)
                    tqdm.tqdm.write('\n')

                delta_elbos = mean_predictions - mean_predictions[0]
                evol_indices = - delta_elbos.detach().cpu().numpy()
        elif aggregation_method == "online":
            # Extension: Completely online, reduce memory by factor of num_samples (20k) with hopefully small overhead
            mean_predictions = torch.zeros(len(list_valid_mutations))
            std_predictions = torch.zeros(len(list_valid_mutations))
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, 'Looping through mutation batches')):
                    x = batch.type(self.dtype).to(self.device)
                    print(
                        f"{datetime.datetime.now()} tmp Peak memory in GB: {getrusage(RUSAGE_SELF).ru_maxrss / 1024 ** 2:.3f}")
                    # Simplest: Aggregate mean and std online per sample
                    online_mean = torch.zeros(len(x), dtype=self.dtype, device=self.device)
                    online_s = torch.zeros(len(x), dtype=self.dtype, device=self.device)

                    # Run this once per batch to speed up remaining loop
                    mu, log_var = self.encoder(x)

                    for j in tqdm(range(num_samples),
                                       'Looping through number of samples for batch #: ' + str(i + 1), mininterval=5):
                        seq_predictions, _, _ = self.all_likelihood_components_z(x, mu, log_var)
                        # Using Welford's method https://stackoverflow.com/a/15638726/10447904
                        # All still on GPU
                        if j == 0:
                            online_mean = seq_predictions.detach()
                            # online_s stays 0
                        else:
                            delta = seq_predictions - online_mean
                            online_mean = online_mean + delta / (j + 1)  # / n in original formula
                            online_s = online_s + delta * (seq_predictions - online_mean)

                    variance = online_s / (num_samples - 1)  # j will end as n-1
                    std = variance.sqrt()
                    # Fill in mean and std arrays for this batch
                    mean_predictions[i * batch_size:i * batch_size + len(x)] = online_mean.detach().cpu()
                    std_predictions[i * batch_size:i * batch_size + len(x)] = std.detach().cpu()
                    tqdm.tqdm.write('\n')

                delta_elbos = mean_predictions - mean_predictions[0]
                evol_indices = - delta_elbos.detach().cpu().numpy()
        else:
            raise ValueError("Invalid aggregation method. Must be one of 'full', 'batch' or 'online'.")

        return list_valid_mutations, evol_indices, mean_predictions[
            0].detach().cpu().numpy(), std_predictions.detach().cpu().numpy()
