import os
import numpy as np
import pandas as pd
import time
import tqdm
from scipy.special import erfinv
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

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
            random_seed
            ):
        
        super().__init__()
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        
        self.seq_len = data.seq_len
        self.alphabet_size = data.alphabet_size
        self.Neff = data.Neff

        self.encoder_parameters=encoder_parameters
        self.decoder_parameters=decoder_parameters

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
        z = torch.exp(0.5*log_var) * eps + mu
        return z

    def KLD_diag_gaussians(self, mu, logvar, p_mu, p_logvar):
        """
        KL divergence between diagonal gaussian with prior diagonal gaussian.
        """
        KLD = 0.5 * (p_logvar - logvar) + 0.5 * (torch.exp(logvar) + torch.pow(mu-p_mu,2)) / (torch.exp(p_logvar)+1e-20) - 0.5

        return torch.sum(KLD)

    def annealing_factor(self, annealing_warm_up, training_step):
        """
        Annealing schedule of KL to focus on reconstruction error in early stages of training
        """
        if training_step < annealing_warm_up:
            return training_step/annealing_warm_up
        else:
            return 1

    def KLD_global_parameters(self):
        """
        KL divergence between the variational distributions and the priors (for the decoder weights).
        """
        KLD_decoder_params = 0.0
        zero_tensor = torch.tensor(0.0).to(self.device) 
        
        for layer_index in range(len(self.decoder.hidden_layers_sizes)):
            for param_type in ['weight','bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['hidden_layers_mean.'+str(layer_index)+'.'+param_type].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['hidden_layers_log_var.'+str(layer_index)+'.'+param_type].flatten(),
                                    zero_tensor,
                                    zero_tensor
                )
                
        for param_type in ['weight','bias']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                        self.decoder.state_dict(keep_vars=True)['last_hidden_layer_'+param_type+'_mean'].flatten(),
                                        self.decoder.state_dict(keep_vars=True)['last_hidden_layer_'+param_type+'_log_var'].flatten(),
                                        zero_tensor,
                                        zero_tensor
                )

        if self.decoder.include_sparsity:
            self.logit_scale_sigma = 4.0
            self.logit_scale_mu = 2.0**0.5 * self.logit_scale_sigma * erfinv(2.0 * self.logit_sparsity_p - 1.0)

            sparsity_mu = torch.tensor(self.logit_scale_mu).to(self.device) 
            sparsity_log_var = torch.log(torch.tensor(self.logit_scale_sigma**2)).to(self.device)
            KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['sparsity_weight_mean'].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['sparsity_weight_log_var'].flatten(),
                                    sparsity_mu,
                                    sparsity_log_var
            )
            
        if self.decoder.convolve_output:
            for param_type in ['weight']:
                KLD_decoder_params += self.KLD_diag_gaussians(
                                    self.decoder.state_dict(keep_vars=True)['output_convolution_mean.'+param_type].flatten(),
                                    self.decoder.state_dict(keep_vars=True)['output_convolution_log_var.'+param_type].flatten(),
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

    def loss_function(self, x_recon_log, x, mu, log_var, kl_latent_scale, kl_global_params_scale, annealing_warm_up, training_step, Neff):
        """
        Returns mean of negative ELBO, reconstruction loss and KL divergence across batch x.
        """
        BCE = F.binary_cross_entropy_with_logits(x_recon_log, x, reduction='sum') / x.shape[0]
        KLD_latent = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())) / x.shape[0]
        if self.decoder.bayesian_decoder:
            KLD_decoder_params_normalized = self.KLD_global_parameters() / Neff
        else:
            KLD_decoder_params_normalized = 0.0
        warm_up_scale = self.annealing_factor(annealing_warm_up,training_step)
        neg_ELBO = BCE + warm_up_scale * (kl_latent_scale * KLD_latent + kl_global_params_scale * KLD_decoder_params_normalized)
        return neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized
    
    def all_likelihood_components(self, x):
        """
        Returns tensors of ELBO, reconstruction loss and KL divergence for each point in batch x.
        """
        mu, log_var = self.encoder(x)
        z = self.sample_latent(mu, log_var)
        recon_x_log = self.decoder(z)

        recon_x_log = recon_x_log.view(-1,self.alphabet_size*self.seq_len)
        x = x.view(-1,self.alphabet_size*self.seq_len)
        
        BCE_batch_tensor = torch.sum(F.binary_cross_entropy_with_logits(recon_x_log, x, reduction='none'),dim=1)
        KLD_batch_tensor = (-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(),dim=1))
        
        ELBO_batch_tensor = -(BCE_batch_tensor + KLD_batch_tensor)

        return ELBO_batch_tensor, BCE_batch_tensor, KLD_batch_tensor

    def train_model(self, data, training_parameters):
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
            filename = training_parameters['training_logs_location']+os.sep+self.model_name+"_losses.csv"
            with open(filename, "a") as logs:
                logs.write("Number of sequences in alignment file:\t"+str(data.num_sequences)+"\n")
                logs.write("Neff:\t"+str(self.Neff)+"\n")
                logs.write("Alignment sequence length:\t"+str(data.seq_len)+"\n")

        optimizer = optim.Adam(self.parameters(), lr=training_parameters['learning_rate'], weight_decay = training_parameters['l2_regularization'])
        
        if training_parameters['use_lr_scheduler']:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=training_parameters['lr_scheduler_step_size'], gamma=training_parameters['lr_scheduler_gamma'])

        if training_parameters['use_validation_set']:
            x_train, x_val, weights_train, weights_val = train_test_split(data.one_hot_encoding, data.weights, test_size=training_parameters['validation_set_pct'], random_state=self.random_seed)
            best_val_loss = float('inf')
            best_model_step_index=0
        else:
            x_train = data.one_hot_encoding
            weights_train = data.weights
            best_val_loss = None
            best_model_step_index = training_parameters['num_training_steps']

        batch_order = np.arange(x_train.shape[0])
        seq_sample_probs = weights_train / np.sum(weights_train)

        self.Neff_training = np.sum(weights_train)
        N_training =  x_train.shape[0]
        
        start = time.time()
        train_loss = 0
        
        for training_step in tqdm.tqdm(range(1,training_parameters['num_training_steps']+1), desc="Training model"):

            batch_index = np.random.choice(batch_order, training_parameters['batch_size'], p=seq_sample_probs).tolist()
            x = torch.tensor(x_train[batch_index], dtype=self.dtype).to(self.device)
            optimizer.zero_grad()

            mu, log_var = self.encoder(x)
            z = self.sample_latent(mu, log_var)
            recon_x_log = self.decoder(z)
            
            neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized = self.loss_function(recon_x_log, x, mu, log_var, training_parameters['kl_latent_scale'], training_parameters['kl_global_params_scale'], training_parameters['annealing_warm_up'], training_step, self.Neff_training)
            
            neg_ELBO.backward()
            optimizer.step()
            
            if training_parameters['use_lr_scheduler']:
                scheduler.step()
            
            if training_step % training_parameters['log_training_freq'] == 0:
                progress = "|Train : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, KLD_decoder_params_norm: {4:.3f}, Time: {5:.2f} |".format(training_step, neg_ELBO, BCE, KLD_latent, KLD_decoder_params_normalized, time.time() - start)
                print(progress)

                if training_parameters['log_training_info']:
                    with open(filename, "a") as logs:
                        logs.write(progress+"\n")

            if training_step % training_parameters['save_model_params_freq']==0:
                self.save(model_checkpoint=training_parameters['model_checkpoint_location']+os.sep+self.model_name+"_step_"+str(training_step),
                            encoder_parameters=self.encoder_parameters,
                            decoder_parameters=self.decoder_parameters,
                            training_parameters=training_parameters)
            
            if training_parameters['use_validation_set'] and training_step % training_parameters['validation_freq'] == 0:
                x_val = torch.tensor(x_val, dtype=self.dtype).to(self.device)
                val_neg_ELBO, val_BCE, val_KLD_latent, val_KLD_global_parameters = self.test_model(x_val, weights_val, training_parameters['batch_size'])

                progress_val = "\t\t\t|Val : Update {0}. Negative ELBO : {1:.3f}, BCE: {2:.3f}, KLD_latent: {3:.3f}, KLD_decoder_params_norm: {4:.3f}, Time: {5:.2f} |".format(training_step, val_neg_ELBO, val_BCE, val_KLD_latent, val_KLD_global_parameters, time.time() - start)
                print(progress_val)
                if training_parameters['log_training_info']:
                    with open(filename, "a") as logs:
                        logs.write(progress_val+"\n")

                if val_neg_ELBO < best_val_loss:
                    best_val_loss = val_neg_ELBO
                    best_model_step_index = training_step
                    self.save(model_checkpoint=training_parameters['model_checkpoint_location']+os.sep+self.model_name+"_best",
                                encoder_parameters=self.encoder_parameters,
                                decoder_parameters=self.decoder_parameters,
                                training_parameters=training_parameters)
                self.train()
    
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
            
            neg_ELBO, BCE, KLD_latent, KLD_global_parameters = self.loss_function(recon_x_log, x, mu, log_var, kl_latent_scale=1.0, kl_global_params_scale=1.0, annealing_warm_up=0, training_step=1, Neff = self.Neff_training) #set annealing factor to 1
            
        return neg_ELBO.item(), BCE.item(), KLD_latent.item(), KLD_global_parameters.item()
        

    def save(self, model_checkpoint, encoder_parameters, decoder_parameters, training_parameters, batch_size=256):
        torch.save({
            'model_state_dict':self.state_dict(),
            'encoder_parameters':encoder_parameters,
            'decoder_parameters':decoder_parameters,
            'training_parameters':training_parameters,
            }, model_checkpoint)
    
    def compute_evol_indices(self, msa_data, list_mutations_location, num_samples, batch_size=256):
        """
        The column in the list_mutations dataframe that contains the mutant(s) for a given variant should be called "mutations"
        """
        #Multiple mutations are to be passed colon-separated
        list_mutations=pd.read_csv(list_mutations_location, header=0)
        if 'var' in list_mutations.columns:
            list['mutations']=list['var']
        elif 'mutant_id' in list_mutations.columns:
            list['mutations']=list['mutant_id']
        else:
            list['mutations']=list['mutant']
        #Remove (multiple) mutations that are invalid
        list_valid_mutations = ['wt']
        list_valid_mutated_sequences = {}
        list_valid_mutated_sequences['wt'] = msa_data.focus_seq_trimmed # first sequence in the list is the wild_type
        for mutation in list_mutations['mutations']:
            individual_substitutions = mutation.split(':')
            mutated_sequence = list(msa_data.focus_seq_trimmed)[:]
            fully_valid_mutation = True
            for mut in individual_substitutions:
                wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
                if pos not in msa_data.uniprot_focus_col_to_wt_aa_dict or msa_data.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa or mut not in msa_data.mutant_to_letter_pos_idx_focus_list:
                    print ("Not a valid mutant: "+mutation)
                    fully_valid_mutation = False
                    break
                else:
                    wt_aa,pos,idx_focus = msa_data.mutant_to_letter_pos_idx_focus_list[mut]
                    mutated_sequence[idx_focus] = mut_aa #perform the corresponding AA substitution
            
            if fully_valid_mutation:
                list_valid_mutations.append(mutation)
                list_valid_mutated_sequences[mutation] = ''.join(mutated_sequence)
        
        #One-hot encoding of mutated sequences
        mutated_sequences_one_hot = np.zeros((len(list_valid_mutations),len(msa_data.focus_cols),len(msa_data.alphabet)))
        for i,mutation in enumerate(list_valid_mutations):
            sequence = list_valid_mutated_sequences[mutation]
            for j,letter in enumerate(sequence):
                if letter in msa_data.aa_dict:
                    k = msa_data.aa_dict[letter]
                    mutated_sequences_one_hot[i,j,k] = 1.0

        mutated_sequences_one_hot = torch.tensor(mutated_sequences_one_hot)
        dataloader = torch.utils.data.DataLoader(mutated_sequences_one_hot, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        prediction_matrix = torch.zeros((len(list_valid_mutations),num_samples))

        with torch.no_grad():
            for i, batch in enumerate(tqdm.tqdm(dataloader, 'Looping through mutation batches')):
                x = batch.type(self.dtype).to(self.device)
                for j in tqdm.tqdm(range(num_samples), 'Looping through number of samples for batch #: '+str(i+1)):
                    seq_predictions, _, _ = self.all_likelihood_components(x)
                    prediction_matrix[i*batch_size:i*batch_size+len(x),j] = seq_predictions
                tqdm.tqdm.write('\n')
            mean_predictions = prediction_matrix.mean(dim=1, keepdim=False)
            std_predictions = prediction_matrix.std(dim=1, keepdim=False)
            delta_elbos = mean_predictions - mean_predictions[0]
            evol_indices =  - delta_elbos.detach().cpu().numpy()

        return list_valid_mutations, evol_indices, mean_predictions[0].detach().cpu().numpy(), std_predictions.detach().cpu().numpy()