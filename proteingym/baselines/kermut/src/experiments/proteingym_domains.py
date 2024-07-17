"""Script to evaluate GPs on ProteinGym assays with training on 1M and testing on 2M"""

from pathlib import Path

import gpytorch
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import trange

from src.data.data_utils import (
    load_embeddings,
    load_proteingym_dataset,
    load_zero_shot,
    prepare_datasets,
    prepare_gp_kwargs,
    zero_shot_name_to_col,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="proteingym_gpr",
)
def main(cfg: DictConfig) -> None:
    # Experiment settings
    standardize = cfg.standardize
    progress_bar = cfg.progress_bar
    sequence_col, target_col = "mutated_sequence", "DMS_score"
    assert cfg.split_method == "domain"

    # Prepare dataset(s)
    datasets, wt_sequences = prepare_datasets(cfg, use_multiples=True)

    # If datasets is empty
    if not datasets:
        print("All results already exist. Exiting...")
        return

    if cfg.use_gpu and torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    # Model settings
    use_global_kernel = cfg.gp.use_global_kernel
    use_mutation_kernel = cfg.gp.use_mutation_kernel
    use_zero_shot = cfg.gp.use_zero_shot
    use_prior = cfg.gp.use_prior
    noise_prior_scale = cfg.gp.noise_prior_scale if use_prior else None


    for i, (dataset, wt_sequence) in enumerate(zip(datasets, wt_sequences)):
        print(f"--- ({i+1}/{len(datasets)}) {dataset} ---", flush=True)

        # Reproducibility
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Setup current experiment
        model_name = cfg.custom_name if "custom_name" in cfg else cfg.gp.name
        pred_path = Path("results/predictions") / dataset / f"{model_name}_domain.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)

        # Load data
        df = load_proteingym_dataset(dataset, multiples=True)
        gp_kwargs, tokenizer = prepare_gp_kwargs(
            DMS_id=dataset, wt_sequence=wt_sequence, cfg=cfg
        )
        if use_zero_shot:
            zero_shot_method = cfg.gp.zero_shot_method
            zero_shot_col = zero_shot_name_to_col(zero_shot_method)
            df_zero = load_zero_shot(dataset, zero_shot_method)
            df = pd.merge(left=df, right=df_zero, on="mutant")
            df = df.reset_index(drop=True)
            zero_shot_full = torch.tensor(df[zero_shot_col].values, dtype=torch.float32)

        y_full = torch.tensor(df[target_col].values, dtype=torch.float32)

        if use_global_kernel:
            embedding_dim = cfg.gp.embedding_dim
            embedding_type = cfg.gp.embedding_type
            embeddings = load_embeddings(
                dataset,
                df,
                multiples=True,
                embedding_type=embedding_type,
            )
            gp_kwargs["embedding_dim"] = embedding_dim

        if use_mutation_kernel:
            x_seq = df[sequence_col].values
            x_tokens = tokenizer(x_seq).squeeze()

        # Concatenate features for single tensor input (GPyTorch requirement)
        if use_mutation_kernel:
            # [B, 20*seq_len]
            x_full = x_tokens
            if use_zero_shot:
                # [B, 20*seq_len + 1]
                x_full = torch.cat([x_full, zero_shot_full.unsqueeze(-1)], dim=-1)
            if use_global_kernel:
                # [B, 20*seq_len (+ 1) + emb_dim]
                x_full = torch.cat([x_full, embeddings], dim=-1)
        elif use_global_kernel:
            # [B, emb_dim]
            x_full = embeddings
            if use_zero_shot:
                # [B, emb_dim + 1]
                x_full = torch.cat([x_full, zero_shot_full.unsqueeze(-1)], dim=-1)
        else:
            raise NotImplementedError

        # Move to GPU
        if use_cuda:
            x_full = x_full.cuda()
            y_full = y_full.cuda()

        # Sample 80% of single mutants for training.
        train_idx = (
            df[df["n_mutations"] == 1].sample(frac=0.8, random_state=cfg.seed).index
        )
        test_idx = df[~df.index.isin(train_idx)].index
        train_idx = train_idx.tolist()
        test_idx = test_idx.tolist()
        x_train = x_full[train_idx]
        x_test = x_full[test_idx]
        y_train = y_full[train_idx]
        y_test = y_full[test_idx]

        # Standardize given current training set
        if standardize:
            y_train_mean = y_train.mean()
            y_train_std = y_train.std()
            y_train = (y_train - y_train_mean) / y_train_std
            y_test = (y_test - y_train_mean) / y_train_std

        if use_prior:
            noise_prior = gpytorch.priors.HalfCauchyPrior(scale=noise_prior_scale)
        else:
            noise_prior = None

        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
        model = hydra.utils.instantiate(
            cfg.gp.gp_model,
            train_x=x_train,
            train_y=y_train,
            likelihood=likelihood,
            _recursive_=False,
            **gp_kwargs,
        )

        # Move to GPU
        if use_cuda:
            model = model.cuda()
            likelihood = likelihood.cuda()

        # Train model
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model.train()
        likelihood.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.gp.optim.lr)

        for _ in trange(cfg.gp.optim.n_steps, disable=not progress_bar):
            optimizer.zero_grad()
            output = model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate model
        model.eval()
        likelihood.eval()

        with torch.no_grad():
            # Predictive distribution
            y_preds_dist = likelihood(model(x_test))
            y_preds_mean = y_preds_dist.mean.detach().cpu().numpy()
            y_preds_var = y_preds_dist.covariance_matrix.diag().detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()

            df_predictions = pd.DataFrame(
                {
                    "mutant": df.loc[test_idx, "mutant"],
                    "n_mutations": df.loc[test_idx, "n_mutations"],
                    "y": y_test,
                    "y_pred": y_preds_mean,
                    "y_var": y_preds_var,
                }
            )

        df_predictions.to_csv(pred_path, index=False)


if __name__ == "__main__":
    main()
