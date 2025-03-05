"""Main benchmarking script to evaluate GPs on ProteinGym DMS assays"""

from pathlib import Path

import gpytorch
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm, trange

from kermut.data.data_utils import (
    load_embeddings,
    load_zero_shot,
    prepare_gp_kwargs,
    zero_shot_name_to_col,
)


@hydra.main(version_base=None, config_path="../configs", config_name="proteingym_gpr")
def main(cfg: DictConfig) -> None:
    # Experiment settings
    standardize = cfg.standardize
    split_method = cfg.split_method
    progress_bar = cfg.progress_bar
    sequence_col, target_col = "mutated_sequence", "DMS_score"
    use_multiples = True if cfg.split_method == "fold_rand_multiples" else False

    # Verify input paths
    DMS_data_folder = Path(cfg.DMS_data_folder)
    DMS_reference_file_path = Path(cfg.DMS_reference_file_path)
    output_scores_folder = Path(cfg.output_scores_folder)
    auxiliary_data_folder = Path(cfg.auxiliary_data_folder)

    print(f"Using DMS data folder: {DMS_data_folder.resolve()}")
    print(f"Using DMS reference file: {DMS_reference_file_path.resolve()}")
    print(f"Using output scores folder: {output_scores_folder.resolve()}")
    print(f"Using auxiliary data folder: {auxiliary_data_folder.resolve()}")

    # Model settings
    use_global_kernel = cfg.gp.use_global_kernel
    use_mutation_kernel = cfg.gp.use_mutation_kernel
    use_zero_shot = cfg.gp.use_zero_shot
    use_prior = cfg.gp.use_prior
    noise_prior_scale = cfg.gp.noise_prior_scale if use_prior else None

    # Load dataset
    DMS_idx = cfg.DMS_idx
    df_ref = pd.read_csv(DMS_reference_file_path)
    DMS_id = df_ref.loc[DMS_idx, "DMS_id"]
    wt_sequence = df_ref.loc[DMS_idx, "target_seq"]

    if DMS_id == "BRCA2_HUMAN_Erwood_2022_HEK293T":
        # Disable distance kernel due to sequenc length
        cfg.gp.mutation_kernel.use_distances = False
        cfg.gp.mutation_kernel.model._target_ = "kermut.model.kernel.Kermut_no_d"

    output_path = output_scores_folder / f"{split_method}/kermut/{DMS_id}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        if not cfg.overwrite:
            print(f"Output file already exists: {output_path.resolve()}")
            return
        else:
            print(f"Overwriting existing output file: {output_path.resolve()}")

    if cfg.use_gpu and torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False
        print("CUDA not available, running on CPU.")

    print(
        f"Using {split_method} split on DMS idx {DMS_idx}: {DMS_id}",
        flush=True,
    )

    # Reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Setup current experiment
    gp_kwargs, tokenizer = prepare_gp_kwargs(
        DMS_id=DMS_id, wt_sequence=wt_sequence, cfg=cfg
    )
    df = pd.read_csv(DMS_data_folder / f"{DMS_id}.csv")

    # Prepare output
    df_predictions = pd.DataFrame(columns=["fold", "mutant", "y", "y_pred", "y_var"])

    # Load zero-shot predictions
    df_zero = load_zero_shot(DMS_id, cfg)
    df = pd.merge(left=df, right=df_zero, on="mutant", how="left")
    df = df.reset_index(drop=True)
    zero_shot_col = zero_shot_name_to_col(cfg.gp.zero_shot_method)
    zero_shot_full = torch.tensor(df[zero_shot_col].values, dtype=torch.float32)

    y_full = torch.tensor(df[target_col].values, dtype=torch.float32)

    if use_global_kernel:
        # Load embeddings
        embeddings = load_embeddings(
            dataset=DMS_id, df=df, cfg=cfg, multiples=use_multiples
        )
        embedding_dim = cfg.gp.embedding_dim
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

    unique_folds = df[split_method].unique()
    for i, test_fold in enumerate(tqdm(unique_folds, disable=not progress_bar)):
        # Assign splits
        train_idx = (df[split_method] != test_fold).tolist()
        test_idx = (df[split_method] == test_fold).tolist()
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

            df_pred_fold = pd.DataFrame(
                {
                    "fold": test_fold,
                    "mutant": df.loc[test_idx, "mutant"],
                    "y": y_test,
                    "y_pred": y_preds_mean,
                    "y_var": y_preds_var,
                }
            )
            if df_predictions.empty:
                df_predictions = df_pred_fold
            else:
                df_predictions = pd.concat([df_predictions, df_pred_fold])

    df_predictions.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
