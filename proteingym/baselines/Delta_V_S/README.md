# Delta V-s baseline (supervised)

Delta V-s (MISE-SC: Multi-model Integration with Structural Context and
Conservation) is a pure-Python supervised ensemble strategy for the
ProteinGym supervised DMS substitution benchmark. It combines the
cross-validated predictions of six supervised baselines (Kermut, ProteinNPT,
MSA Transformer embeddings, Tranception embeddings, ESM-1v embeddings,
DeepSequence one-hot) using per-mutation adaptive weights derived from
embedding similarity, structural context (RSA / burial), and MSA
conservation signals.

Notable properties:

- **No training, no checkpoints, no GPU.** The strategy is a deterministic
  numpy/scipy pipeline over existing supervised baseline scores.
- **No label access at any point.** The database built by
  `build_supervised_db.py` contains no `DMS_score` / `DMS_score_bin` /
  `normalized_targets` columns — label leakage is impossible by construction.
- **Fast.** Under 1 second per protein-fold on a single CPU core.

## Performance (official pipeline)

Evaluated with the unmodified `proteingym/merge_supervised.py` +
`proteingym/performance_DMS_supervised_benchmarks.py`
(10,000-iteration bootstrap, `--top_model Kermut`) on the DMS substitution
supervised benchmark (217 assays, 3 CV fold schemes):

| Metric | Delta V-s | Kermut | ProteinNPT |
|---|---|---|---|
| Spearman | 0.680 | 0.657 | 0.619 |
| MSE | 0.681 | 0.605 | 0.687 |

Note on prediction scale: supervised baselines are trained to regress the
normalized targets, so their outputs live on that scale by construction.
Delta V-s is an ensemble whose raw output has arbitrary scale; per-assay
z-scoring of predictions (a label-free affine alignment) maps them onto the
normalized-target scale. Spearman is invariant to this transform; MSE is
reported against the official `normalized_targets`, matching how
`merge_supervised.py` scores every baseline.

## Setup

```bash
bash setup.sh
```

Downloads the official ProteinGym files (reference CSV, DMS assays 1 GB,
supervised model scores 3.3 GB, optional AF2 structures 84 MB) and builds
`data/Delta_V_S.db` (~2.7 GB). Idempotent. Requires: numpy, scipy, pandas,
tqdm (biopython optional, for structure features).

## Scoring

Use `scripts/scoring_DMS_supervised/scoring_Delta_V_S_substitutions.sh`,
or directly:

```bash
python compute_fitness.py \
    --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --DMS_index 0 \
    --fold_scheme fold_random_5 \
    --supervised_scores_folder <DMS_supervised_substitutions_scores/> \
    --output_scores_folder <output/Delta_V_S/> \
    --Delta_V_db_path data/Delta_V_S.db
```

Writes `<fold_scheme>/<DMS_id>.csv` with columns
`mutant, predictions_fitness, labels_fitness` — the standard supervised
baseline score-file format, consumed by `merge_supervised.py` via the
`config.json` entry `"Delta-V-s"` (key: `mutant`).

After scoring all 217 assays × 3 fold schemes, run the repo's standard
supervised pipeline (`merge_all_scores.sh` → `performance_substitutions.sh`)
to aggregate metrics.
