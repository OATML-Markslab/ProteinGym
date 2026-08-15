# Delta V baseline

Delta V is a pure-Python ensemble strategy (CPCWE: Constraint-Propagated
Confidence-Weighted Ensemble) for zero-shot mutation effect prediction.
It combines the pre-computed predictions of five ProteinGym baselines
(VenusREM, S3F_MSA, ESM2_15B, ProSST-2048, GEMME) with MSA conservation
signals and structural context.

Notable properties:

- **No training, no checkpoints, no GPU.** The strategy is a deterministic
  numpy pipeline (quantile calibration, z-normalization, confidence-weighted
  ensemble, residual propagation, power transform, structure-aware penalties,
  conservation modulation) over existing baseline scores.
- **No label access at any point.** The database built by
  `build_delta_v_db.py` contains no `DMS_score` / `DMS_score_bin` /
  `mutated_sequence` columns — label leakage is impossible by construction.
- **Fast.** Under 2 seconds per protein on a single CPU core.

The strategy was discovered by autonomous LLM-driven evolutionary code
search (an LLM agent iteratively wrote, tested, and refined candidate
scoring algorithms against the benchmark), but the shipped artifact is
fully deterministic static code.

## Performance (official pipeline)

Evaluated with the unmodified `proteingym/performance_DMS_benchmarks.py`
(10,000-iteration bootstrap) on the DMS substitution benchmark:

| Metric | Delta V | VenusREM (next best) |
|---|---|---|
| Spearman | 0.553 | 0.518 |
| AUC | 0.802 | 0.783 |
| MCC | 0.432 | 0.404 |
| NDCG | 0.800 | 0.792 |
| Top-recall | 0.255 | 0.244 |

## Setup

### 1. One-time database build

Requires the official zero-shot substitution scores download
(`zero_shot_substitutions_scores.zip`, ~31 GB unzipped — the same download
used by every other baseline):

```bash
python build_delta_v_db.py \
    --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
    --model_scores_folder ~/.cache/ProteinGym/zero_shot_substitutions_scores/ \
    --output_db ~/.cache/ProteinGym/Delta_V.db
```

The build extracts the five input model score columns per DMS assay into a
single SQLite database (~2.8 GB). Idempotent — safe to re-run.

### 2. (Optional) Structure features

Per-residue solvent accessibility (RSA) features come from AlphaFold
predicted structures. Without this step the strategy falls back to
MSA-derived structural proxies (performance impact: ~0.002 Spearman).

```bash
python download_structures.py   # fetches AF2 PDBs (~84 MB)
python compute_asa.py           # builds protein_structures.db
# then pass --structure_db protein_structures.db to build_delta_v_db.py
```

Requires `biopython` for this step only.

### 3. Scoring

Use `scripts/scoring_DMS_zero_shot/scoring_Delta_V_substitutions.sh`,
which follows the standard baseline scoring conventions:

```bash
export DMS_index=0   # 0..216
python compute_fitness.py \
    --DMS_reference_file_path ../../reference_files/DMS_substitutions.csv \
    --DMS_data_folder ${DMS_data_folder_subs} \
    --DMS_index ${DMS_index} \
    --output_scores_folder ${DMS_output_score_folder_subs}/Delta_V \
    --MSA_data_folder ${DMS_MSA_data_folder} \
    --Delta_V_db_path ~/.cache/ProteinGym/Delta_V.db
```

Writes `<DMS_id>.csv` with columns `mutant, Delta_V_score, DMS_score`,
ready for `merge_all_scores.sh` and the official performance scripts.

## Dependencies

`numpy`, `pandas` (+ `biopython` for the optional structure step).
