# ESMC-TabICL

Supervised variant-effect predictor for the ProteinGym substitution benchmark: **ESMC-600M
embeddings + zero-shot scores → TabICLv2 regression**. No GP, no structure, no MSA.

## Method

For each variant, `compute_scores.py` builds a 1164-dim feature vector:

| feature | dims | description |
|---|---|---|
| embedding | 1152 | ESMC-600M `hidden_states[-2]` (second-to-last layer), mean-pooled over residues |
| pseudo-LL | 1 | approximate pseudo-log-likelihood — one **unmasked** forward pass, `Σ_i log P(s_i \| s)` |
| masked wt-marginal | 11 | **masked** mutation-effect score (mask each mutated position in the wild type, read the blind prediction `log P[mut] − log P[wt]`), aggregated as deciles over the mutated positions |

These features go to a [TabICLv2](https://github.com/soda-inria/tabicl) regressor under ProteinGym's
CV protocol: for each split, train on 4 folds and predict the held-out fold, with targets
standardized per fold. Output: one score file per assay per split,
`<output_dir>/<cv_scheme>/<DMS_id>.csv` with columns `[mutant, y, y_pred, fold]`.

The masked wt-marginal is the ESM "wt-marginals" zero-shot score (Meier et al., 2021) in its masked
form. The decile aggregation is identity for single mutants (it up-weights this strong scalar) and
generalizes to multi-mutation assays.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "esm@git+https://github.com/Biohub/esm.git@main"   # provides biohub/ESMC-600M + esmc arch
pip install tabicl scipy pandas numpy
```
`biohub/ESMC-600M` (≈2.4 GB) downloads from the HuggingFace Hub on first use. Both ESMC-600M and
TabICLv2 are open source.

## Usage

```bash
# all 217 assays (loads the model once, loops assays)
python compute_scores.py \
    --dms_reference  ../../../reference_files/DMS_substitutions.csv \
    --dms_folder     <path to cv_folds_singles_substitutions> \
    --output_dir     <scores_out>

# a single assay (row index in the reference file), e.g. for parallel scoring
python compute_scores.py ... --dms_index 13
```

`--dms_folder` is the per-assay CV-fold CSVs from ProteinGym (columns `mutant`,
`mutated_sequence`, `DMS_score`, `fold_random_5`, `fold_modulo_5`, `fold_contiguous_5`).

See `score_all.sh` for the exact invocation used to produce the submitted scores.

## Notes
- Each TabICL fit peaks ~80 GB GPU memory at n=10k; folds with >10k training rows are seeded-
  subsampled to 10k (`TRAIN_CAP`). The held-out fold is never subsampled (all mutants are scored).
- fp32 forward passes; bf16 gives essentially identical results (embeddings cosine 0.99997).
