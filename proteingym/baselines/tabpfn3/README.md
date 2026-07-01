# TabPFN3 supervised DMS substitutions baseline

This baseline scores ProteinGym supervised DMS substitution assays with TabPFN3
using precomputed 960-dimensional ESMC-derived feature matrices. The scoring
script consumes the released feature matrices directly; it does not regenerate
ESMC embeddings from raw sequences.

Reference:

```text
Guan, D., Zhang, L., Wijesinghe, A., Zhu, A., Zhao, H., Power, H.,
Ahmed, F. H., Warden, A., Ong, C. S., & Steinberg, D. M. (2026).
Can Tabular In-Context Learners Generalize to Biomolecular Property Prediction?
arXiv:2606.31126. https://arxiv.org/abs/2606.31126
```

## External artifacts

This PR does not vendor raw ProteinGym data, derived feature matrices, model
checkpoints, or generated model scores.

Derived feature matrices used for reproduction are available on Zenodo. Each
`X_float16.dat` file is a raw float16 matrix with shape
`(rows_in_matching_cv_csv, 960)`:

```text
https://zenodo.org/records/21093006
DOI: 10.5281/zenodo.21093006
File: proteingym_tabicl_tabpfn3_official_singles_features_20260701.tar.zst
SHA256: e46849123c6b45a77f935759b78303032ede5023c8f0304269750b21c35bb73a
```

TabPFN3 weights are obtained through the official TabPFN/Prior Labs package and
license-acceptance flow. They are not redistributed in this repository. In
headless environments, set the required TabPFN authentication variables or pass
an approved local checkpoint with `MODEL_PATH`.

## Feature provenance note

This submission should be interpreted as a learner-plus-representation baseline:
TabPFN3 fitted on fixed 960-dimensional ESMC-derived matrices. The released
feature artifact is required to reproduce the submitted scores exactly. A
variant that regenerates new ESMC600M/ESMC6B, mutation-delta, or token-site
features should be submitted as a separate baseline because it changes the
representation rather than only the TabPFN3 scoring wrapper.

## One-assay scoring example

Run from the ProteinGym repository root:

```bash
CSV=/path/to/ProteinGym/cv/A0A140D2T1_ZIKV_Sourisseau_2019.csv \
FEATURE_ROOT=/path/to/proteingym_feature_memmaps \
FOLD_COLUMN=fold_random_5 \
scripts/scoring_DMS_supervised/scoring_TabPFN3_substitutions.sh
```

Useful optional environment variables:

```text
DEVICE=auto
N_ESTIMATORS=8
MODEL_PATH=/path/to/local/tabpfn3/checkpoint
PREDICTION_CHUNK_SIZE=1024
FORCE=1
```

## Submitted score format

The complete score archive linked from the associated `new model` issue uses:

```text
model_scores/supervised_substitutions/<cv_scheme>/tabpfn3/<DMS_id>.csv
```

Each CSV contains:

```text
mutant,y,y_pred,fold
```

`y` and `y_pred` are standardized within each held-out fold using the
corresponding training-fold mean and standard deviation.
