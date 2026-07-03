# TabPFN3-ESMC600M supervised DMS substitutions baseline

This baseline scores ProteinGym supervised DMS substitution assays with TabPFN3
using precomputed ESMC600M-derived protein features.

## External artifacts

This PR does not vendor raw ProteinGym data, derived feature matrices, model
checkpoints, or generated model scores.

Derived ESMC600M feature matrices used for reproduction are available on
Zenodo:

```text
https://zenodo.org/records/21137791
DOI: 10.5281/zenodo.21137791
Concept DOI: 10.5281/zenodo.21137790
File: proteingym_tabpfn3_esmc600m_all_zs_features_20260702.tar.zst
SHA256: 445f1b1dd966abf86799df8acde0d4b0c27d21b8f51fa3e68a64d2281e4b04bd
```

TabPFN3 weights are obtained through the official TabPFN/Prior Labs package and
license-acceptance flow. They are not redistributed in this repository. In
headless environments, set the required TabPFN authentication variables or pass
an approved local checkpoint with `MODEL_PATH`.

## One-assay scoring example

Run from the ProteinGym repository root:

```bash
CSV=/path/to/ProteinGym/cv/A0A140D2T1_ZIKV_Sourisseau_2019.csv \
FEATURE_ROOT=/path/to/proteingym_feature_memmaps \
FOLD_COLUMN=fold_random_5 \
scripts/scoring_DMS_supervised/scoring_TabPFN3_ESMC600M_substitutions.sh
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
model_scores/supervised_substitutions/<cv_scheme>/tabpfn3_esmc600m/<DMS_id>.csv
```

Each CSV contains:

```text
mutant,y,y_pred,fold
```

`y` and `y_pred` are standardized within each held-out fold using the
corresponding training-fold mean and standard deviation.
