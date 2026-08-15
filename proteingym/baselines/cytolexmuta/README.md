# cytolexmuta

cytolexmuta is a zero-shot score-level ensemble for ProteinGym DMS substitutions.

It combines four prediction-only components with fixed weights:

```text
sequence_anchor = 0.5 * ESMC-600M + 0.5 * ProSST-2048
cytolexmuta  = 0.5 * robust_z(sequence_anchor)
              + 0.25 * robust_z(GEMME)
              + 0.25 * robust_z(ESM-IF1)
```

`robust_z` is computed independently within each DMS assay using only that assay prediction distribution. No DMS labels are used for training, weight selection, stacking, or assay-specific calibration.

The wrapper expects precomputed component score files in ProteinGym usual per-assay CSV format and writes one output CSV per assay with columns:

```text
mutant,cytolexmuta
```

Component defaults match the ProteinGym configuration entries:

- `ESMC-600M`: `ESM_C/600M`, column `esmc_600M_score`
- `ProSST-2048`: `ProSST/ProSST-2048`, column `ProSST-2048`
- `GEMME`: `GEMME`, column `GEMME_score`
- `ESM-IF1`: `ESM-IF1`, column `esmif1_ll`

GEMME and JET2 are not redistributed here; use the existing ProteinGym GEMME baseline setup or provide paths to already computed GEMME scores.
