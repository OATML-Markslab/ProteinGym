# cytolexmuta

cytolexmuta is a zero-shot map-fusion baseline for ProteinGym DMS substitutions.

It treats four prediction-only components as complementary protein energy maps: sequence-language plausibility, ProSST structure-aware likelihood, evolutionary coupling, and inverse-folding structural compatibility. The wrapper projects those maps into a shared assay-local coordinate system and follows a fixed convex path through the aligned map family.

The exact deterministic scoring rule is implemented in `compute_fitness.py`. Normalization is computed independently within each DMS assay using only that assay prediction distribution. No DMS labels are used for training, weight selection, stacking, or assay-specific calibration.

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
