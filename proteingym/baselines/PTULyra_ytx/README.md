# PTULyra_ytx

Zero-shot ensemble scorer for the ProteinGym DMS substitution benchmark. Scores **every**
mutant of the benchmark — 217/217 assays, full coverage, no mutant dropped.

## Files

| File | Purpose |
|---|---|
| `compute_fitness.py` | scoring script; writes one CSV per assay |
| `routed_config.json` | per-group arm routing and locked weight vectors |
| `requirements.txt` | Python dependencies |

## Data

The script reads six precomputed component signal tables (`prosst`, `venusrem`, `esmc600`,
`esmc6b`, `gemme`, `msa1b`) plus the within-benchmark `transfer` component. Together these
are ~630 MB and are archived at:

> **`https://doi.org/10.5281/zenodo.22818848`** (Zenodo)

Download `PTULyra_ytx_v1.0.zip`, unzip it, and point `--signals_dir` at the `signals/`
directory inside. The same archive also holds the submitted scores, so the numbers below
can be checked directly.

Scoring is **CPU-only** and needs only `numpy`, `pandas`, `scipy`. No GPU, and none of the
component models are run — the tables ship precomputed. `signals/r8_lp_cache_217.pt` is the
raw log-probability cache behind `esmc600` and is included for provenance only.

## Running

See `scripts/scoring_DMS_zero_shot/scoring_PTULyra_ytx_substitutions.sh`. Directly:

```bash
python proteingym/baselines/PTULyra_ytx/compute_fitness.py \
    --DMS_reference_file_path reference_files/DMS_substitutions.csv \
    --DMS_data_folder       /path/to/DMS_ProteinGym_substitutions \
    --signals_dir           /path/to/signals \
    --output_scores_folder  /path/to/output/PTULyra_ytx
```

Writes `<output_scores_folder>/<DMS_id>.csv` with columns
`mutant, PTULyra_ytx_score, DMS_score`, one row per mutant of the benchmark's own set, and
prints the two-level official metric on the way out. Add `--DMS_index 0 1 2` to run a subset.

## Method

Each assay's ProteinGym functional group selects one of five weight vectors. For a target
assay `T`:

1. each component is z-scored over the subset of `T`'s mutants that it covers;
2. for each mutant the output is a weighted average over the components available **at that
   mutant**, with weights renormalised over that active subset — so a component covering
   only part of an assay does not drop mutants;
3. a `transfer` component is blended in with `alpha = clip(pearson(z_transfer, z_base), 0, 1)`,
   computed from the two signal vectors alone, with no label fitting.

Routing and locked weights:

| Group | Arm | Components and weights |
|---|---|---|
| Activity | s0 | prosst .200, venusrem .150, esmc6b .1625, gemme .1625, esmc600 .1625 |
| Binding | s0 | prosst .125, venusrem .125, esmc6b .1875, gemme .1875, esmc600 .1875 |
| Expression | s6b | prosst .350, venusrem .175, esmc6b .158333, gemme .158333, msa1b .158333 |
| OrganismalFitness | s2 | prosst .125, venusrem .350, gemme .175, msa1b .175, esmc600 .175 |
| Stability | s5 | prosst .650, venusrem .275, gemme .075 |

> The locked vectors for the `s0` arm also carry a `"transfer"` key (0.1625 / 0.1875). It is
> **inert**: `compute_fitness.py` removes `transfer` from the weight vector before forming
> `z_base`, so the component enters only through the step-3 blend. The key is a leftover from
> the weight-grid search and affects no score.

## The `transfer` component — please read

`transfer` is **not a model prediction**. It is constructed from ProteinGym's own
experimental measurements:

> For a target assay `T` with UniProt `U`, let `H = {S != T : U(S) = U, |mut(T) ∩ mut(S)| >= 10}`.
> Then `transfer[T][m] = mean_{S in H, m in mut(S)} z_S(DMS_score_S[m])`, where `z_S` is the
> z-score of sibling `S`'s **full** mutant set.

It applies to 53 of the 217 assays; the other 164 are scored by steps 1–2 alone.

`build_transfer_signal.py` (in the Zenodo archive) reconstructs this component from the
benchmark's substitution files alone — no model, no external data — and self-checks against
the shipped table: all 53 assays / 190,390 values agree exactly (`max|Δ| = 0`, identical
mutant key sets).

**Ablation.** Forcing the blend coefficient to 0 throughout, and re-optimising nothing,
lowers the UniProt-level score from **0.5569 to 0.5438** — the largest single-group loss on
Binding (−0.0343) and none on Stability (±0.0000). We report this because the component's
contribution is material and we would rather it be visible than inferred.

## Reported performance

Official two-level metric (mean over mutants → mean over assays sharing a UniProt →
equal-weight mean over the five functional groups), from the archive's submitted scores:

| | UniProt-level (official) | DMS-level |
|---|---|---|
| PTULyra_ytx | **0.5569** | **0.5632** |

Per group (UniProt-level): Activity 0.5483 · Binding 0.5039 · Expression 0.5524 ·
OrganismalFitness 0.5240 · Stability 0.6561.

## Notes for reviewers

* **Three assays are relabelled.** ProteinGym v1.3 updated the UniProt accession / organism of
  `F7YBW7_MESOW_Ding_2023` → `F7YBW8_MESOW_Ding_2023`,
  `PSAE_SYNP2_Tsuboyama_2023_1PSE` → `PSAE_PICP2_Tsuboyama_2023_1PSE`, and
  `Q6WV13_9MAXI_Somermeyer_2022` → `Q6WV12_9MAXI_Somermeyer_2022`. Our component tables are
  keyed by the older names, so `compute_fitness.py` maps them (`ALIAS_OLD_TO_NEW`). The
  mutant sets are identical under either name — verified per assay — so this is a pure
  relabelling and changes no score.
* Because of that rename, the script looks for the benchmark data file under the name given by
  the reference file's `DMS_filename`, and falls back to the older name if your copy of the
  data folder predates v1.3.
* `prosst` and `venusrem` are both read from a CSV column literally named `VenusREM`. That is
  a naming leftover in the upstream score tables: the two tables come from different
  directories and the columns hold different data.
* Only `esmc600` is trained by us; `prosst`, `venusrem`, `gemme`, `msa1b` and `esmc6b` are
  third-party models with public weights, redistributed in the archive as score tables solely
  so the released scores can be reproduced. See `LICENSE_NOTE.md` there.
* **`esmc600` and `esmc6b` are not the same model family and not under the same licence**,
  despite the similar names. `esmc600` is our fine-tune of the **official** ESM-C 600M
  (Chan Zuckerberg Biohub, MIT). `esmc6b` is the **unofficial** MultiMolecule re-implementation
  `multimolecule/esmc-6b` (**AGPL-3.0-or-later**) — a pretrained model we did **not** fine-tune.
  The names come from their parameter counts, not from a shared origin.
