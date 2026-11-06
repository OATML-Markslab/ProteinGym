# MSA Pairformer with Logan MSAs

This adapter scores the ProteinGym zero-shot substitution benchmark using the
combined Logan/ProteinGym alignments from the MSA Pairformer paper. The submitted
predictions cover all 217 assays and 2,465,767 variants. Official aggregate metrics
are Spearman 0.453, AUC 0.749, MCC 0.355, NDCG 0.784, and top recall 0.224.

Install the baseline dependencies in a CUDA environment with Python 3.11:

```bash
pip install -r proteingym/baselines/msa_pairformer/requirements.txt
```

The requirements pin the model source to commit
`4f291abd876b9f25bc3ccf425780fc653be64da3`, the source snapshot used for the GPU
check, and constrain its dependencies to the tested environment versions.

## Inputs

- Official ProteinGym substitution assay CSVs and `reference_files/DMS_substitutions.csv`.
- The combined Logan/ProteinGym MSA archive, with one `<DMS_id>.a3m` per assay.
- The original NumPy sequence-weight arrays. Weights correspond to sequences after
  ProteinGym MSA preprocessing, in that sequence order. The provided
  `logan_msa_mapping.csv` records the expected MSA and weight filenames.
  Preserve FASTA identifiers and record order: the original loader groups
  repeated identifiers before preprocessing. Weight counts therefore refer to
  retained identifiers, rather than the number of FASTA records in the raw file.
- The public `yakiyama/MSA-Pairformer` checkpoint, supplied as an immutable Hugging
  Face revision or a local snapshot containing `model_cuex.bin`,
  `confind_contact.bin`, and `contact.bin`.

The tested checkpoint revision is `7563e77a87536b5572f91683c39073ef348639a4`.
Its `model_cuex.bin` SHA-256 is
`a52b49ad10022ea52731d296ca34c41d0970f507c6de491e779f4f616798208c`.
The public archive URLs must be filled in before submission. The script requires cached
sequence weights and never automatically recomputes them. Recomputing weights
with different preprocessing or assigning them to a different sequence order can
change the sampled MSA and the predictions.

## Scoring

Run from the ProteinGym repository root:

```bash
python proteingym/baselines/msa_pairformer/compute_fitness.py \
  --dms_mapping reference_files/DMS_substitutions.csv \
  --dms_index 0 \
  --dms-input /path/to/DMS_ProteinGym_substitutions \
  --dms-output /path/to/scores/MSA_Pairformer_Logan \
  --msa-path /path/to/logan_proteingym_combined_msas \
  --msa-weights-folder /path/to/logan_proteingym_combined_weights \
  --msa-mapping proteingym/baselines/msa_pairformer/logan_msa_mapping.csv \
  --model-revision 7563e77a87536b5572f91683c39073ef348639a4 \
  --msa-samples 4096 --seed 0 --numpy-seed 42 --num-cpus 1 --device cuda:0
```

Assay indices are zero-based. `--dms-id` can select an assay by name instead.
The example shell launcher is
`scripts/scoring_DMS_zero_shot/scoring_MSA_Pairformer_Logan_substitutions.sh`.
Each complete assay produces `<DMS_id>.csv` with `mutant` and `mutation_score`,
plus a provenance JSON file containing arguments and input/checkpoint hashes.

Scoring follows `Figure5B_ProteinGym_VEP_ProteinGym_MSAs.py`:

- Keep the query and sample 4,095 homologues with replacement using normalized
  sequence weights and Python random seed 0.
- Mask each mutated query position independently. Use the sum of mutant-minus-WT
  log probabilities for multiple substitutions.
- Use bfloat16 and `torch.compile(dynamic=True)` by default.
- For MSAs longer than 1,024 residues and deeper than 2,048 sequences, use the
  original per-position window: start at `max(0, position_zero_based - 512)` and
  stop at `min(msa_length, start + 1024)`.
- Derive the MSA query's position within the full assay target sequence and retain
  the corresponding mutation offset.

The adapter reuses ProteinGym's `MSA_processing`; its class and weight-calculation
functions are identical to the helpers used by the paper script at ProteinGym
commit `144fe22b07dfaeec2b366f2346203a9838a55b4c`. It skips the unused full-MSA
dense one-hot allocation when loading cached weights; preprocessing and sampling
are unchanged.

## Reproduction check

Add `--validate-inputs` to check an assay's files, query placement, variant keys,
and weight-array validity without importing the model. This check does not prove
that cached weights match the processed MSA sequence order.

For a small GPU check, add `--max-variants 5` to the scoring command and compare
its `mutation_score` values with the archived paper predictions, joined by
`mutant`. Subset results use `<DMS_id>.subset.csv` to distinguish them from full
benchmark submissions. Include an assay with a cropped query and a long assay
when checking the adapter's offsets and window handling. Differences due to
hardware or PyTorch/CUDA versions should be measured and recorded before fixing
a numerical tolerance.

The adapter was checked on the complete 39-residue
`YNZC_BACSU_Tsuboyama_2023_2JVD` assay: 2,300 variants, including 714 single
substitutions and 1,586 multiple substitutions, with 4,096 MSA sequences and seed
0. This check used `--no-compile`, bfloat16, an NVIDIA H200 NVL, PyTorch
2.8.0+cu128, and cuEquivariance 0.8.1. Score Spearman correlation with the archived
paper predictions was 0.999982; mean absolute difference was 0.019273 and maximum
absolute difference was 0.126404. Spearman against experimental measurements was
0.492775 for the archived scores and 0.493194 for the reproduced scores.
This demonstrates close numerical agreement on one short assay. Compiled
execution and numerical reproduction of the remaining assays were not checked.

## Official evaluation

Register `MSA_Pairformer_Logan` in `config.json` with `mutation_score` as its input
column, `mutant` as its key, directionality 1, and model type `MSA`. Run ProteinGym's
merge and `performance_DMS_benchmarks.py --performance_by_depth` scripts.

The single-model evaluation's official bootstrap columns are zero because the
script bootstraps differences from the best included model. Those values do not
estimate absolute uncertainty. Rank 1 in that evaluation identifies its sole
model and is not a full ProteinGym leaderboard rank.

## Reference

MSA Pairformer: https://doi.org/10.1101/2025.08.02.668173

Model and source: https://github.com/yoakiyama/MSA_Pairformer
