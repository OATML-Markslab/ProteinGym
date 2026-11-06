# MSA Pairformer with Logan MSAs

Zero-shot substitution scoring with combined Logan/ProteinGym alignments.
[Paper](https://doi.org/10.1101/2025.08.02.668173) ·
[Model source](https://github.com/yoakiyama/MSA_Pairformer)

## Data

Download [MSA_Pairformer_data.zip](https://zenodo.org/records/24022898/files/MSA_Pairformer_data.zip?download=1)
from [Zenodo version 24022898](https://doi.org/10.5281/zenodo.24022898). After extraction,
the ProteinGym assets are under `MSA_Pairformer_data/Figure5_CASP_ProteinGym/ProteinGym/`:

- `logan_proteingym_combined_msas/`: 217 assay alignments.
- `logan_proteingym_combined_weights/`: 192 sequence-weight arrays.
- `MSA_Pairformer_Logan_scores.zip`: archived predictions for all 217 assays and 2,465,767 variants.
- `MSA_Pairformer_Logan_evaluation.zip`: official metrics, validation, and reproduction reports.

Extract the scores ZIP into the model-score directory used by ProteinGym's merge
script. Use the official substitution assay CSVs and `reference_files/DMS_substitutions.csv`.
Keep MSA identifiers and record order unchanged; the cached weights follow ProteinGym preprocessing.

## Scoring

Use Python 3.11 and CUDA. From the ProteinGym repository root:

```bash
pip install -r proteingym/baselines/msa_pairformer/requirements.txt

LOGAN_DATA=/path/to/MSA_Pairformer_data/Figure5_CASP_ProteinGym/ProteinGym
python proteingym/baselines/msa_pairformer/compute_fitness.py \
  --dms_mapping reference_files/DMS_substitutions.csv \
  --dms-id YNZC_BACSU_Tsuboyama_2023_2JVD \
  --dms-input /path/to/DMS_ProteinGym_substitutions \
  --dms-output /path/to/scores/MSA_Pairformer_Logan \
  --msa-path "$LOGAN_DATA/logan_proteingym_combined_msas" \
  --msa-weights-folder "$LOGAN_DATA/logan_proteingym_combined_weights" \
  --msa-mapping proteingym/baselines/msa_pairformer/logan_msa_mapping.csv \
  --model-revision 7563e77a87536b5572f91683c39073ef348639a4 \
  --msa-samples 4096 --seed 0 --numpy-seed 42 --num-cpus 1 \
  --device cuda:0 --no-compile
```

Requirements pin the tested model source and dependency versions. The adapter
keeps the query, samples homologues with replacement, and sums masked-marginal
log-odds across substitutions. Query offsets and long-sequence windows follow
the paper's scoring script. Outputs are `<DMS_id>.csv` (`mutant`, `mutation_score`)
and a provenance JSON file.

Use `--dms_index` for a zero-based assay index. Omit `--no-compile` to enable the
default `torch.compile(dynamic=True)`. `--validate-inputs` checks inputs without
inference; `--max-variants N` writes a `.subset.csv` for a small check.
The shell example is `scripts/scoring_DMS_zero_shot/scoring_MSA_Pairformer_Logan_substitutions.sh`.

## Validation

Official evaluation of the archived predictions:

| Spearman | AUC | MCC | NDCG | Top recall |
|---|---|---|---|---|
| 0.453 | 0.749 | 0.355 | 0.784 | 0.224 |

The complete short YNZC assay (39 residues, 2,300 variants) was checked on H200
with bfloat16 eager inference: score Spearman **0.999982** against archived scores,
mean absolute difference **0.019273**, maximum **0.126404**. This checks one assay;
compiled execution and full-benchmark inference were not rerun.

The evaluation archive explains the single-model bootstrap columns: their zeros
come from self-comparison and do not estimate absolute uncertainty.
