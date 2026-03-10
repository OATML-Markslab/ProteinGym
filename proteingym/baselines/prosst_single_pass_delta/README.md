# ProSST Single-Pass Delta

This baseline uses `AI4Protein/ProSST-2048` with a single wild-type forward pass per assay and an additive mutation readout.

Scoring details:
- one forward pass on the wild-type sequence plus quantized structure tokens
- mutation scores are summed across substitutions
- the per-position readout uses a delta that combines:
  - mutant log-probability
  - a background correction
  - an expected log-probability penalty
  - a wild-type confidence bonus

Required benchmark layout:
- `residue_sequence/*.fasta`
- `structure_sequence/2048/*.fasta`
- `substitutions/*.csv`

Example usage:

```bash
python ../../proteingym/baselines/prosst_single_pass_delta/compute_fitness.py \
  --model_name AI4Protein/ProSST-2048 \
  --base_dir /path/to/proteingym_benchmark \
  --reference_file_path ../../reference_files/DMS_substitutions.csv \
  --output_scores_folder /path/to/output_scores/ProSST/ProSST-2048-Single-Pass-Delta
```
