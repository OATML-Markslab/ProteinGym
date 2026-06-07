# ProSST PIT Tail Rank

This baseline uses `AI4Protein/ProSST-2048` with a single wild-type forward pass per assay and an additive PIT/tail/rank mutation readout.

Scoring details:
- one forward pass on the wild-type sequence plus quantized structure tokens
- mutation scores are summed across substitutions
- the per-position readout uses an equal-weight sum of:
  - centered lower-tail residual
  - protein-level PIT calibration residual
  - protein-level rank residual
- no learned parameters or tuned scalar weights are introduced

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
  --output_scores_folder /path/to/output_scores/ProSST/ProSST-2048-PIT-Tail-Rank
```
