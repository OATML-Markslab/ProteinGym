#!/bin/bash

##############
# Main results
##############

# Computes Spearman and MSE for all assays and splits. Concatenates with baselines from ProteinGym.
# Output file: results/merged_scores.csv
python src/process_results/merge_score_files.py

# Aggregate results using ProteinGym functionality. Arguments are not necessary. 
# Output files (average and per split) can be found in results/summary/
python src/process_results/performance_DMS_supervised_benchmarks.py \
    --input_scoring_file results/merged_scores.csv \
    --output_performance_file_folder results/summary

##############
# Ablation results
##############

python src/process_results/merge_score_files.py --ablation
python src/process_results/performance_DMS_supervised_benchmarks.py \
    --input_scoring_file results/merged_ablation_scores.csv \
    --output_performance_file_folder results/ablation_summary

