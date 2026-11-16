#!/bin/bash
# Score the ProteinGym supervised substitutions benchmark with cytomuta-preview.
# Follows the layout of scripts/scoring_DMS_zero_shot/scoring_RITA_substitutions.sh.

source ../zero_shot_config.sh

export output_performance_file_folder=../../benchmarks/DMS_supervised/substitutions
export input_scoring_file=../../benchmarks/raw_score_files/DMS_supervised_substitutions.csv
export DMS_data_folder="path to supervised DMS data"

# 1. per-assay scoring: refit X on the training folds, fuse with the frozen experts
for DMS_file in "${DMS_data_folder}"/*.csv; do
  python ../../proteingym/baselines/cytomuta/compute_fitness.py \
      --dms_file "${DMS_file}" \
      --expert_table "${CYTOMUTA_EXPERT_TABLE}" \
      --student_preds "${CYTOMUTA_STUDENT_PREDS}" \
      --out "${CYTOMUTA_SCORES}/$(basename "${DMS_file}" .csv)_scores.csv"
done

# 2. official aggregation: three schemes, UniProt grouping, category balancing
python ../../proteingym/performance_DMS_supervised_benchmarks.py \
    --input_scoring_file ${input_scoring_file} \
    --output_performance_file_folder ${output_performance_file_folder} \
    --DMS_reference_file_path ${DMS_reference_file_path_subs}
