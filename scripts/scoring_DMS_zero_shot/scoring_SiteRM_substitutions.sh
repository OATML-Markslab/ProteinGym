#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

subsample_n_sequences=100000000
export tree_estimation_rate_matrix_path="proteingym/baselines/SiteRM/lg.txt"
export regularization_rate_matrix_path="proteingym/baselines/SiteRM/lg_with_gaps.txt"
regularization_strength=0.5
evolutionary_time=1.0
num_processes=8
device="cpu"

export output_scores_folder="${DMS_output_score_folder_subs}/SiteRM"

n=216
DMS_indices=$(seq 0 $n | tr '\n' ' ' | sed 's/ $//')
cd ../..
python -m proteingym.baselines.SiteRM.compute_fitness \
            --regularization_strength ${regularization_strength} \
            --subsample_n_sequences ${subsample_n_sequences} \
            --DMS_indices ${DMS_indices} \
            --output_scores_folder ${output_scores_folder} \
            --evolutionary_time ${evolutionary_time} \
            --num_processes ${num_processes} \
            --regularization_rate_matrix_path ${regularization_rate_matrix_path} \
            --tree_estimation_rate_matrix_path ${tree_estimation_rate_matrix_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --DMS_MSA_data_folder ${DMS_MSA_data_folder} \
            --device ${device}
