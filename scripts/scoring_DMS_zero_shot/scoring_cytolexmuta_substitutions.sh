#!/bin/bash

source ../zero_shot_config.sh
source activate proteingym_env

export DMS_output_score_folder=${DMS_output_score_folder_subs}/cytolexmuta

# These folders should contain ProteinGym per-assay component score CSVs.
export ESMC_SCORE_FOLDER=${DMS_output_score_folder_subs}/ESM_C/600M
export PROSST_SCORE_FOLDER=${DMS_output_score_folder_subs}/ProSST/ProSST-2048
export GEMME_SCORE_FOLDER=${DMS_output_score_folder_subs}/GEMME
export ESM_IF1_SCORE_FOLDER=${DMS_output_score_folder_subs}/ESM-IF1

python ../../proteingym/baselines/cytolexmuta/compute_fitness.py \
    --reference_csv ${DMS_reference_file_path_subs} \
    --dms_dir ${DMS_data_folder_subs} \
    --output_dir ${DMS_output_score_folder} \
    --esmc_scores_dir ${ESMC_SCORE_FOLDER} \
    --prosst_scores_dir ${PROSST_SCORE_FOLDER} \
    --gemme_scores_dir ${GEMME_SCORE_FOLDER} \
    --esm_if1_scores_dir ${ESM_IF1_SCORE_FOLDER}
