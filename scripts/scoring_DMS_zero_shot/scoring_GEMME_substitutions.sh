#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export DMS_index="Experiment index to run (e.g. 1,2,...217)"

export GEMME_LOCATION="path to GEMME installation"
export JET2_LOCATION="path to JET2 installation"
export TEMP_FOLDER="./gemme_tmp/"
export DMS_output_score_folder="${DMS_output_score_folder_subs}/GEMME/"

python ../../proteingym/baselines/gemme/compute_fitness.py --DMS_index=$DMS_index --DMS_reference_file_path=$DMS_reference_file_path_subs \
--DMS_data_folder=$DMS_data_folder_subs --MSA_folder=$DMS_MSA_data_folder --output_scores_folder=$DMS_output_score_folder \
--GEMME_path=$GEMME_LOCATION --JET_path=$JET2_LOCATION --temp_folder=$TEMP_FOLDER