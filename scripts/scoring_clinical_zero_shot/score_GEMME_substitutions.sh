#!/bin/bash 

source activate proteingym_env

DMS_index="variant index to run (e.g. 1,2,...2525)"
OUTPUT_SCORES_FOLDER="${clinical_output_score_folder_subs}/GEMME"
GEMME_LOCATION="path to GEMME installation"
JET2_LOCATION="path to JET2 installation"
TEMP_FOLDER="./gemme_tmp/"

python ../../proteingym/baselines/gemme/compute_fitness.py --DMS_index=$DMS_index --DMS_reference_file_path=$clinical_reference_file_path_subs \
--DMS_data_folder=$clinical_data_folder_subs --MSA_folder=$clinical_MSA_data_folder_subs --output_scores_folder=$OUTPUT_SCORES_FOLDER \
--GEMME_path=$GEMME_LOCATION --JET_path=$JET2_LOCATION --temp_folder=$TEMP_FOLDER