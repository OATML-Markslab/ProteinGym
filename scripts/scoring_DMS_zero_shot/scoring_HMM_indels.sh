#!/bin/bash 

source ../zero_shot_config.sh
source activate proteingym_env

export TEMP_FOLDER="./HMM_temp"
export output_score_folder="${DMS_output_score_folder_indels}/HMM/"
export HMMER_PATH="path to HMMER installation"
export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/HMM/score_hmm.py \
--DMS_reference_file=$DMS_reference_file_path_indels --DMS_folder=$DMS_data_folder_indels \
--DMS_index=$DMS_index \
--hmmer_path=$HMMER_PATH \
--MSA_folder=$DMS_MSA_data_folder \
--output_scores_folder=$output_score_folder --intermediate_outputs_folder=$TEMP_FOLDER
