#!/bin/bash 

source ../zero_shot_config.sh
source activate s3f

export S2F_model_checkpoint="path to S2F model checkpoint" #https://zenodo.org/records/14257708
export S2F_output_scores_folder="${DMS_output_score_folder_subs}/S2F"
export S2F_config=../../proteingym/baselines/S3F/config/evaluate/s2f.yaml
export DMS_index="Experiment index to run (e.g. 1,2,...217)"

python ../../proteingym/baselines/S3F/compute_fitness.py \
        -c $S2F_config \
        --datadir ${DMS_data_folder_subs} \
        --ProteinGym_reference_file ${DMS_reference_file_path_subs} \
        --structdir ${DMS_structure_folder} \
        --ckpt $S2F_model_checkpoint \
        --output_scores_folder $S2F_output_scores_folder \
        --DMS_index $DMS_index