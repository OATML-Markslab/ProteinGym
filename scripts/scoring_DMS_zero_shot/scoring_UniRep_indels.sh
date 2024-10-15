#!/bin/bash 

source ../zero_shot_config.sh
source activate protein_fitness_prediction_hsu

export OMP_NUM_THREADS=1

export model_path="path to initial unirep weight checkpoint"
export output_dir=${DMS_data_folder_indels}/UniRep/
export DMS_index="Experiment index to run (E.g. 0,1,2,...,217)"

python ../../proteingym/baselines/unirep/unirep_inference.py \
            --model_path $model_path \
            --data_path $DMS_data_folder_indels \
            --output_dir $output_dir \
            --mapping_path $DMS_reference_file_path_indels \
            --DMS_index $DMS_index \
            --batch_size 32