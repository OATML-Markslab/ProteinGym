#!/bin/bash 

source ../zero_shot_config.sh

# EVE example
export DMS_index="Experiment index to run (E.g. 0,1,2,...,217)"

python ../../proteingym/baselines/EVE/calc_weights.py \
    --MSA_data_folder ${DMS_MSA_data_folder} \
    --DMS_reference_file_path ${DMS_reference_file_path_subs} \
    --DMS_index "${DMS_index}" \
    --MSA_weights_location ${DMS_MSA_weights_folder} \
    --num_cpus -1 \
    --calc_method evcouplings \
    --threshold_focus_cols_frac_gaps 1 \
    --skip_existing
    #--overwrite

