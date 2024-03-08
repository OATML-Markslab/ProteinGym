source ../zero_shot_config.sh
source activate proteingym_env

export model_name="mifst" #[mif|mifst]
export model_path="Path to MIFST checkpoints"
export DMS_output_score_folder=${DMS_output_score_folder_subs}/MIFST
export performance_file='MIFST_performance.csv'

srun python3 ../../proteingym/baselines/carp_mif/compute_fitness.py \
            --model_name ${model_name} \
            --model_path ${model_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --DMS_index $SLURM_ARRAY_TASK_ID \
            --output_scores_folder ${DMS_output_score_folder} \
            --performance_file ${performance_file} \
            --structure_data_folder ${DMS_structure_folder}