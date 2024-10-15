source ../zero_shot_config.sh
source activate proteingym_env

export model_name="carp_640M" #[carp_600k|carp_38M|carp_76M|carp_640M]
export model_path="Path to CARP checkpoints"
export DMS_output_score_folder=${DMS_output_score_folder_subs}/CARP
export performance_file='CARP_640M_performance.csv'

srun python3 ../../proteingym/baselines/carp_mif/compute_fitness.py \
            --model_name ${model_name} \
            --model_path ${model_path} \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --DMS_index $SLURM_ARRAY_TASK_ID \
            --output_scores_folder ${DMS_output_score_folder} \
            --performance_file ${performance_file} 