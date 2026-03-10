source ../zero_shot_config.sh

export model_name="AI4Protein/ProSST-2048"
export output_scores_folder="${DMS_output_score_folder_subs}/ProSST/ProSST-2048-Single-Pass-Delta"

# This benchmark archive should contain residue_sequence, structure_sequence/2048,
# and substitutions. It is the same archive used by the official ProSST baseline.
export benchmark_folder="Path to unzipped ProSST benchmark folder"

python ../../proteingym/baselines/prosst_single_pass_delta/compute_fitness.py \
    --model_name ${model_name} \
    --base_dir ${benchmark_folder} \
    --reference_file_path ../../reference_files/DMS_substitutions.csv \
    --output_scores_folder ${output_scores_folder}
