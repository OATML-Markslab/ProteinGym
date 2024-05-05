source ../zero_shot_config.sh
source activate deprot

# All models can be found at https://huggingface.co/AI4Protein
# DeProt models: Deprot-20 Deprot-128 Deprot-512 Deprot-1024 Deprot-2048 Deprot-4096 Deprot-3di
# export model_name=AI4Protein/Deprot-20 AI4Protein/Deprot-128 AI4Protein/Deprot-512 AI4Protein/Deprot-1024 AI4Protein/Deprot-2048 AI4Protein/Deprot-4096 AI4Protein/Deprot-3di
export model_hyp=2048
export model_name="AI4Protein/Deprot-$model_hyp"

# the structure pdb files can be found in ProtSSN: https://github.com/tyang816/ProtSSN
# please download and unzip the following files to a folder: https://drive.google.com/file/d/1lSckfPlx7FhzK1FX7EtmmXUOrdiMRerY/view?usp=sharing
export DMS_folder="Path to unzipped data folder"
export DMS_residue_folder="${DMS_folder}/residue_sequence"
export DMS_structure_folder="${DMS_folder}/structure_sequence"
export DMS_data_folder_subs="${DMS_folder}/substitutions"
export DMS_output_score_folder="Path to folder where all model predictions should be stored"

python ../../proteingym/baselines/deprot/compute_fitness.py \
    --model_name ${model_name} \
    --base_dir ${DMS_folder} \
    --output_scores_folder ${DMS_output_score_folder}