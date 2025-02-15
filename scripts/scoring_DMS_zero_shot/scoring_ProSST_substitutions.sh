source ../zero_shot_config.sh
source activate prosst

# All models can be found at https://huggingface.co/AI4Protein
# ProSST models: ProSST-20 ProSST-128 ProSST-512 ProSST-1024 ProSST-2048 ProSST-4096 ProSST-3di
# export model_name=AI4Protein/ProSST-20 AI4Protein/ProSST-128 AI4Protein/ProSST-512 AI4Protein/ProSST-1024 AI4Protein/ProSST-2048 AI4Protein/ProSST-4096 AI4Protein/ProSST-3di
export model_hyp=2048
export model_name="AI4Protein/ProSST-$model_hyp"

# the structure pdb files can be found in ProtSSN: https://github.com/tyang816/ProtSSN
# please download and unzip the following files to a folder: https://drive.google.com/file/d/1lSckfPlx7FhzK1FX7EtmmXUOrdiMRerY/view?usp=sharing
export DMS_folder="Path to unzipped data folder"
export DMS_residue_folder="${DMS_folder}/residue_sequence"
export DMS_structure_folder="${DMS_folder}/structure_sequence"
export DMS_data_folder_subs="${DMS_folder}/substitutions"
export DMS_output_score_folder="Path to folder where all model predictions should be stored"

python ../../proteingym/baselines/prosst/compute_fitness.py \
    --model_name ${model_name} \
    --base_dir ${DMS_folder} \
    --output_scores_folder ${DMS_output_score_folder}