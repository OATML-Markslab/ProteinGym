#!/bin/bash

source ../zero_shot_config.sh
source activate protssn

# please download and unzip the following files to a folder: https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProteinGym_substitutions_pdb-csv_checked.zip
# eg. data/mutant_example/ProteinGym_substitutions_pdb-csv_checked
export DMS_and_structure_folder="Path to unzipped data folder"

# model checkpoint is at: https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProtSSN.model.tar
# please download and unzip the files to a folder
export model_checkpoint="Path to untarred model checkpoints"

# To ensemble all models use: "k10_h512 k20_h512 k30_h512 k10_h768 k20_h768 k30_h768 k10_h1280 k20_h1280 k30_h1280"
# To use a single model, only reference a single model string via the model_name argument (eg., k20_h512)
export model_name="k10_h512 k20_h512 k30_h512 k10_h768 k20_h768 k30_h768 k10_h1280 k20_h1280 k30_h1280"
export gnn_config=../../proteingym/baselines/protssn/src/config/egnn.yaml
export score_info=../../protssn_scores.csv

export DMS_output_score_folder="Path to folder where all model predictions should be stored"

python ../../proteingym/baselines/protssn/compute_fitness.py \
    --gnn_config ${gnn_config} \
    --gnn_model_dir ${model_checkpoint} \
    --gnn_model_name ${model_name} \
    --use_ensemble \
    --mutant_dataset_dir ${DMS_and_structure_folder} \
    --output_scores_folder ${DMS_output_score_folder} \
    --score_info ${score_info}