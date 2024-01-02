source ../zero_shot_config.sh
source activate protssn

# see README for more details
# checked proteingym pdb and csv files: https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProteinGym_substitutions_pdb-csv_checked.zip
# please download and unzip the files to a folder
# eg. data/mutant_example/ProteinGym_substitutions_pdb-csv_checked
export DMS_and_sturcture_folder="path to ProteinGym_substitutions_pdb-csv_checked"

# model checkpoint is at: https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProtSSN.model.tar
# please download and unzip the files to a folder
export model_checkpoint="path to ProtSSN checkpoint"
# eg. (k10_h512 k20_h512 k30_h512 k10_h768 k20_h768 k30_h768 k10_h1280 k20_h1280 k30_h1280)
export model_name="k10_h512 k20_h512 k30_h512 k10_h768 k20_h768 k30_h768 k10_h1280 k20_h1280 k30_h1280"
export DMS_output_score_folder="${DMS_output_score_folder_subs}/ProtSSN"

python ../../proteingym/baselines/protsnn/compute_fitness.py \
    --gnn_model_dir ${model_checkpoint} \
    --gnn_model_name ${model_name} \
    --use_ensemble \
    --mutant_dataset_dir ${DMS_and_sturcture_folder} \
    --output_scores_folder ${DMS_output_score_folder}
