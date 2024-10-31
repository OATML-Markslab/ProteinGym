source ../zero_shot_config.sh
source activate protrem

# the structure pdb files can be found in ProtSSN: https://github.com/tyang816/ProtSSN
# ProteinGym a2m homology sequences (EVCouplings): https://huggingface.co/datasets/tyang816/ProtREM/blob/main/aa_seq_aln_a2m.tar.gz. 
# The original a2m files are downloaded at [ProteinGym](https://github.com/OATML-Markslab/ProteinGym).
export DMS_folder="Path to unzipped data folder"
export DMS_residue_folder="${DMS_folder}/aa_seq"
export DMS_residue_alignment_folder="${DMS_folder}/aa_seq_aln_a2m"
export DMS_structure_folder="${DMS_folder}/struc_seq"
export DMS_data_folder_subs="${DMS_folder}/substitutions"
export DMS_output_score_folder="Path to folder where all model predictions should be stored"

python ../../proteingym/baselines/protrem/compute_fitness.py \
    --base_dir ${DMS_folder} \
    --output_scores_folder ${DMS_output_score_folder}
