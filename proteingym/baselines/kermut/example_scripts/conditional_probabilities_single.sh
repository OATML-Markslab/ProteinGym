#!/bin/bash

# Specify path to ProteinMPNN installation 
protein_mpnn_dir=$PROTEINMPNN_DIR

# Majority of assays
DATASET=TCRG1_MOUSE
path_to_PDB="data/structures/pdbs/${DATASET}.pdb"
output_dir="data/conditional_probs/raw_ProteinMPNN_outputs/${DATASET}/proteinmpnn"

if [ ! -d $output_dir ]
then
    mkdir -p $output_dir
fi

python $protein_mpnn_dir/protein_mpnn_run.py \
        --pdb_path $path_to_PDB \
        --save_score 1 \
        --conditional_probs_only 1 \
        --num_seq_per_target 10 \
        --batch_size 1 \
        --out_folder $output_dir \
        --seed 37

# Note: BRCA2_HUMAN is a special case. See example_scripts/conditional_probabilities_all.sh for details.