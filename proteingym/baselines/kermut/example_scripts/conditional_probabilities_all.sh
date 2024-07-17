#!/bin/bash

# Specify path to ProteinMPNN installation 
protein_mpnn_dir=$PROTEINMPNN_DIR

# Majority of assays
for DATASET in $(awk -F "\"*,\"*" '{print $3}' $ASSAYS_FILE)
do
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
done

# Special case 
DATASET=BRCA2_HUMAN
for SUFFIX in "_1-1000" "_1001-2085" "_2086-2832"
do
    path_to_PDB="data/structures/pdbs/${DATASET}${SUFFIX}.pdb"
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
done