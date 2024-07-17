#!/bin/bash
# Note: Assumes esm2_t33_650M_UR50D.pt is placed in models/directory.

# Generate embeddings for all single mutants (main benchmark). 
# Note: Requires approximately 1.7 TB of disk space.
python src/data/extract_esm2_embeddings.py \
    --dataset=all\
    --toks_per_batch=131072\
    --which="singles"


# Generate embeddings for subset of multi-mutant assays.
python src/data/extract_esm2_embeddings.py \
    --dataset=all\
    --toks_per_batch=131072\
    --which="multiples"