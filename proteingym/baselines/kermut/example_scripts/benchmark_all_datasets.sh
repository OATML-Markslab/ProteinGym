#!/bin/bash

# Majority of datasets (215/217) can be evaluated on a GPU with 48GB VRAM
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=benchmark \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut

# BRCA2_HUMAN_Erwood_2022_HEK293T cannot make use of the distance kernel. 
# See Appendix for details.
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=BRCA2_HUMAN_Erwood_2022_HEK293T \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut_no_d \
    ++custom_name=kermut

# Remainder is evaluated on CPU
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=large \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut \
    use_gpu=false

