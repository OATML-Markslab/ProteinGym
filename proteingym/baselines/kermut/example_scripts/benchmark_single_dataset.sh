#!/bin/bash

python src/experiments/proteingym_benchmark.py --multirun \
    dataset=TCRG1_MOUSE_Tsuboyama_2023_1E0L \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut