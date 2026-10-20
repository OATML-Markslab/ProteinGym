#!/bin/bash

source ../zero_shot_config.sh
source activate ptulyra_ytx

# PTULyra_ytx's component signal tables (~630 MB, six precomputed signal tables plus the
# within-benchmark `transfer` component) are archived at Zenodo:
#
#     https://doi.org/10.5281/zenodo.22818848
#
# Download PTULyra_ytx_v1.0.zip, unzip it, and point PTULYRA_SIGNALS_DIR at the `signals/`
# directory inside it. Scoring is CPU-only: no GPU is needed, and no component model or
# checkpoint has to be downloaded or run, because the tables are shipped precomputed.

export PTULYRA_SIGNALS_DIR="path to the unzipped signals/ directory from the Zenodo archive"
export output_scores_folder="${DMS_output_score_folder_subs}/PTULyra_ytx"

# All 217 assays in one call. Add --DMS_index with 0-based indices to run a subset.
python ../../proteingym/baselines/PTULyra_ytx/compute_fitness.py \
            --DMS_reference_file_path ${DMS_reference_file_path_subs} \
            --DMS_data_folder ${DMS_data_folder_subs} \
            --signals_dir ${PTULYRA_SIGNALS_DIR} \
            --output_scores_folder ${output_scores_folder}
