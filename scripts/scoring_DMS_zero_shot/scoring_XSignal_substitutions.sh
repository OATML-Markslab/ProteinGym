#!/bin/bash

source ../zero_shot_config.sh

# XSignal uses ProteinGym substitution assay files, ProteinGym MSA files,
# ProteinGym MSA weights, precomputed ProteinGym AF2 structures, and ten
# self-generated intermediate XSignal sensor score folders.
#
# Required folder layout for legacy intermediates:
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureRSALORLike_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureContextRank_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureFamilyConvMLM_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureFamilyVAE_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureStructureEnergy_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PurePottsPLL_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureTTPhysics_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureContextualMSA_v1b/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureSubfamilyMSA_v1/*.csv
# ${XSIGNAL_LEGACY_SCORE_FOLDER}/PureStructureGraph_v1/*.csv

export output_scores_folder="${DMS_output_score_folder_subs}/XSignal"
export DMS_index="${DMS_index:-0}"
export XSIGNAL_LEGACY_SCORE_FOLDER="${XSIGNAL_LEGACY_SCORE_FOLDER:-${DMS_output_score_folder_subs}}"

mkdir -p "${output_scores_folder}"

python ../../../proteingym/baselines/xsignal/compute_fitness.py \
  --DMS_reference_file_path "${DMS_reference_file_path_subs}" \
  --DMS_data_folder "${DMS_data_folder_subs}" \
  --DMS_index "${DMS_index}" \
  --output_scores_folder "${output_scores_folder}" \
  --MSA_folder "${DMS_MSA_data_folder}" \
  --MSA_weights_folder "${DMS_MSA_weights_folder}" \
  --AF2_structures_folder "${DMS_structure_folder}" \
  --legacy_score_folder "${XSIGNAL_LEGACY_SCORE_FOLDER}"
