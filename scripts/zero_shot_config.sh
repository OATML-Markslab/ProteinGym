# This file has all general filepaths and directories used in the scoring pipeline. The individual scripts may have 
# additional parameters specific to each method 
# See here for further details: https://github.com/OATML-Markslab/ProteinGym/issues/63

# Most of these files can be automatically downloaded 
# using the `proteingym.utils.download.download_resources()` function.
# If so, you can set `PROTEINGYM_CACHE` here 
# to the same as `proteingym.utils.download.PROTEINGYM_CACHE`
# By default, this will be: "$HOME/.cache/ProteinGym"    
export PROTEINGYM_CACHE="$HOME/.cache/ProteinGym"    

# ---- DMS zero-shot parameters ---- #

# Folders containing the csvs with the variants for each DMS assay
## DMS benchmark - Substitutions	1.0GB	DMS_ProteinGym_substitutions.zip
export DMS_data_folder_subs="${PROTEINGYM_CACHE}/DMS_ProteinGym_substitutions/" 
## DMS benchmark - Indels	200MB	DMS_ProteinGym_indels.zip
export DMS_data_folder_indels="${PROTEINGYM_CACHE}/DMS_ProteinGym_indels/"

# Folders containing multiple sequence alignments and MSA weights for all DMS assays
## Multiple Sequence Alignments (MSAs) for DMS assays	5.2GB	DMS_msa_files.zip
export DMS_MSA_data_folder="${PROTEINGYM_CACHE}/DMS_msa_files/"
## Redundancy-based sequence weights for DMS assays	200MB	DMS_msa_weights.zip
export DMS_MSA_weights_folder="${PROTEINGYM_CACHE}/DMS_msa_weights/"

# Reference files for substitution and indel assays
## Included in GitHub repo
export DMS_reference_file_path_subs=../../reference_files/DMS_substitutions.csv
## Included in GitHub repo
export DMS_reference_file_path_indels=../../reference_files/DMS_indels.csv

# Folders where fitness predictions for baseline models are saved 
## Zero-shot DMS Model scores - Substitutions	31GB	zero_shot_substitutions_scores.zip
export DMS_output_score_folder_subs="${PROTEINGYM_CACHE}/zero_shot_substitutions_scores/"
## Zero-shot DMS Model scores - Indels	5.2GB	zero_shot_indels_scores.zip
export DMS_output_score_folder_indels="${PROTEINGYM_CACHE}/zero_shot_indels_scores/"

# Folder containing EVE models for each DMS assay
## This is where you would store your local copy of trained EVE models 
## (eg., needed for scoring mutated sequences with EVE or TranceptEVE). 
## Note that given the substantial size of all EVE model checkpoints that is not something that we make available for download by default.
## But the code to train these models from scratch is provided here:
## https://github.com/OATML-Markslab/ProteinGym/blob/main/scripts/scoring_DMS_zero_shot/training_EVE_models.sh
export DMS_EVE_model_folder="${PROTEINGYM_CACHE}/DMS_EVE_models/"

# Folders containing merged score files for each DMS assay
## you can specify any location of your choosing. This is where merged files will be stored if you run 
## the score merge script (https://github.com/OATML-Markslab/ProteinGym/blob/main/scripts/scoring_DMS_zero_shot/merge_all_scores.sh).
## This merge script takes scores from all baselines in DMS_output_score_folder_subs (the ones we provide,
## plus new ones you may have computed yourself), and creates individual files per assay that includes
## scores for all baselines.
export DMS_merged_score_folder_subs="${PROTEINGYM_CACHE}/merged_scores/subs/"
## Same thing for indels
export DMS_merged_score_folder_indels="${PROTEINGYM_CACHE}/merged_scores/indels/"

# Folders containing predicted structures for the DMSs (AF2=AlphaFold2)
## Predicted 3D structures from inverse-folding models	84MB	ProteinGym_AF2_structures.zip
export DMS_structure_folder="${PROTEINGYM_CACHE}/ProteinGym_AF2_structures/"


# ---- Clinical parameters ---- #

# Folder containing variant csvs 
## Clinical benchmark - Substitutions	123MB	clinical_ProteinGym_substitutions.zip
export clinical_data_folder_subs="${PROTEINGYM_CACHE}/clinical_ProteinGym_substitutions/"
## Clinical benchmark - Indels	2.8MB	clinical_ProteinGym_indels.zip
export clinical_data_folder_indels="${PROTEINGYM_CACHE}/clinical_ProteinGym_indels/"

# Folders containing multiple sequence alignments and MSA weights for all clinical datasets
## Clinical MSAs	17.8GB	clinical_msa_files.zip
export clinical_MSA_data_folder="${PROTEINGYM_CACHE}/clinical_msa_files/"
## Clinical MSA weights	250MB	clinical_msa_weights.zip
export clinical_MSA_weights_folder="${PROTEINGYM_CACHE}/clinical_msa_weights/"

# Folder containing MSA weights for all clinical datasets
## use files in clinical_msa_weights.zip
export clinical_MSA_weights_folder_subs="${PROTEINGYM_CACHE}/clinical_msa_weights/subs/"
## use files in clinical_msa_weights.zip
export clinical_MSA_weights_folder_indels="${PROTEINGYM_CACHE}/clinical_msa_weights/indels/"

# reference files for substitution and indel clinical variants 
## Included in GitHub repo
export clinical_reference_file_path_subs=../../reference_files/clinical_substitutions.csv
## Included in GitHub repo
export clinical_reference_file_path_indels=../../reference_files/clinical_indels.csv

# Folder where clinical benchmark fitness predictions for baseline models are saved
## Clinical Model scores - Substitutions	0.9GB	zero_shot_clinical_substitutions_scores.zip
export clinical_output_score_folder_subs="${PROTEINGYM_CACHE}/zero_shot_clinical_substitutions_scores/"
## Clinical Model scores - Indels	0.7GB	zero_shot_clinical_indels_scores.zip
export clinical_output_score_folder_indels="${PROTEINGYM_CACHE}/zero_shot_clinical_indels_scores/"

# Folder containing EVE models for each clinical variant
## same as DMS_EVE_model_folder, but for proteins included in the clinical benchmarks.
export clinical_EVE_model_folder="${PROTEINGYM_CACHE}/clinical_EVE_models/"

# Folder containing merged score files for each clinical variant
##  same as DMS_merged_score_folder_subs, but for the clinical benchmark.
export clinical_merged_score_folder_subs="${PROTEINGYM_CACHE}/merged_scores/subs/"
## same as DMS_merged_score_folder_indels, but for the clinical benchmark.
export clinical_merged_score_folder_indels="${PROTEINGYM_CACHE}/merged_scores/indels/"
