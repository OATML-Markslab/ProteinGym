# This file has all general filepaths and directories used in the scoring pipeline. The individual scripts may have 
# additional parameters specific to each method 

# DMS zero-shot parameters

# Folders containing the csvs with the variants for each DMS assay
export DMS_data_folder_subs="Folder containing DMS substitution csvs"
export DMS_data_folder_indels="Folder containing DMS indel csvs"

# Folders containing multiple sequence alignments and MSA weights for all DMS assays
export DMS_reference_file_path_indels=../../reference_files/DMS_indels.csv

# Folders where fitness predictions for baseline models are saved 
export DMS_output_score_folder_subs="folder for DMS substitution scores"

export DMS_output_score_folder_indels="folder for DMS indel scores"

# Folder containing EVE models for each DMS assay
export DMS_EVE_model_folder="folder for DMS assay specific EVE models"

# Folders containing merged score files for each DMS assay
export DMS_merged_score_folder_subs="folder for merged scores for DMS substitutions"
export DMS_merged_score_folder_indels="folder for merged score for DMS indels"

# Folders containing predicted structures for the DMSs 
export DMS_structure_folder="folder containing predicted structures for each DMS assay"


# Clinical parameters 

# Folder containing variant csvs 
export clinical_data_folder_subs="folder containing clinical substitution csvs"
export clinical_data_folder_indels="folder containing clinical indel csvs"

# Folders containing multiple sequence alignments and MSA weights for all clinical datasets
export clinical_MSA_data_folder_subs="folder containing clinical MSA files for substitutions"
export clinical_MSA_data_folder_indels="folder containing clinical MSA files for indels"

# Folder containing MSA weights for all clinical datasets
export clinical_MSA_weights_folder_subs="folder containing clinical MSA weights for substitutions"
export clinical_MSA_weights_folder_indels="folder containing clinical MSA weights for indels"

# reference files for substitution and indel clinical variants 
export clinical_reference_file_path_subs=../../reference_files/clinical_substitutions.csv
export clinical_reference_file_path_indels=../../reference_files/clinical_indels.csv

# Folder where clinical benchmark fitness predictions for baseline models are saved
export clinical_output_score_folder_subs="folder for clinical substitution scores"
export clinical_output_score_folder_indels="folder for clinical indel scores"

# Folder containing EVE models for each clinical variant
export clinical_EVE_model_folder="folder for clinical EVE models"

# Folder containing merged score files for each clinical variant
export clinical_merged_score_folder_subs="folder for merged scores for clinical substitutions"
export clinical_merged_score_folder_indels="folder for merged score for clinical indels"
