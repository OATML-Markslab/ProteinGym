source ../zero_shot_config.sh

# Please follow instructions at proteingym/baselines/AIDO to install ragplm environment and other env variables
conda activate ragplm
export AIDO_DATA_PATH=$PROTEINGYM_CACHE/ProteinGYM-DMS-RAG-zeroshot
export HF_HOME=$AIDO_DATA_PATH/HF_cache
export DMS_output_score_folder=$AIDO_DATA_PATH/output

# One example: PTEN_HUMAN_Mighell_2018
python ../../proteingym/baselines/AIDO/compute_fitness.py \
        --dms_ids PTEN_HUMAN_Mighell_2018 \
        --input_data_path $AIDO_DATA_PATH \
        --output_path $DMS_output_score_folder \
        --hf_cache_location $HF_HOME
