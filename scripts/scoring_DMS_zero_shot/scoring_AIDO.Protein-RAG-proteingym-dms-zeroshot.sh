source ../zero_shot_config.sh
# Please follow proteingym/baselines/AIDO.Protein-RAG-proteingym-dms-zeroshot/README.md to install ragplm environment
conda activate ragplm

# Model can be found at https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot
# Data (MSA/Structure) can be found at https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS-RAG-zeroshot

# Following the commands to download the MSA/Structure data to the specific directories
# OLD_PWD=$PWD
# cd ../../proteingym/baselines/AIDO.Protein-RAG-proteingym-dms-zeroshot/
# git clone https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS-RAG-zeroshot
# cd ProteinGYM-DMS-RAG-zeroshot
# mv msa_data dms_data.tar.gz struc_data.tar.gz query.fasta ..
# cd ..
# tar xf dms_data.tar.gz
# tar xf struc_data.tar.gz
# cd $OLD_PWD

export DMS_output_score_folder="Path to folder where all model predictions should be stored"

# One example: PTEN_HUMAN_Mighell_2018
python ../../proteingym/baselines/prosst/compute_fitness.py \
    --dms_ids PTEN_HUMAN_Mighell_2018 \
    --output_path ${DMS_output_score_folder}
