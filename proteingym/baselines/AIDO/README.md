# AIDO.Protein-RAG-16B-proteingym-dms-zeroshot for Protein Zero-Shot Fitness Prediction

### Installation

```bash
conda create -n ragplm python=3.11 -y
conda activate ragplm

conda install anaconda::git-lfs -y
pip install tabulate seaborn deepspeed
pip install git+https://github.com/genbio-ai/ModelGenerator.git
pip install hf_transfer
pip install --upgrade transformers==4.51.0
```

### Download and decompress DMS, MSA and structure files

```bash
#PROTEINGYM_CACHE is defined in scripts/zero_shot_config.sh. Set it to a location on disk where you store large ProteinGym files (need ~34GB total including model checkpoints)
git clone https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS-RAG-zeroshot $PROTEINGYM_CACHE/ProteinGYM-DMS-RAG-zeroshot
export AIDO_DATA_PATH=$PROTEINGYM_CACHE/ProteinGYM-DMS-RAG-zeroshot
cd $AIDO_DATA_PATH
git lfs install
git lfs pull
tar xf dms_data.tar.gz
tar xf struc_data.tar.gz
mkdir output
mkdir HF_cache
export HF_HOME=$AIDO_DATA_PATH/HF_cache
```

### Verify Installation

Test whether the program runs without errors:

```shell
python proteingym/baselines/AIDO/compute_fitness.py --dms_ids PTEN_HUMAN_Mighell_2018 --input_data_path $AIDO_DATA_PATH --output_path $AIDO_DATA_PATH/output --hf_cache_location $HF_HOME
```

Expected result:

```shell
PTEN_HUMAN_Mighell_2018: R=0.5182
```

### Model

The checkpoint is maintained under HuggingFace of genbio-ai: [genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot)

### How to Analyze Your Own Protein

1. Add your protein sequence to the `query.fasta` file. The header format should be:

   ```
   >[name] [start]-[end]
   ```

   The name must match the structure file name in the `struc_data` directory, the MSA file name in the `msa_data` directory, and the DMS table file name in the `dms_data` directory. The `start` and `end` positions indicate that the sequence, MSA, and structure file represent a fragment of the full protein from the DMS table. For example, if you have a protein called `abc` with a length of 345, add the sequence to `query.fasta` as follows:

   ```
   >abc 1-345
   [protein sequence]....
   ```

   Ensure that the following files exist:

   - `abc.pdb` in the `struc_data` directory
   - `abc.txt.gz` in the `msa_data` directory
   - `abc.csv` in the `dms_data` directory

2. Add the protein structure to the `struc_data` directory.

3. Add the MSA to the `msa_data` directory.

4. Add the DMS mutation table to the `dms_data` directory. The table should contain `mutant` and `DMS_score` columns. You can fill 0 values to the `DMS_score` column.

5. Run the inference:

   ```bash
   python compute_fitness.py --dms_ids abc --output_path $AIDO_DATA_PATH/outputs
   ```

### Citation

Please cite AIDO.Protein-RAG-16B-proteingym-dms-zeroshot using the following BibTex code:

```
@inproceedings{sun_mixture_2024,
    title = {Mixture of Experts Enable Efficient and Effective Protein Understanding and Design},
    url = {https://www.biorxiv.org/content/10.1101/2024.11.29.625425v1},
    doi = {10.1101/2024.11.29.625425},
    publisher = {bioRxiv},
    author = {Sun, Ning and Zou, Shuxian and Tao, Tianhua and Mahbub, Sazan and Li, Dian and Zhuang, Yonghao and Wang, Hongyi and Cheng, Xingyi and Song, Le and Xing, Eric P.},
    year = {2024},
    booktitle={NeurIPS 2024 Workshop on AI for New Drug Modalities},
}

@article {Li2024.12.02.626519,
    author = {Li, Pan and Cheng, Xingyi and Song, Le and Xing, Eric},
    title = {Retrieval Augmented Protein Language Models for Protein Structure Prediction},
    url = {https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1},
    year = {2024},
    doi = {10.1101/2024.12.02.626519},
    publisher = {bioRxiv},
    booktitle={NeurIPS 2024 Workshop on Machine Learning in Structural Biology},
}
```

