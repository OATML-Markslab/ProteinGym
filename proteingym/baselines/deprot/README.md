# ProtSSN

## Environment

Please make sure you have installed **[Anaconda3](https://www.anaconda.com/download)** or **[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)**.
The `torch_geometric` package should be updated to 2.3 or higher.

```shell
conda env create -f deprot_environment.yaml
conda activate deprot
```

## Model Checkpoints

You can use different settings of models in [huggingface](https://huggingface.co/AI4Protein):

- AI4Protein/Deprot-20
- AI4Protein/Deprot-128
- AI4Protein/Deprot-512
- AI4Protein/Deprot-1024
- AI4Protein/Deprot-2048
- AI4Protein/Deprot-4096
- AI4Protein/Deprot-3di

## Data

the structure pdb files can be found in ProtSSN: https://github.com/tyang816/ProtSSN

please download and unzip the following files to a folder: https://drive.google.com/file/d/1lSckfPlx7FhzK1FX7EtmmXUOrdiMRerY/view?usp=sharing

The files should be arranged as follows:

```
data/proteingym-benchmark
|——residue_sequence
|——|——Protein1.fasta
|——|——...
|——structure_sequence (K=x)
|——|——20
|——|——|——Protein1.fasta
|——|——128
|——|——...
|——substitutions (mutant files)
|——|——Protein1.csv
```

- Because the model requires **structure (PDB)** as input, authors recommend using [**Alphafold**](https://github.com/google-deepmind/alphafold) for folding, or **[ESMFold](https://github.com/facebookresearch/esm)**.
- You may also search the structure for a protein of interest via its Uniprot ID in the **AlphaFold database** (https://alphafold.ebi.ac.uk/).

## Usage

Please refer to the scoring script under `scripts/scoring_DMS_zero_shot/scoring_DeProt_substitutions.sh`

## Acknowledgements

For more details about DeProt, please refer to the official [DeProt GitHub repo](https://github.com/ginnm/DeProt/).

Please cite the following paper if you use DeProt in your work:

```
@article{li2024deprot,
  title={DeProt: A protein language model with quantizied structure and disentangled attention},
  author={Li, Mingchen and Tan, Yang and Zhong, Bozitao and Zhou, Ziyi and Yu, Huiqun and Ma, Xinzhu and Ouyang, Wanli and Hong, Liang and Zhou, Bingxin and Tan, Pan},
  journal={bioRxiv},
  pages={2024--04},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}

@article{tan2023protssn,
  title={Semantical and Topological Protein Encoding Toward Enhanced Bioactivity and Thermostability},
  author={Tan, Yang and Zhou, Bingxin and Zheng, Lirong and Fan, Guisheng and Hong, Liang},
  journal={bioRxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```
