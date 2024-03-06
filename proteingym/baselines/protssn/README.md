# ProtSSN

## Environment

Please make sure you have installed **[Anaconda3](https://www.anaconda.com/download)** or **[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)**.
The `torch_geometric` package should be updated to 2.3 or higher.

```shell
conda env create -f protssn_environment.yaml
conda activate protssn
```

## Model Checkpoints

There are 9 distinct models that can be used, separately or via ensembling.
If you only want to only use **one model**, authors recommend using **k20_h512**.

| # Version | # Param | # Link                                                       |
| --------- | ------- | ------------------------------------------------------------ |
| k10_h512  | 148     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k10_h512.pt |
| k10_h768  | 160     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k10_h768.pt |
| k10_h1280 | 184     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k10_h1280.pt |
| k20_h512  | 148     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k20_h512.pt |
| k20_h768  | 160     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k20_h768.pt |
| k20_h1280 | 184     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k20_h1280.pt |
| k30_h512  | 148     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k30_h512.pt |
| k30_h768  | 160     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k30_h768.pt |
| k30_h1280 | 184     | https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k30_h1280.pt |

```shell
mkdir model
cd model
wget https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/model/protssn_k20_h512.pt
```

`ProtSSN.tar` contains all the model checkpoints. The **training records** and **configs** can be found in `model/history` and `model/config`.

```shell
wget https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProtSSN.model.tar
tar -xvf ProtSSN.model.tar
rm ProtSSN.model.tar
```
## Data

The pdb and csv files can be downloaded from https://lianglab.sjtu.edu.cn/files/ProtSSN-2024/ProteinGym_substitutions_pdb-csv_checked.zip.
The files should be arranged as follows:

```
data/proteingym-benchmark
|——DATASET
|——|——Protein1
|——|——|——Protein1.pdb
|——|——|——Protein1.tsv
|——|——Protein2
|——|——...
```

- Because the model requires **structure (PDB)** as input, authors recommend using [**Alphafold**](https://github.com/google-deepmind/alphafold) for folding, or **[ESMFold](https://github.com/facebookresearch/esm)**.
- You may also search the structure for a protein of interest via its Uniprot ID in the **AlphaFold database** (https://alphafold.ebi.ac.uk/).

## Usage

Please refer to the scoring script under `scripts/scoring_DMS_zero_shot/scoring_ProtSSN_substitutions.sh`

## Acknowledgements

For more details about ProtSSN, please refer to the official [ProtSSN GitHub repo](https://github.com/tyang816/ProtSSN).

Please cite the following paper if you use ProtSSN in your work:

```
@article{tan2023protssn,
  title={Semantical and Topological Protein Encoding Toward Enhanced Bioactivity and Thermostability},
  author={Tan, Yang and Zhou, Bingxin and Zheng, Lirong and Fan, Guisheng and Hong, Liang},
  journal={bioRxiv},
  pages={2023--12},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

