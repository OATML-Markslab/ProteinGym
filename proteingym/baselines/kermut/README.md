# Kermut

This is the official code repository for the paper _Kermut: Composite kernel regression for protein variant effects_ ([preprint](https://www.biorxiv.org/content/10.1101/2024.05.28.596219v1)).


## Overview
Kermut is a carefully constructed Gaussian process which obtains state-of-the-art performance for protein property prediction on ProteinGym's supervised substitution benchmark while providing meaningful overall calibration.

### Results on ProteinGym 

Kermut as been applied to the supervised ProteinGym substitution benchmark ([paper](https://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html), [repo](https://github.com/OATML-Markslab/ProteinGym)) and reaches state of the art performance across splits.

Below is a table showing the aggregated Spearman scores per cross validation scheme. In March 2024, the modulo and contiguous splits were updated (see [GitHub issue](https://github.com/OATML-Markslab/ProteinNPT/issues/13)), and we show the performance on the corrected splits as well as the old splits to compare to ProteinNPT.


| Model | Average  | Random | Modulo | Contiguous |
|------------|------------------|--------------------------------|--------------------------------|-------------------------------------|
| Kermut     | 0.655            | 0.744                          | 0.631                          | 0.591                              |
| Kermut (old splits)     | 0.662            | 0.744                          | 0.633                          | 0.610                              |
|ProteinNPT (old splits)|  0.613 | 0.730 | 0.564 | 0.547 |

Below is a table showing the aggregated Spearman scores per functional category.

| Model | Activity | Binding | Expression | Organismal Fitness | Stability |
|------------|-------------------|------------------|---------------------|---------------------------|-------------------|
| Kermut     | 0.605             | 0.614            | 0.662               | 0.580                     | 0.817             |
| Kermut (old splits)     | 0.606             | 0.630            | 0.672               | 0.581                     | 0.824             |
| ProteinNPT (old splits) | 0.547 | 0.470 | 0.584 | 0.493 | 0.749 |


## Installation

After cloning the repository, the environment can be installed via

```bash
conda env create -f environment.yml
conda activate kermut_env
pip install -e .
```
### Directory structure
Two paths are defined throughout, `PROTEINGYM_DIR` and `KERMUT_DIR`. 
- `PROTEINGYM_DIR` is the base directory for ProteinGym and is where data files are assumed to be. E.g., reference files in `PROTEINGY_DIR/reference_files` and CV-files in  `PROTEINGYM_DIR/data/substitutions_singles`.
- `KERMUT_DIR` is the base directory for Kermut and contains source code as well as additional data files such as pre-computed embeddings (e.g., ESM-2 embeddings in `KERMUT_DIR/data/embeddings/substitutions_singles/ESM2`)

The main benchmarking script is `kermut/proteingym_benchmark.py` and should be run from the ProteinGym base directory.

## Reproduce results [with precomputed embeddings]


### Download pre-computed embeddings

All outputs from the preprocessing procedure (i.e., precomputed ESM-2 embeddings, conditional amino acid distributions, processed coordinate files, and zero-shot scores from ESM-2) can be readily accessed via a zip-archive hosted by the Electronic Research Data Archive (ERDA) by the University of Copenhagen using the following [link](https://sid.erda.dk/sharelink/c2EWrbGSCV). The file takes up approximately 4GB. To download and extract the data, run the following (from the Kermut base directory):

```bash
# Download zip archive
curl -o kermut_data.zip https://sid.erda.dk/share_redirect/c2EWrbGSCV/kermut_data.zip
# Unpack and remove zip archive
unzip kermut_data.zip && rm kermut_data.zip
```

### Compute fitness 

To compute the fitness for the 0th assay in the reference file, run the following:

```bash
cd path_to_proteingym_base
python proteingym/baselines/kermut/kermut/proteingym_benchmark.py \
    DMS_idx=0 \
    split_method=fold_random_5
```

Per-mutant predictions will be placed in: 
`model_scores/supervised_substitutions/fold_random_5/kermut/assay_name_for_idx_0.csv`



## Reproduce results [from scratch]
To run Kermut from scratch without precomputed resources, e.g., for a new dataset, the ProteinMPNN repository must be installed. Additionally, the ESM-2 650M parameter model must be saved locally: 
### ProteinMPNN
Kermut leverages structure-conditioned amino acid distributions from [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187), which can has to installed from the [official repository](https://github.com/dauparas/ProteinMPNN). An environment variable pointing to the installation location can then be set for later use:

```bash
export PROTEINMPNN_DIR=<path-to-ProteinMPNN-installation>
```

### ESM-2 models 
Kermut leverages protein sequence embeddings and zero-shot scores extracted from ESM-2 ([paper](https://www.science.org/doi/10.1126/science.ade2574), [repo](https://github.com/facebookresearch/esm)). We concretely use the 650M parameter model (`esm2_t33_650M_UR50D`). While the ESM repository is installed above /via the yml-file), the model weights should be downloaded separately and placed in the `models` directory:

```bash
curl -o models/esm2_t33_650M_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

curl -o models/esm2_t33_650M_UR50D-contact-regression.pt https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```

### Sequence embeddings
ESM-2 embeddings can be generated by calling:
```bash
python kermut/data/extract_esm2_embeddings.py \
    --DMS_idx=0 \
    --which=singles
```
For each assay, an `h5` file is generated which contains all embeddings for all variants. Since the Kermut GP only uses the mean-pooled embeddings, only these are stored. To obtain per AA embeddings, the extraction script can be altered by removing the mean operator.
The embeddings are located in `kermut/data/embeddings/substitutions_singles/ESM2` (for the single-mutant assays).


### Structure-conditioned amino acid distributions

The structure-conditioned amino acid distributions for all residues and assays, can be computed with ProteinMPNN via

```
bash scripts/conditional_probabilities_all.sh
```
This generates per-assay directories in `data/conditional_probs/raw_ProteinMPNN_outputs`. After this, postprocessing for easier access is performed via
```bash
python kermut/data/extract_ProteinMPNN_probs.py
```
This generates per-assay `npy`-files in `data/conditional_probs/ProteinMPNN`.

### 3D coordinates
Lastly, the 3D coordinates can be extracted from each PDB file via
```bash
python kermut/data/extract_3d_coords.py
```
This saves `npy`-files for each assay in `data/structures/coords`. 

### Optional: Zero-shot scores
If not relying on pre-computed zero-shot scores from ProteinGym, they can be computed for ESM-2 via:
```bash
python kermut/data/extract_esm2_zero_shots.py --DMS_idx 0
```
See the script for usage details. For multi-mutant datasets, the log-likelihood ratios are summed for each mutant.

### Compute fitness

See above.





## Pre-computed results:

The per-variant predictions (with uncertainties) for all assays can be downloaded via:

```bash
# Download zip archive
curl -o predictions.zip https://sid.erda.dk/share_redirect/c2EWrbGSCV/predictions.zip
# Unpack and remove zip archive
unzip predictions.zip && rm predictions.zip
```

The per-variant predictions for the old split and the ablation predictions can be downloaded via 

```bash
# Download zip archive
curl -o predictions_old_split.zip https://sid.erda.dk/share_redirect/c2EWrbGSCV/predictions_old_split.zip
# Unpack and remove zip archive
unzip predictions_old_split.zip && rm predictions_old_split.zip
```

