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

## Installation

After cloning the repository, the environment can be installed via

```bash
conda env create -f environment.yml
conda activate kermut_env
pip install -e .
```

The `environment.yml` file is a minimal version. For exact reproduction, the `environment_full.yml` file should be used (which will however be system dependent).

### Optional 
To run Kermut from scratch without precomputed resources, e.g., for a new dataset, the ProteinMPNN repository must be installed. Additionally, the ESM-2 650M parameter model must be saved locally: 
#### ProteinMPNN
Kermut leverages structure-conditioned amino acid distributions from [ProteinMPNN](https://www.science.org/doi/10.1126/science.add2187), which can has to installed from the [official repository](https://github.com/dauparas/ProteinMPNN). An environment variable pointing to the installation location can then be set for later use:

```bash
export PROTEINMPNN_DIR=<path-to-ProteinMPNN-installation>
```

#### ESM-2 models 
Kermut leverages protein sequence embeddings and zero-shot scores extracted from ESM-2 ([paper](https://www.science.org/doi/10.1126/science.ade2574), [repo](https://github.com/facebookresearch/esm)). We concretely use the 650M parameter model (`esm2_t33_650M_UR50D`). While the ESM repository is installed above /via the yml-file), the model weights should be downloaded separately and placed in the `models` directory:

```bash
curl -o models/esm2_t33_650M_UR50D.pt https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt

curl -o models/esm2_t33_650M_UR50D-contact-regression.pt https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
```


## Data access
This section describes how to access the data that was used to generate the results. To reproduce _all_ results from scratch, follow all steps in this section and in the [Data preprocessing](#data-preprocessing) section. To reproduce the benchmark results using precomputed resources (ESM-2 embeddings, conditional amino-acid distributions, etc.) see the section on [precomputed resources](#precomputed-resources).

Kermut is evaluated on the ProteinGym benchmark ([paper](https://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html), [repo](https://github.com/OATML-Markslab/ProteinGym)).
For full details on downloading the relevant data, please see the ProteinGym [resources](https://github.com/OATML-Markslab/ProteinGym?tab=readme-ov-file#resources). In the following, commands are provided to extract the relevant data.

- __Reference file__: A [reference file](https://github.com/OATML-Markslab/ProteinGym/blob/main/reference_files/DMS_substitutions.csv) with details on all assays can be downloaded from the ProteinGym repo and should be saved as `data/DMS_substitutions.csv`

The file can be downloaded by running the following:
```bash
curl -o data/DMS_substitutions.csv https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv
```

- __Assay data__: All assays (with CV folds) can be downloaded and extracted to `data`. This results in two subdirectories: `data/substitutions_singles` and `data/substitutions_multiples`. The data can be accessed via `CV folds - Substitutions - <Singles,Multiples>` in ProteinGym. 

To download the data, run the following:
```bash
# Download zip archive
curl -o cv_folds_singles_substitutions.zip https://marks.hms.harvard.edu/proteingym/cv_folds_singles_substitutions.zip
# Unpack and remove zip archive
unzip cv_folds_singles_substitutions.zip -d data
rm cv_folds_singles_substitutions.zip
```


- __PDBs__: All predicted structure files are downloaded and placed in `data/structures/pdbs`. PDBs are accessed via `Predicted 3D structures from inverse-folding models` in ProteinGym.

```bash
# Download zip archive
curl -o ProteinGym_AF2_structures.zip https://marks.hms.harvard.edu/proteingym/ProteinGym_AF2_structures.zip
# Unpack and remove zip archive
unzip ProteinGym_AF2_structures.zip -d data/structures/pdbs
rm ProteinGym_AF2_structures.zip
```

- __Zero-shot scores__: For the zero-shot mean function, precomputed scores can be downloaded and placed in `zero_shot_fitness_predictions`, where each zero-shot method has its own directory. The precomputed zero-shot scores from ProteinGym can be accessed via `Zero-shot DMS Model scores - Substitutions`. __NOTE__: The full zip archive with all scores takes up approximately 44GB of storage. Alternatively, the zero-shot scores for the 650M parameter ESM-2 model is included in the [precomputed resources](#precomputed-resources), which in total is only approximately 4GB.

```bash
# Download zip archive
curl -o zero_shot_substitutions_scores.zip https://marks.hms.harvard.edu/proteingym/zero_shot_substitutions_scores.zip
unzip zero_shot_substitutions_scores.zip -d data/zero_shot_fitness_predictions
# Unpack and remove zip archive
rm zero_shot_substitutions_scores.zip
```

- (Optional) __Baselines scores__: Results from ProteinNPT and the baseline models from ProteinGym can be accessed via `Supervised DMS Model performance - Substitutions`. The resulting csv-file can be placed in `results/baselines`.

```bash 
curl -o DMS_supervised_substitutions_scores.zip https://marks.hms.harvard.edu/proteingym/DMS_supervised_substitutions_scores.zip
# Unpack and remove zip archive
unzip DMS_supervised_substitutions_scores.zip -d results/baselines
rm DMS_supervised_substitutions_scores.zip
```



## Data preprocessing
### Sequence embeddings
After downloading and extracting the relevant data in the [Data access section](#data-access), ESM-2 embeddings can be generated via the `example_scripts/generate_embeddings.sh` script or simply via: 

```bash
python src/data/extract_esm2_embeddings.py \
    --dataset=all \
    --which=singles
```
The embeddings for individual assays can be generated by replacing `all` with the full assay name (e.g., `--dataset=BLAT_ECOLX_Stiffler_2015`).
For each assay, an `h5` file is generated which contains all embeddings for all variants. Since the Kermut GP only uses the mean-pooled embeddings, only these are stored. To obtain per AA embeddings, the extraction script can be altered by removing the mean operator.
The embeddings are located in `data/embeddings/substitutions_singles/ESM2` (for the single-mutant assays).
For the multi-mutant assays, replace `singles` with `multiples`.


### Structure-conditioned amino acid distributions

The structure-conditioned amino acid distributions for all residues and assays, can be computed with ProteinMPNN via

```
bash example_scripts/conditional_probabilities.sh
```
For a single dataset, see `example_scripts/conditional_probabilities_single.sh` or `example_scripts/conditional_probabilities_all.sh`. This generates per-assay directories in `data/conditional_probs/raw_ProteinMPNN_outputs`. After this, postprocessing for easier access is performed via
```bash
python src/data/extract_ProteinMPNN_probs.py
```
This generates per-assay `npy`-files in `data/conditional_probs/ProteinMPNN`.

### 3D coordinates
Lastly, the 3D coordinates can be extracted from each PDB file via
```bash
python src/data/extract_3d_coords.py
```
This saves `npy`-files for each assay in `data/structures/coords`. 

### Optional: Zero-shot scores
If not relying on pre-computed zero-shot scores from ProteinGym, they can be computed for ESM-2 via:
```bash
python src/data/extract_esm2_zero_shots.py --dataset all 
# for all datasets, or 
python src/data/extract_esm2_zero_shots.py --dataset name_of_dataset
# for a single dataset.
```
See the script for usage details. For multi-mutant datasets, the log-likelihood ratios are summed for each mutant.


## Precomputed resources
All outputs from the preprocessing procedure (i.e., precomputed ESM-2 embeddings, conditional amino acid distributions, processed coordinate files, and zero-shot scores from ESM-2) can be readily accessed via a zip-archive hosted by the Electronic Research Data Archive (ERDA) by the University of Copenhagen using the following [link](https://sid.erda.dk/sharelink/c2EWrbGSCV). The file takes up approximately 4GB. To download and extract the data, run the following:

```bash
# Download zip archive
curl -o kermut_data.zip https://sid.erda.dk/share_redirect/c2EWrbGSCV/kermut_data.zip
# Unpack and remove zip archive
unzip kermut_data.zip && rm kermut_data.zip
```




## Usage

### Reproduce the main results
To reproduce the results (assuming that preprocessing has been carried out for all 217 DMS assays), run the following script:
```bash
bash example_scripts/benchmark_all_datasets.sh
```
This will run the Kermut GP on all 217 assays using the predefined random, modulo, and contiguous splits for cross validation. 
Each assay and split configuration generates a csv-file with the per-variant predictive means and variances. 
The prediction files can be merged using
```bash
python src/process_results/merge_score_files.py
```
This script generates two files: `kermut_scores.csv` and `merged_scores.csv`, where the former contains all Kermut's processed results, while the latter additionally includes all the reference models from ProteinGym.

Lastly, the results are aggregated across proteins and functional categories to obtain the final scores using ProteinGym functionality:
```bash
python src/process_results/performance_DMS_supervised_benchmarks.py \
    --input_scoring_file results/merged_scores.csv \
    --output_performance_file_folder results/summary
```
The post-processing steps for the main and ablation results can be seen in `example_scripts/process_results.sh`.

### Run on single assay
Assuming that preprocessed data from ProteinGym is used, Kermut can be evaluated on any assay individually as follows:
```bash
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=ASSAY_NAME \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut \
    use_gpu=true
```
The codebase was developed using [Hydra](https://hydra.cc/). The ablation GPs/kernels can be accessed straightforwardly:
```bash
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=ASSAY_NAME \
    split_method=fold_random_5,fold_modulo_5,fold_contiguous_5 \
    gp=kermut_constant_mean,kermut_no_g
```

### Running on new splits
Any dataset for which the structure follows the ProteinGym data (and has been processed as above) can be evaluated using new splits. 
This simply requires adding new columns to the assay file in `data/substitutions_singles`. The column should contain integer values for each variant indicating which fold it belongs to. The chosen GP will then be evaluated using CV (i.e., tested on all unique paritions while trained on the remaining):
```bash
python src/experiments/proteingym_benchmark.py --multirun \
    dataset=ASSAY_NAME \
    split_method=new_split_col \
    gp=kermut \
    use_gpu=true
```
