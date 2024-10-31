
# VespaG: Expert-Guided Protein Language Models enable Accurate and Blazingly Fast Fitness Prediction

<img align="right" src="images/vespag.png" alt="image" height="20%" width="20%" />

**VespaG** is a single amino acid variant effect predictor, leveraging embeddings of the protein language model [ESM-2](https://github.com/facebookresearch/esm) ([Lin et al. 2022](https://www.science.org/doi/abs/10.1126/science.ade2574)) as input to a minimal deep learning model. 

To overcome the sparsity of experimental training data, authors created a dataset of 39 million single amino acid variants from a subset of the Human proteome, which was then annotated using predictions from the multiple sequence alignment-based effect predictor [GEMME](http://www.lcqb.upmc.fr/GEMME/Home.html) ([Laine et al. 2019](https://doi.org/10.1093/molbev/msz179)) as a proxy for experimental scores. 

More details on **VespaG** can be found in the corresponding [preprint](https://doi.org/10.1101/2024.04.24.590982).

### Installation
1. `conda env create -n vespag python==3.10 poetry==1.8.3` (exchange `conda` for `mamba`, `miniconda` or `micromamba` as you like)
2. `conda activate vespag`
3. `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
4. `poetry install`

### Quick Start: Running Inference with VespaG
Run `python -m vespag predict` with the following options:  
**Required:**
- `--input/-i`: Path to FASTA-formatted file containing protein sequence(s).  
**Optional:**
- `--output/-o`:Path for saving created CSV and/or H5 files. Defaults to `./output`.
- `--embeddings/-e`: Path to pre-computed ESM2 (`esm2_t36_3B_UR50D`) input embeddings. Embeddings will be generated from scratch if no path is provided and saved in `./output`. Please note that embedding generation on CPU can be slow.
- `--mutation-file`: CSV file specifying specific mutations to score. If not provided, the whole single-site mutational landscape of all input proteins will be scored.
- `--id-map`: CSV file mapping embedding IDs (first column) to FASTA IDs (second column) if they're different. Does not have to cover cases with identical IDs.
- `--single-csv`: Whether to return one CSV file for all proteins instead of a single file for each protein.
- `--no-csv`: Whether no CSV output should be produced.
- `--h5-output`: Whether a file containing predictions in HDF5 format should be created.
- `--zero-idx`: Whether to enumerate protein sequences (both in- and output) starting at 0.
- `--normalize`: Whether to transform predicted scores to the [0, 1] interval by applying a sigmoid

### Examples
After installing the dependencies above and cloning the **VespaG** repo, you can try out the following examples:
- Run VespaG without precomputed embeddings for the example fasta file with 3 sequences in `data/example/example.fasta`: 
    - `python -m vespag predict -i data/example/example.fasta`. This will save a CSV file for each sequence in the folder `./output`
- Run VespaG with precomputed embeddings for the example fasta file with 3 sequences in `data/example/example.fasta`: 
    - `python -m vespag predict -i data/example/example.fasta -e output/esm2_embeddings.h5 --single-csv`. This will save a single CSV file for all sequences in the folder `./output`

### Re-training VespaG
VespaG uses [DVC](https://dvc.org/) for pipeline orchestration and [WandB](https://wandb.ai/) for experiment tracking. 

Using WandB is optional; a username and project for WandB can be specified in `params.yaml`.

Using DVC is non-optional. There is a `dvc.yaml` file in place that contains stages for generating pLM embeddings from FASTA files, but you can also download pre-computed embeddings and GEMME scores from the following [Zenodo repository](https://doi.org/10.5281/zenodo.11085958). Adjust paths in `params.yaml` to your context, and feel free to play around with model parameters. You can simply run a training run using `dvc repro -s train@<model_type>-{esm2|prott5}-<dataset>`, with `<model_type>` and `<dataset>` each corresponding to a named block in `params.yaml`.

### Evaluation
You can reproduce model evaluation using the `eval` subcommand, which pre-processes data into a format usable by VespaG, runs `predict`, and computes performance metrics.

#### ProteinGym
Run evaluation on the ProteinGym DMS substituions benchmark with `python -m vespag eval proteingym`, with the following options:
**Optional:**
- `--reference-file`: Path to ProteinGym reference file. Will download to `data/test/proteingym217/reference.csv` or `data/test/proteingym87/reference.csv` if not provided.
- `--dms-directory`: Path to directory containing per-DMS score files in CSV format. Will download to `data/test/proteingym217/raw_dms_files/` or `data/test/proteingym87/raw_dms_files/` if not provided.
- `--output/-o`:Path for saving created CSV with scores for all assays and variants as well as a CSV with Spearman correlation coefficients for each DMS. Defaults to `./output/proteingym217` or `./output/proteingym87`.
- `--embeddings/-e`, `--id-map`, `--normalize-scores`: identical to `predict`, used for the internal call to it.
- `--v1` if you want to get a result for the first iteration of ProteinGym with 87 assays.

## Preprint Citation
If you find VespaG helpful in your work, please cite the following pre-print:
```
@article{vespag,
	author = {Celine Marquet and Julius Schlensok and Marina Abakarova and Burkhard Rost and Elodie Laine},
	title = {VespaG: Expert-guided protein Language Models enable accurate and blazingly fast fitness prediction},
	year = {2024},
	doi = {10.1101/2024.04.24.590982},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/04/28/2024.04.24.590982},
	journal = {bioRxiv}}
```
