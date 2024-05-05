
# VespaG: Expert-Guided Protein Language Models enable Accurate and Blazingly Fast Fitness Prediction

<img align="right" src="images/vespag.png" alt="image" height="20%" width="20%" />

**VespaG** is a blazingly fast single amino acid variant effect predictor, leveraging embeddings of the protein language model [ESM-2](https://github.com/facebookresearch/esm) ([Lin et al. 2022](https://www.science.org/doi/abs/10.1126/science.ade2574)) as input to a minimal deep learning model. 

To overcome the sparsity of experimental training data, we created a dataset of 39 million single amino acid variants from a subset of the Human proteome, which we then annotated using predictions from the multiple sequence alignment-based effect predictor [GEMME](http://www.lcqb.upmc.fr/GEMME/Home.html) ([Laine et al. 2019](https://doi.org/10.1093/molbev/msz179)) as a proxy for experimental scores. 

Assessed on the [ProteinGym](https://proteingym.org) ([Notin et al. 2023](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1)) benchmark, **VespaG** matches state-of-the-art methods while being several orders of magnitude faster, predicting the entire single-site mutational landscape for a human proteome in under a half hour on a consumer-grade laptop.

More details on **VespaG** can be found in the corresponding [preprint](https://doi.org/10.1101/2024.04.24.590982).

## Quick Start
### Running Inference with VespaG
1. Install necessary dependencies: `conda env create -f environment.yml -n VespaG` and `conda activate VespaG`
2. Run `python -m vespag predict` with the following options:  
**Required:**
- `--input/-i`: Path to FASTA-formatted file containing protein sequence(s).  
**Optional:**
- `--output/-o`:Path for saving created CSV and/or H5 files. Defaults to `./output`.
- `--embeddings/-e`: Path to pre-generated ESM2 (`esm2_t36_3B_UR50D`) input embeddings. Embeddings will be generated from scratch if no path is provided and saved in `./output`. **Please note that embedding generation on CPU is extremely slow and not recommended.**
- `--mutation-file`: CSV file specifying specific mutations to score. If not provided, the whole single-site mutational landscape of all input proteins will be scored.
- `--id-map`: CSV file mapping embedding IDs (first column) to FASTA IDs (second column) if they're different. Does not have to cover cases with identical IDs.
- `--single-csv`: Whether to return one CSV file for all proteins instead of a single file for each protein.
- `--no-csv`: Whether no CSV output should be produced.
- `--h5-output`: Whether a file containing predictions in HDF5 format should be created.
- `--zero-idx`: Whether to enumerate protein sequences (both in- and output) starting at 0.
- `--normalize`: Whether to normalize predicted scores to [0,1] interval by applying a sigmoid to the predicted GEMME substitution scores (which are on a broader spectrum of about [-10,2], although some predicted values fall out of this)

#### Examples

After installing the dependencies above and cloning the **VespaG** repo, you can try out the following examples:
- Run VespaG without precomputed embeddings for the example fasta file with 3 sequences in `data/example/example.fasta`: 
    - `python -m vespag predict -i data/example/example.fasta`. This will save a CSV file for each sequence in the folder `./output`
- Run VespaG with precomputed embeddings for the example fasta file with 3 sequences in `data/example/example.fasta`: 
    - `python -m vespag predict -i data/example/example.fasta -e output/esm2_embeddings.h5 --single-csv`. This will save a single CSV file for all sequences in the folder `./output`

Kindly note that we are working on making data pre-processing and model training available in the public GitHub repository as soon as possible.

### Evaluation
You can reproduce our evaluation using the `eval` subcommand, which pre-processes data into a format usable by VespaG, runs `predict`, and computes performance metrics.

#### ProteinGym217
Based on the [ProteinGym](https://proteingym.org) ([Notin et al. 2023](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1)) DMS substitutions benchmark, dubbed _ProteinGym217_ by us. Run it with `python -m vespag eval proteingym`, with the following options:
**Required**
- `--reference-file`: Path to ProteinGym reference file
- `--dms-directory`: Path to directory containing per-DMS score files in CSV format
**Optional:**
- `--output/-o`:Path for saving created CSV with scores for all assays and variants as well as a CSV with Spearman correlation coefficients for each DMS. Defaults to `./output/proteingym217`.
- `--embeddings/-e`, `--id-map`, `--normalize-scores`: identical to `predict`, used for the internal call to it.

## Preprint Citation
If you find VespaG helpful in your work, please be so kind as to cite our pre-print:
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