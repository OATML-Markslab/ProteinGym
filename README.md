# ProteinGym

## Overview

ProteinGym is an extensive set of Deep Mutational Scanning (DMS) assays and annotated human clinical variants curated to enable thorough comparisons of various mutation effect predictors in different regimes. Both the DMS assays and clinical variants are divided into 1) a substitution benchmark which currently consists of the experimental characterisation of ~2.7M missense variants across 217 DMS assays and 2,525 clinical proteins, and 2) an indel benchmark that includes ∼300k mutants across 74 DMS assays and 1,555 clinical proteins.

Each processed file in each benchmark corresponds to a single DMS assay or clinical protein, and contains the following variables:
- mutant (str): describes the set of substitutions to apply on the reference sequence to obtain the mutated sequence (eg., A1P:D2N implies the amino acid 'A' at position 1 should be replaced by 'P', and 'D' at position 2 should be replaced by 'N'). Present in the the ProteinGym substitution benchmark only (not indels).
- mutated_sequence (str): represents the full amino acid sequence for the mutated protein.
- DMS_score (float): corresponds to the experimental measurement in the DMS assay. Across all assays, the higher the DMS_score value, the higher the fitness of the mutated protein. This column is not present in the clinical files, since they are classified as benign/pathogenic, and do not have continuous scores
- DMS_score_bin (int): indicates whether the DMS_score is above the fitness cutoff (1 is fit (pathogenic for clinical variants), 0 is not fit (benign for clinical variants))

Additionally, we provide two reference files for each benchmark that give further details on each assay and contain in particular:
- The UniProt_ID of the corresponding protein, along with taxon and MSA depth category
- The target sequence (target_seq) used in the assay
- For the assays, details on how the DMS_score was created from the raw files and how it was binarized 

To download the benchmarks, please see `DMS benchmark: Substitutions` and `DMS benchmark: Indels` in the "Resources" section below.

## Fitness prediction performance

The [benchmarks](https://github.com/OATML-Markslab/ProteinGym/tree/main/benchmarks) folder provides detailed performance files for all baselines on the DMS and clinical benchmarks.

Metrics for DMS assays (both supervised and zero-shot): Spearman, NDCG, AUC, MCC and Top-K recall
Metrics for clinical benchmark: AUC

Metrics are aggregated as follows:
1. Aggregating by UniProt ID (to avoid biasing results towards proteins for which several DMS assays are available in ProteinGym)
2. Aggregating by different functional categories, and taking the mean across those categories.

These files are named e.g. `DMS_substitutions_Spearman_DMS_level.csv`, `DMS_substitutions_Spearman_Uniprot_level` and `DMS_substitutions_Spearman_Uniprot_Selection_Type_level` respectively for these different steps.

For other deep dives (performance split by taxa, MSA depth, mutational depth and more), these are all contained in the `benchmarks/DMS_zero_shot/substitutions/Spearman/Summary_performance_DMS_substitutions_Spearman.csv` folder (resp. DMS_indels/clinical_substitutions/clinical_indels & their supervised counterparts). These files are also what are hosted on the website.

We also include, as on the website, a bootstrapped standard error of these aggregated metrics to reflect the variance in the final numbers with respect to the individual assays.

To calculate the DMS substitution benchmark metrics:
1. Download the model scores from the website
2. Run `./scripts/scoring_DMS_zero_shot/performance_substitutions.sh`

And for indels, follow step #1 and run `./scripts/scoring_DMS_zero_shot/performance_substitutions_indels.sh`.

### ProteinGym benchmarks - Leaderboard

The full ProteinGym benchmarks performance files are also accessible via our dedicated website: https://www.proteingym.org/.
It includes leaderboards for the substitution and indel benchmarks, as well as detailed DMS-level performance files for all baselines.
The current version of the substitution benchmark includes the following baselines:

Model name | Model type | Reference
--- | --- | --- |
Site Independent | Alignment-based model | [Hopf, T.A., Ingraham, J., Poelwijk, F.J., Schärfe, C.P., Springer, M., Sander, C., & Marks, D.S. (2017). Mutation effects predicted from sequence co-variation. Nature Biotechnology, 35, 128-135.](https://www.nature.com/articles/nbt.3769)
EVmutation | Alignment-based model | [Hopf, T.A., Ingraham, J., Poelwijk, F.J., Schärfe, C.P., Springer, M., Sander, C., & Marks, D.S. (2017). Mutation effects predicted from sequence co-variation. Nature Biotechnology, 35, 128-135.](https://www.nature.com/articles/nbt.3769)
WaveNet | Alignment-based model | [Shin, J., Riesselman, A.J., Kollasch, A.W., McMahon, C., Simon, E., Sander, C., Manglik, A., Kruse, A.C., & Marks, D.S. (2021). Protein design and variant prediction using autoregressive generative models. Nature Communications, 12.](https://www.nature.com/articles/s41467-021-22732-w)
DeepSequence | Alignment-based model | [Riesselman, A.J., Ingraham, J., & Marks, D.S. (2018). Deep generative models of genetic variation capture the effects of mutations. Nature Methods, 15, 816-822.](https://www.nature.com/articles/s41592-018-0138-4)
GEMME | Alignment-based model | [Laine, É., Karami, Y., & Carbone, A. (2019). GEMME: A Simple and Fast Global Epistatic Model Predicting Mutational Effects. Molecular Biology and Evolution, 36, 2604 - 2619.](https://pubmed.ncbi.nlm.nih.gov/31406981/)
EVE | Alignment-based model | [Frazer, J., Notin, P., Dias, M., Gomez, A.N., Min, J.K., Brock, K.P., Gal, Y., & Marks, D.S. (2021). Disease variant prediction with deep generative models of evolutionary data. Nature.](https://www.nature.com/articles/s41586-021-04043-8)
Unirep | Protein language model | [Alley, E.C., Khimulya, G., Biswas, S., AlQuraishi, M., & Church, G.M. (2019). Unified rational protein engineering with sequence-based deep representation learning. Nature Methods, 1-8](https://www.nature.com/articles/s41592-019-0598-1)
ESM-1b | Protein language model | Original model: [Rives, A., Goyal, S., Meier, J., Guo, D., Ott, M., Zitnick, C.L., Ma, J., & Fergus, R. (2019). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences of the United States of America, 118](https://www.biorxiv.org/content/10.1101/622803v4); Extensions: [Brandes, N., Goldman, G., Wang, C.H. et al. Genome-wide prediction of disease variant effects with a deep protein language model. Nat Genet 55, 1512–1522 (2023).](https://doi.org/10.1038/s41588-023-01465-0)

ESM-1v | Protein language model | [Meier, J., Rao, R., Verkuil, R., Liu, J., Sercu, T., & Rives, A. (2021). Language models enable zero-shot prediction of the effects of mutations on protein function. NeurIPS.](https://proceedings.neurips.cc/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html)
VESPA | Protein language model | [Marquet, C., Heinzinger, M., Olenyi, T., Dallago, C., Bernhofer, M., Erckert, K., & Rost, B. (2021). Embeddings from protein language models predict conservation and variant effects. Human Genetics, 141, 1629 - 1647.](https://link.springer.com/article/10.1007/s00439-021-02411-y)
RITA | Protein language model | [Hesslow, D., Zanichelli, N., Notin, P., Poli, I., & Marks, D.S. (2022). RITA: a Study on Scaling Up Generative Protein Sequence Models. ArXiv, abs/2205.05789.](https://arxiv.org/abs/2205.05789)
ProtGPT2 | Protein language model | [Ferruz, N., Schmidt, S., & Höcker, B. (2022). ProtGPT2 is a deep unsupervised language model for protein design. Nature Communications, 13.](https://www.nature.com/articles/s41467-022-32007-7)
ProGen2 | Protein language model | [Nijkamp, E., Ruffolo, J.A., Weinstein, E.N., Naik, N., & Madani, A. (2022). ProGen2: Exploring the Boundaries of Protein Language Models. ArXiv, abs/2206.13517.](https://arxiv.org/abs/2206.13517)
MSA Transformer | Hybrid |[Rao, R., Liu, J., Verkuil, R., Meier, J., Canny, J.F., Abbeel, P., Sercu, T., & Rives, A. (2021). MSA Transformer. ICML.](http://proceedings.mlr.press/v139/rao21a.html)
Tranception | Hybrid | [Notin, P., Dias, M., Frazer, J., Marchena-Hurtado, J., Gomez, A.N., Marks, D.S., & Gal, Y. (2022). Tranception: protein fitness prediction with autoregressive transformers and inference-time retrieval. ICML.](https://proceedings.mlr.press/v162/notin22a.html)
TranceptEVE | Hybrid | [Notin, P., Van Niekerk, L., Kollasch, A., Ritter, D., Gal, Y. & Marks, D.S. &  (2022). TranceptEVE: Combining Family-specific and Family-agnostic Models of Protein Sequences for Improved Fitness Prediction. NeurIPS, LMRL workshop.](https://www.biorxiv.org/content/10.1101/2022.12.07.519495v1?rss=1)
CARP | Protein language model | [Yang, K.K., Fusi, N., Lu, A.X. (2022). Convolutions are competitive with transformers for protein sequence pretraining.](https://doi.org/10.1101/2022.05.19.492714)
MIF | Inverse folding | [Yang, K.K., Yeh, H., Zanichelli, N. (2022). Masked Inverse Folding with Sequence Transfer for Protein Representation Learning.](https://doi.org/10.1101/2022.05.25.493516)
<!-- TODO Add all the others -->

Except for the WaveNet model (which only uses alignments to recover a set of homologous protein sequences to train on, but then trains on non-aligned sequences), all alignment-based methods are unable to score indels given the fixed coordinate system they are trained on. Similarly, the masked-marginals procedure to generate the masked-marginals for ESM-1v and MSA Transformer requires the position to exist in the wild-type sequence. All the other model architectures listed above (eg., Tranception, RITA, ProGen2) are included in the indel benchmark.

For clinical baselines, we used dbNSFP 4.4a as detailed in the manuscript appendix (and in `proteingym/clinical_benchmark_notebooks/clinical_subs_processing.ipynb`).

## Resources

To download and unzip the data, run the following commands for each of the data sources you would like to download, e.g. for all the baseline scores on the DMS substitution scores:
```
curl -o scores_all_models_proteingym_substitutions.zip https://marks.hms.harvard.edu/proteingym/scores_all_models_proteingym_substitutions.zip
unzip scores_all_models_proteingym_substitutions.zip
rm scores_all_models_proteingym_substitutions.zip
```
Data | Size (unzipped) | Link
--- | --- | --- |
DMS benchmark: Substitutions | 1.1GB | https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_substitutions.zip
DMS benchmark: Indels | 200MB | https://marks.hms.harvard.edu/proteingym/DMS_ProteinGym_indels.zip
DMS Model scores: Substitutions | 44.1GB | https://marks.hms.harvard.edu/proteingym/zero_shot_substitutions_scores.zip
DMS Model scores: Indels | 9.6GB | https://marks.hms.harvard.edu/proteingym/zero_shot_indels_scores.zip
Multiple Sequence Alignments (MSAs) for DMS assays | 5.2GB | https://marks.hms.harvard.edu/proteingym/DMS_msa_files.zip
Redundancy-based sequence weights for DMS assays | 200MB | https://marks.hms.harvard.edu/proteingym/DMS_msa_weights.zip
Predicted 3D structures from inverse-folding models | 84MB | https://marks.hms.harvard.edu/proteingym/ProteinGym_AF2_structures.zip
Clinical benchmark: Substitutions | 123MB | https://marks.hms.harvard.edu/proteingym/clinical_ProteinGym_substitutions.zip
Clinical benchmark: Indels | 2.8MB | https://marks.hms.harvard.edu/proteingym/clinical_ProteinGym_indels.zip
Clinical MSAs | 17.8GB | https://marks.hms.harvard.edu/proteingym/clinical_msa_files.zip
Clinical MSA weights | 250MB | https://marks.hms.harvard.edu/proteingym/clinical_msa_weights.zip
Clinical Model scores: Substitutions | 0.9GB | https://marks.hms.harvard.edu/proteingym/zero_shot_clinical_substitutions_scores.zip
Clinical Model scores: Indels | 0.7GB | https://marks.hms.harvard.edu/proteingym/zero_shot_clinical_indels_scores.zip
CV folds: Substitutions - Singles | 50M | https://marks.hms.harvard.edu/proteingym/cv_folds_singles_substitutions.zip
CV folds: Substitutions - Multiples | 81M | https://marks.hms.harvard.edu/proteingym/cv_folds_multiples_substitutions.zip
CV folds: Indels | 19MB | https://marks.hms.harvard.edu/proteingym/cv_folds_indels.zip

Then we also host the raw DMS assays (before preprocessing)

Data | Size (unzipped) | Link
--- | --- | --- |
DMS benchmark: Substitutions (raw) | 500MB | https://marks.hms.harvard.edu/proteingym/substitutions_raw_DMS.zip
DMS benchmark: Indels (raw) | 450MB | https://marks.hms.harvard.edu/proteingym/indels_raw_DMS.zip
Clinical benchmark: Substitutions (raw) | 58MB | https://marks.hms.harvard.edu/proteingym/substitutions_raw_clinical.zip
Clinical benchmark: Indels (raw) | 12.4MB | https://marks.hms.harvard.edu/proteingym/indels_raw_clinical.zip

## How to contribute?

### New assays
If you would like to suggest new assays to be part of ProteinGym, please raise an issue on this repository with a `new_assay' label. The criteria we typically consider for inclusion are as follows:
1. The corresponding raw dataset needs to be publicly available
2. The assay needs to be protein-related (ie., exclude UTR, tRNA, promoter, etc.)
3. The dataset needs to have insufficient number of measurements
4. The assay needs to have a sufficiently high dynamic range
5. The assay has to be relevant to fitness prediction

### New baselines
If you would like new baselines to be included in ProteinGym (ie., website, performance files, detailed scoring files), please follow the following steps:
1. Submit a PR to our repo with two things:
   - A new subfolder under proteingym/baselines named with your new model name. This subfolder should include a python scoring script similar to [this script](https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/baselines/rita/compute_fitness.py), as well as all code dependencies required for the scoring script to run properly
   - An example bash script (e.g., under scripts/scoring_DMS_zero_shot) with all relevant hyperparameters for scoring, similar to [this script](https://github.com/OATML-Markslab/ProteinGym/blob/main/scripts/scoring_DMS_zero_shot/scoring_RITA_substitutions.sh)
2. Raise an issue with a 'new model' label, providing instructions on how to download relevant model checkpoints for scoring, and reporting the performance of your model on the relevant benchmark using our performance scripts (e.g., [for zero-shot DMS benchmarks](https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/performance_DMS_benchmarks.py)). Please note that our DMS performance scripts correct for various biases (e.g., number of assays per protein family and function groupings) and thus the resulting aggregated performance is not the same as the arithmetic average across assays.

At this point we are only considering new baselines satisfying the following conditions:
1. The model is able to score all mutants in the relevant benchmark (to ensure all models are compared exactly on the same set of mutants everywhere);
2. The corresponding model is open source (we should be able to reproduce scores if needed).

At this stage, we are only considering requests for which all model scores for all mutants in a given benchmark (substitution or indel) are provided by the requester; but we are planning on regularly scoring new baselines ourselves for methods with wide adoption by the community and/or suggestions with many upvotes.

### Notes
12 December 2023: The code for training and evaluating supervised models is currently shared in https://github.com/OATML-Markslab/ProteinNPT. We are in the process of moving the code to this repo.

## Instructions

If you would like to compute all performance metrics for the various benchmarks, please follow the following steps:
1. Download locally all relevant files as per instructions above (see Resources)
2. Update the paths for all files downloaded in the prior step in the [config script](https://github.com/OATML-Markslab/ProteinGym/blob/main/scripts/zero_shot_config.sh)
3. If adding a new model, adjust the [config.json](https://github.com/OATML-Markslab/ProteinGym/blob/main/config.json) file accordingly and add the model scores to the relevant path (e.g., [DMS_output_score_folder_subs](https://github.com/OATML-Markslab/ProteinGym/blob/main/scripts/zero_shot_config.sh#L19))
4. If focusing on DMS benchmarks, run the [merge script](https://github.com/OATML-Markslab/ProteinGym/blob/main/scripts/scoring_DMS_zero_shot/merge_all_scores.sh). This will create a single file for each DMS assay, with scores for all model baselines
5. Run the relevant performance script (eg., [scripts/scoring_DMS_zero_shot/performance_substitutions.sh](https://github.com/OATML-Markslab/ProteinGym/blob/main/scripts/scoring_DMS_zero_shot/performance_substitutions.sh))

## Acknowledgements

Our codebase leveraged code from the following repositories to compute baselines:

Model | Repo
--- | ---
UniRep | https://github.com/churchlab/UniRep
UniRep | https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data
EVE | https://github.com/OATML-Markslab/EVE
GEMME | https://hub.docker.com/r/elodielaine/gemme
ESM | https://github.com/facebookresearch/esm
EVmutation | https://github.com/debbiemarkslab/EVcouplings
ProGen2 | https://github.com/salesforce/progen
HMMER | https://github.com/EddyRivasLab/hmmer
MSA Transformer | https://github.com/rmrao/msa-transformer
ProtGPT2 | https://huggingface.co/nferruz/ProtGPT2
ProteinMPNN | https://github.com/dauparas/ProteinMPNN
RITA | https://github.com/lightonai/RITA
Tranception | https://github.com/OATML-Markslab/Tranception
VESPA | https://github.com/Rostlab/VESPA
CARP | https://github.com/microsoft/protein-sequence-models
MIF | https://github.com/microsoft/protein-sequence-models

We would like to also thank the teams of experimentalists who developed and performed the assays that ProteinGym is built on. If you are using ProteinGym in your work, please consider citing the corresponding papers. To facilitate this, we have prepared a file (assays.bib) containing the bibtex entries for all these papers.

## License
This project is available under the MIT license found in the LICENSE file in this GitHub repository.

## Reference
If you use ProteinGym in your work, please cite the following paper:
[ProteinGym: Large-Scale Benchmarks for Protein Design and Fitness Prediction](https://www.biorxiv.org/content/10.1101/2023.12.07.570727v1)

## Links
Website: https://www.proteingym.org/
