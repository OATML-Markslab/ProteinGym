# VenusREM2

VenusREM2 scores substitution mutants by running the vrh calibration head on the six official ProSST checkpoints (`K = 20, 128, 512, 1024, 2048, 4096`). The head is:

1. entropy-weighted mixing of ProSST log-probabilities with the MSA column distribution
2. gated column-context correction (CCD) computed on the raw ProSST logits
3. above-mean RSA decay, and above-mean pLDDT decay on predicted structures
4. a per-protein z-mean across the six checkpoints, written as `VenusREM2`

Per-checkpoint columns are `VenusREM2__ProSST-{K}`. Higher scores are more favorable.

ProteinGym only needs this one model. Run `scoring_VenusREM2_substitutions.sh` once. That command loads the six ProSST checkpoints inside a single job and writes the `VenusREM2` column used by `config.json`. The six `VenusREM2__ProSST-{K}` columns stay in the same CSV for inspection. Do not score them again as separate baselines, and do not run the vrh head on other backbones for this submission.

## Environment

The process needs `torch`, `transformers`, `biopython`, `pandas`, `numpy`, and `scipy`, and it must be able to load `AI4Protein/ProSST-*` with `trust_remote_code`. Create that stack from `proteingym/baselines/prosst/prosst_environment.yaml` if you do not already have it. An environment named `prosst` is not enough when `torch` is missing.

```shell
conda env create -f proteingym/baselines/prosst/prosst_environment.yaml
conda activate prosst
```

RSA uses Biopython's Shrake–Rupley implementation. `mdtraj` is optional and is tried first when it is installed. pLDDT is read from CA B-factors and is skipped for crystal / NMR / cryo-EM structures.

Any checkpoint that is not already cached is downloaded from Hugging Face on first use:

- `AI4Protein/ProSST-20`
- `AI4Protein/ProSST-128`
- `AI4Protein/ProSST-512`
- `AI4Protein/ProSST-1024`
- `AI4Protein/ProSST-2048`
- `AI4Protein/ProSST-4096`

## Data

If `struc_seq/{K}/{assay}.fasta` is already present, scoring reads those comma-separated ProSST tokens. If a file is missing and the assay PDB is available, `compute_fitness.py` builds it with the same `PdbQuantizer` shipped under `proteingym/baselines/venusrem` (`AE.pt` and `{K}.joblib` in `venusrem/structure/static/`). The PDB is also used for RSA and for pLDDT on predicted structures. Sequence alignments are the ProteinGym a2m files used by VenusREM.

```
data/
├── aa_seq/                      # or residue_sequence/
│   └── Protein1.fasta
├── struc_seq/                   # or structure_sequence/
│   ├── 20/Protein1.fasta
│   ├── 128/Protein1.fasta
│   └── ...
├── aa_seq_aln_a2m/
│   └── Protein1.a2m
├── pdb/
│   └── Protein1.pdb
└── substitutions/
    └── Protein1.csv             # column: mutant
```

The same step can be run on its own before scoring:

```shell
python proteingym/baselines/venusrem2/get_struc_seq.py \
    --pdb_dir /path/to/pdbs \
    --vocab_size 20 128 512 1024 2048 4096 \
    --out_dir /path/to/struc_seq
```

The ProteinGym MSA archive is `DMS_msa_files.zip`, and predicted structures are `ProteinGym_AF2_structures.zip`. PDB filenames must match the assay stem (`Protein1.pdb`).

## Scoring

Set `DMS_folder` in `scripts/scoring_DMS_zero_shot/scoring_VenusREM2_substitutions.sh` to the prepared directory, then run that script once. It does not take a model list: the six official checkpoints and the `VenusREM2` z-mean are the default.

```shell
python compute_fitness.py \
    --base_dir /path/to/data \
    --pdb_dir /path/to/ProteinGym_AF2_structures \
    --output_scores_folder /path/to/VenusREM2
```

Each assay CSV is written directly under `--output_scores_folder`, which is the layout expected by `config.json` (`location: VenusREM2`, column `VenusREM2`).

## Citation

VenusREM2 is the ProSST-ensemble instantiation of the vrh recipe in [VenusREM-Harness](https://github.com/tyang816/VenusREM-Harness). The structure backbone is ProSST.

```
@article{tan2026venusremharness,
    title={A General Harness for Protein Foundation Model Fitness Prediction},
    author={Tan, Yang and Tian, Qijia and Sun, Gangyu and Zhong, Bozitao and Li, Mingchen and Yu, Yuanxi and Dong, Nanqing and Hong, Liang},
    journal={arXiv preprint arXiv:2609.34654},
    year={2026},
    url={https://arxiv.org/abs/2609.34654}
}

@inproceedings{
    li2024prosst,
    title={Pro{SST}: Protein Language Modeling with Quantized Structure and Disentangled Attention},
    author={Mingchen Li and Yang Tan and Xinzhu Ma and Bozitao Zhong and Huiqun Yu and Ziyi Zhou and Wanli Ouyang and Bingxin Zhou and Pan Tan and Liang Hong},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=4Z7RZixpJQ}
}
```
