# Retrieval-Enhanced Mutation Mastery: Augmenting Zero-Shot Prediction of Protein Language Model

## 🛫 Requirement

### Conda Enviroment
```
conda env create -f venusrem_environment.yml
conda activate venusrem
```

### Downloads

To score the ProteinGym DMS substitution assays with VenusREM, you will need to download the following files:
- aa_seq and struc_seq files may be downloaded from: https://drive.google.com/file/d/1lSckfPlx7FhzK1FX7EtmmXUOrdiMRerY/view?usp=sharing
- sequence alignments may be downloaded from the standard ProteinGym portal:
```
curl -o DMS_msa_files.zip https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.1/DMS_msa_files.zip
```

### Hardware

- For inference, the venusREM team recommends at least 10G of graphics memory, such as RTX 3080

## 🧬 Zero-shot Prediction for Mutants on ProteinGym

### Evaluation on ProteinGym

Refer to the scoring_VenusREM_substitutions.sh script under `scripts/scoring_DMS_zero_shot`.

## 🙌 Citation

For more details about VenusREM, please refer to the official [VenusREM GitHub repo](https://github.com/tyang816/VenusREM).

Please cite the following works if you have used the VenusREM code or data:

```
@article{li2024prosst,
  title={ProSST: Protein Language Modeling with Quantized Structure and Disentangled Attention},
  author={Li, Mingchen and Tan, Yang and Ma, Xinzhu and Zhong, Bozitao and Zhou, Ziyi and Yu, Huiqun and Ouyang, Wanli and Hong, Liang and Zhou, Bingxin and Tan, Pan},
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

@article{tan2024venusrem,
  title={Retrieval-Enhanced Mutation Mastery: Augmenting Zero-Shot Prediction of Protein Language Model},
  author={Tan, Yang and Wang, Ruilin and Wu, Banghao and Hong, Liang and Zhou, Bingxin},
  journal={arXiv:2410.21127},
  year={2024}
}
```