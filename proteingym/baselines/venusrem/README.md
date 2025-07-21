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

For more details about VenusREM, please refer to the official [VenusREM GitHub repo](https://github.com/ai4protein/VenusREM).

Please cite the following works if you have used the VenusREM code or data:

```
@inproceedings{
    li2024prosst,
    title={Pro{SST}: Protein Language Modeling with Quantized Structure and Disentangled Attention},
    author={Mingchen Li and Yang Tan and Xinzhu Ma and Bozitao Zhong and Huiqun Yu and Ziyi Zhou and Wanli Ouyang and Bingxin Zhou and Pan Tan and Liang Hong},
    booktitle={Advances in Neural Information Processing Systems},
    year={2024},
    url={https://openreview.net/forum?id=4Z7RZixpJQ}
}

@article{tan2023protssn,
  article_type = {journal},
  title = {Semantical and geometrical protein encoding toward enhanced bioactivity and thermostability},
  author = {Tan, Yang and Zhou, Bingxin and Zheng, Lirong and Fan, Guisheng and Hong, Liang},
  editor = {Koo, Peter and Cui, Qiang},
  volume = 13,
  year = 2025,
  month = {May},
  pub_date = {2025-05-02},
  pages = {RP98033},
  citation = {eLife 2025;13:RP98033},
  doi = {10.7554/eLife.98033},
  url = {https://doi.org/10.7554/eLife.98033},
  journal = {eLife},
  issn = {2050-084X},
  publisher = {eLife Sciences Publications, Ltd},
}

@article{tan2025venusrem,
    author = {Tan, Yang and Wang, Ruilin and Wu, Banghao and Hong, Liang and Zhou, Bingxin},
    title = {From high-throughput evaluation to wet-lab studies: advancing mutation effect prediction with a retrieval-enhanced model},
    journal = {Bioinformatics},
    volume = {41},
    number = {Supplement_1},
    pages = {i401-i409},
    year = {2025},
    month = {07},
    issn = {1367-4811},
    doi = {10.1093/bioinformatics/btaf189},
    url = {https://doi.org/10.1093/bioinformatics/btaf189},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/41/Supplement\_1/i401/63745466/btaf189.pdf},
}
```