# PoET

## Environment Setup

1. Have `conda` installed
1. Run `make create_conda_env`. This will create a conda environment named `poet`.
1. Run `make download_model` to download the model (~400MB). The model will be located at `data/poet.ckpt`. Please note the [license](#License).

## Citation

You may cite the paper as

```
@inproceedings{NEURIPS2023_f4366126,
 author = {Truong Jr, Timothy and Bepler, Tristan},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Oh and T. Neumann and A. Globerson and K. Saenko and M. Hardt and S. Levine},
 pages = {77379--77415},
 publisher = {Curran Associates, Inc.},
 title = {PoET: A generative model of protein families as sequences-of-sequences},
 url = {https://proceedings.neurips.cc/paper_files/paper/2023/file/f4366126eba252699b280e8f93c0ab2f-Paper-Conference.pdf},
 volume = {36},
 year = {2023}
}
```

## License

This source code is licensed under the MIT license found in the LICENSE file in the PoET subfolder.

The [PoET model weights](https://zenodo.org/records/10061322) (DOI: `10.5281/zenodo.10061322`) are available under the [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) license for academic use only. The license can also be found in the LICENSE file provided with the model weights. For commercial use, please reach out to us at contact@ne47.bio about licensing. Copyright (c) NE47 Bio, Inc. All Rights Reserved.
