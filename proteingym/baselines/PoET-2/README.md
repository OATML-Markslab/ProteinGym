# PoET-2

This repository contains inference code for [PoET-2](https://arxiv.org/abs/2508.04724),
a multimodal, retrieval-augmented protein language model for state-of-the-art variant
effect prediction and controllable protein sequence generation.

## License

The following table lists the licenses for the components of this project. Please review
the terms for each component carefully.

| Component       | License                                                             |
|-----------------|---------------------------------------------------------------------|
| **Source Code </br> (excludes model weights)**   | [Apache License 2.0](LICENSE) |
| **Model Weights** | [PoET Non-Commercial License Agreement](MODEL_LICENSE.md)               |

For commercial use of the model weights, please reach out to us at contact@ne47.bio.

## Third-Party Components

This repository includes artifacts from third-party projects:

- Code in `src/poet_2/models/modules/norm.py` is adapted from [Mamba](https://github.com/state-spaces/mamba)
  and originally licensed under the Apache License, Version 2.0.
- Code in `src/poet_2/models/modules/glu.py` for the class `GLU` is adapted from
  [x-formers](https://github.com/lucidrains/x-transformers) and originally licensed under
  the MIT License.
- Code in `src/poet_2/models/poet_2.py` (specifically the `decode` function) is adapted
  from [FlashAttention](https://github.com/Dao-AILab/flash-attention) and originally
  licensed under the BSD 3-Clause License.
- Code in `src/poet_2/models/modules/packed_sequence.py` (specifically `unpad_input` and
  `pad_input`) is adapted from [FlashAttention](https://github.com/Dao-AILab/flash-attention)
  and originally licensed under the BSD 3-Clause License.
- Code in `src/poet_2/models/modules/attention_flash_fused_bias.py` is adapted from
  [TurboT5](https://github.com/Knowledgator/TurboT5) and originally licensed under the
  Apache License, Version 2.0.
- Some code in this repository depends on a custom implementation of [FlashAttention](https://github.com/Dao-AILab/flash-attention);
  FlashAttention is originally licensed under the BSD 3-Clause License. This custom
  implementation overrides the default behavior of the `alibi_slopes` parameter of the
  attention function.

Copies of the applicable third-party licenses are available in the
`third_party_licenses/` directory.

Portions of the aforementioned artifacts that the original licenses require to remain
under those licenses continue to be governed by them; all other poritions of the
aforementioned artifacts, and the rest of the repository, is covered by this
repository’s license(s).

## Citation

You may cite the paper as

```
@misc{truong2025understandingproteinfunctionmultimodal,
      title={Understanding protein function with a multimodal retrieval-augmented foundation model}, 
      author={Timothy Fei Truong Jr and Tristan Bepler},
      year={2025},
      eprint={2508.04724},
      archivePrefix={arXiv},
      primaryClass={q-bio.QM},
      url={https://arxiv.org/abs/2508.04724}, 
}
```
