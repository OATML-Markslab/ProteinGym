# XSignal

XSignal is a zero-shot evidence-integration method for ProteinGym substitution assays.
It derives 14 mutation-scoring channels from ProteinGym MSAs, MSA sequence
weights, precomputed AlphaFold2 structures, and fixed amino-acid physicochemical
descriptors. Each channel is transformed into assay-wise percentile ranks and
the final score is the equal-weight average of the 14 calibrated channels.

The output score column is `XSignal_score`, with higher scores indicating higher
predicted function or fitness.

The scorer reads only variant-key columns from DMS assay files:

- `mutant`
- `mutated_sequence`

It does not read DMS labels, binary labels, supervised folds, public baseline
score packages, or official merged-score files during final scoring.

XSignal does not use pretrained protein language model weights. It does use
ProteinGym's precomputed AlphaFold2 structures and ten self-generated
intermediate XSignal evidence folders. The example scoring shell script documents
the expected intermediate folder layout.
