# Supervised summary regression tests

These CPU-only tests invoke the real supervised reporting CLI with small
synthetic assay tables. They require no model weights, external datasets,
or network access.

One tested dependency set is Python 3.12 with pandas 2.2.3, NumPy 2.2.6,
tqdm 4.67.1 and pytest 8.4.2. Run from the repository root:

```bash
python -m pytest tests/test_supervised_summary.py -q
```

The cases cover substitutions and indels, Spearman and MSE outputs,
input-order independence, distinct MSA/taxon groups, unchanged overall
and functional scores, and rejection of missing required categories.
