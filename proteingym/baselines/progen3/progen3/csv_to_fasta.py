import glob
import os
from pathlib import Path

import click
import pandas as pd
from tqdm import tqdm


@click.command()
@click.option("--csv-path", type=str, required=True)
@click.option("--fasta-path", type=str, required=True)
def main(csv_path: str, fasta_path: str) -> None:
    if os.path.isdir(csv_path):
        csv_files = glob.glob(os.path.join(csv_path, "*.csv"))
    elif os.path.isfile(csv_path):
        csv_files = [csv_path]
    else:
        raise ValueError(f"Invalid CSV path: {csv_path}")

    num_sequences = 0
    with open(fasta_path, "w") as f:
        for csv_file in tqdm(csv_files, desc="Converting files to FASTA", ncols=80):
            df = pd.read_csv(csv_file)
            assay_name = Path(csv_file).stem
            for idx, sequence in enumerate(df["mutated_sequence"].values):
                f.write(f">{assay_name}+{idx}\n{sequence}\n")
                num_sequences += 1

    print(f"Wrote {num_sequences} sequences to {fasta_path}")


if __name__ == "__main__":
    main()
