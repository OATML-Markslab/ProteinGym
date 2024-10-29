import warnings
from pathlib import Path

import polars as pl
import typer
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm.rich import tqdm
from typing_extensions import Annotated

from vespag.runner.predict import predict
from vespag.utils import setup_logger

app = typer.Typer()


@app.command()
def proteingym(
    dms_reference_file: Annotated[
        Path, typer.Option("--reference-file", help="Path of DMS reference file")
    ],
    dms_directory: Annotated[
        Path,
        typer.Option(
            "--dms-directory", help="Path of directory containing per-DMS score files"
        ),
    ],
    output_path: Annotated[
        Path,
        typer.Option(
            "-o", "--output", help="Output path. Defaults to ./output/proteingym217"
        ),
    ],
    embedding_file: Annotated[
        Path,
        typer.Option(
            "-e",
            "--embeddings",
            help="Path to pre-generated input embeddings. Embeddings will be generated from scratch if no path is provided.",
        ),
    ] = None,
    checkpoint_file: Annotated[
        Path,
        typer.Option(
            "--checkpoint-file",
            help="Path to model checkpoint.",
        ),
    ] = None,
    id_map_file: Annotated[
        Path,
        typer.Option(
            "--id-map",
            help="CSV file mapping embedding IDs to FASTA IDs if they're different",
        ),
    ] = None,
    normalize_scores: Annotated[
        bool,
        typer.Option(
            "--normalize/--dont-normalize", help="Whether to normalize scores to [0, 1]"
        ),
    ] = True,
):
    logger = setup_logger()
    warnings.filterwarnings("ignore", message="rich is experimental/alpha")

    if not output_path:
        output_path = Path.cwd() / "output/proteingym217"
    output_path.mkdir(parents=True, exist_ok=True)
    sequence_file = output_path / "sequences.fasta"
    reference_df = pl.read_csv(dms_reference_file)
    sequences = [
        SeqRecord(id=row["DMS_id"], seq=Seq(row["target_seq"]))
        for row in reference_df.iter_rows(named=True)
    ]
    logger.info(f"Writing {len(sequences)} sequences to {sequence_file}")
    SeqIO.write(sequences, sequence_file, "fasta")

    logger.info(f"Parsing mutation files from {dms_directory}")
    mutation_file = output_path / "mutations.txt"
    dms_files = {
        row["DMS_id"]: pl.read_csv(dms_directory / row["DMS_filename"])
        for row in reference_df.iter_rows(named=True)
    }
    pl.concat(
        [
            df.with_columns(pl.lit(dms_id).alias("DMS_id")).select(["DMS_id", "mutant"])
            for dms_id, df in dms_files.items()
        ]
    ).write_csv(mutation_file)

    logger.info("Generating predictions")
    predict(
        fasta_file=sequence_file,
        output_path=output_path,
        embedding_file=embedding_file,
        checkpoint_file=checkpoint_file,
        mutation_file=mutation_file,
        id_map_file=id_map_file,
        single_csv=True,
        normalize_scores=normalize_scores,
    )

    mutation_file.unlink()
    sequence_file.unlink()

    prediction_file = output_path / "vespag_scores_all.csv"
    all_preds = pl.read_csv(prediction_file)

    logger.info(
        "Computing Spearman correlations between experimental and predicted scores"
    )
    records = []
    for dms_id, dms_df in tqdm(list(dms_files.items()), leave=False):
        dms_df = dms_df.join(
            all_preds.filter(pl.col("Protein") == dms_id),
            left_on="mutant",
            right_on="Mutation",
        )
        spearman = dms_df.select(
            pl.corr("DMS_score", "VespaG", method="spearman")
        ).item()
        records.append({"DMS_id": dms_id, "spearman": spearman})
    result_csv_path = output_path / "VespaG_Spearman_per_DMS.csv"
    result_df = pl.from_records(records)
    logger.info(f"Writing results to {result_csv_path}")
    logger.info(f"Mean Spearman r: {result_df['spearman'].mean():.3f}")
    result_df.write_csv(result_csv_path)
