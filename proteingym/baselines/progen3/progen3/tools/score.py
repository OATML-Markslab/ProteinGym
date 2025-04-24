import logging

import click

from progen3.common import dist
from progen3.scorer import ProGen3Scorer
from progen3.tools.utils import get_progen3_model, seed_all

logger = logging.getLogger(__name__)


@click.command()
@click.option("--model-name", type=str, required=True)
@click.option("--fasta-path", type=str, required=True)
@click.option("--output-path", type=str, required=True)
@click.option(
    "--max-batch-tokens",
    type=int,
    default=65536,
    help="Maximum number of tokens to score in a batch. Dependent on GPU memory.",
)
@click.option("--fsdp", "fsdp", is_flag=True, help="Use fsdp.")
@click.option("--seed", "seed", type=int, help="Seed for random number generators.", default=42)
def score(
    fasta_path: str,
    model_name: str,
    output_path: str,
    max_batch_tokens: int,
    fsdp: bool,
    seed: int,
) -> None:
    logger.info(f"Using fsdp: {fsdp}")
    if not dist.is_initialized() and fsdp:
        raise ValueError("Distributed training is not initialized but fsdp is set to True.")
    seed_all(seed)
    model = get_progen3_model(model_name, use_fsdp=fsdp)
    scorer = ProGen3Scorer(model, max_batch_tokens=max_batch_tokens)
    scorer.run(fasta_path, output_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    dist.setup_dist()
    try:
        score()
    finally:
        dist.destroy_process_group()
