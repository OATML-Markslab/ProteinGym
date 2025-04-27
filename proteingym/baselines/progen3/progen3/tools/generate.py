import logging

import click
from torch.distributed.elastic.multiprocessing.errors import record

from progen3.common import dist
from progen3.generator import ProGen3Generator
from progen3.tools.utils import get_progen3_model

logger = logging.getLogger(__name__)


@record
@click.command()
@click.option("--prompt-file", type=str, required=True, help="Must be a .csv file with a 'sequence' column.")
@click.option("--model-name", type=str, default="progen3-base")
@click.option("--n-per-prompt", type=int, default=1, help="Number of sequences to generate per prompt.")
@click.option("--output-dir", type=str, default=".", help="Must be a directory.")
@click.option("--max-batch-tokens", type=int, default=65536, help="Number of sequences to score in a batch.")
@click.option("--fsdp", "fsdp", is_flag=True, help="Use FSDP.")
@click.option("--temperature", type=float, default=0.2, help="Temperature for generation.")
@click.option("--top-p", type=float, default=0.95, help="Top-p for generation.")
def generate(
    prompt_file: str,
    model_name: str,
    n_per_prompt: int,
    output_dir: str,
    max_batch_tokens: int,
    fsdp: bool,
    temperature: float,
    top_p: float,
) -> None:
    if not dist.is_initialized() and fsdp:
        raise ValueError("Distributed training is not initialized but fsdp is set to True.")
    model = get_progen3_model(model_name, use_fsdp=fsdp)
    generator = ProGen3Generator(
        model=model,
        max_batch_tokens=max_batch_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    generator.run(prompt_file, output_dir, n_per_prompt)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)s:%(funcName)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    dist.setup_dist()
    try:
        generate()
    finally:
        dist.destroy_process_group()
