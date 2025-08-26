import argparse
import gzip
import itertools
import os
import requests
import shutil
import tempfile
import time
import traceback
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Any, Sequence, cast
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats
import torch

from tqdm import tqdm

import boto3
import ray
import ray.data
import ray.exceptions

from openprotein.protein import Protein

from poet_2.alphabet.sparse_uniref_cluster2 import Alphabet
from poet_2.models.modules.packed_sequence import PackedTensorSequences
from poet_2.msa.sampling import MSASampler, NeighborsSampler

from utils import (
    get_numpy_seed,
    get_names_and_seqs_from_fastalike,
    get_encoded_msa_from_a3m_seqs,
    hash_of_list,
)


FETCH_NUM_WORKERS = int(os.environ.get("FETCH_NUM_WORKERS", "16"))
PARSE_NUM_WORKERS = int(os.environ.get("PARSE_NUM_WORKERS", "16"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))
VERBOSE = os.environ.get("VERBOSE", "1") == "1"
DEBUG = os.environ.get("DEBUG", "0") == "1"
LOAD_QUERY_ONLY = os.environ.get("LOAD_QUERY_ONLY", "0") == "1"
SAMPLE_PROMPTS_ONLY = os.environ.get("SAMPLE_PROMPTS_ONLY", "0") == "1"

DEFAULT_BATCH_SIZE = int(os.environ.get("DEFAULT_BATCH_SIZE", "128"))
N_SUBSAMPLE = int(os.environ.get("N_SUBSAMPLE", "0"))
NO_FETCH_STRUCTURE = os.environ.get("NO_FETCH_STRUCTURE", "0") == "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="~/.cache/ProteinGym/baselines/PoET-2/poet-2.ckpt",
    )
    parser.add_argument(
        "--DMS_reference_file_path",
        type=str,
        default="reference_files/DMS_substitutions.csv",
    )
    parser.add_argument(
        "--DMS_data_folder",
        type=Path,
        default="~/.cache/ProteinGym/DMS_ProteinGym_substitutions",
    )
    parser.add_argument(
        "--DMS_structure_folder",
        type=Path,
        default="~/.cache/ProteinGym/ProteinGym_AF2_structures",
    )
    parser.add_argument("--DMS_index", type=int, default=0)
    parser.add_argument(
        "--output_scores_folder",
        type=Path,
        default="~/.cache/ProteinGym/zero_shot_substitutions_scores/PoET-2",
    )
    parser.add_argument(
        "--MSA_folder",
        type=Path,
        default="~/.cache/ProteinGym/baselines/PoET/scripts/data/msas/DMS_substitutions",
    )
    parser.add_argument(
        "--AF2_cache_folder",
        type=Path,
        default="~/.cache/ProteinGym/baselines/PoET-2/AF2",
    )
    parser.add_argument("--theta", type=float, default=0.2)
    parser.add_argument(
        "--context_length", type=int, nargs="+", default=[6144, 12288, 24576]
    )
    parser.add_argument(
        "--max_similarity", type=float, nargs="+", default=[1.0, 0.95, 0.90, 0.70, 0.50]
    )
    parser.add_argument(
        "--structure_in_context", type=int, nargs="+", default=[1], choices=[0, 1]
    )
    parser.add_argument(
        "--inverse_folding_query", type=int, nargs="+", default=[0, 1], choices=[0, 1]
    )
    parser.add_argument("--relative_to_wt", action="store_true")
    parser.add_argument(
        "--batch_size", type=int, default=8 if DEBUG else DEFAULT_BATCH_SIZE
    )
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if not args.checkpoint.startswith("s3://"):
        args.checkpoint = Path(args.checkpoint).expanduser()
        assert args.checkpoint.is_file()

    args.DMS_data_folder = args.DMS_data_folder.expanduser()
    assert args.DMS_data_folder.is_dir()

    args.DMS_structure_folder = args.DMS_structure_folder.expanduser()
    assert args.DMS_structure_folder.is_dir()

    args.output_scores_folder = args.output_scores_folder.expanduser()
    args.output_scores_folder.mkdir(parents=True, exist_ok=True)

    args.MSA_folder = args.MSA_folder.expanduser()
    assert args.MSA_folder.is_dir()

    args.AF2_cache_folder = args.AF2_cache_folder.expanduser()
    args.AF2_cache_folder.mkdir(parents=True, exist_ok=True)

    for boolean_list_arg_values in [
        args.structure_in_context,
        args.inverse_folding_query,
    ]:
        assert len(set(boolean_list_arg_values)) == len(
            boolean_list_arg_values
        ), "should not have duplicate values"
    args.structure_in_context = [bool(x) for x in args.structure_in_context]
    args.inverse_folding_query = [bool(x) for x in args.inverse_folding_query]
    return args


def sample_context(
    msa_names: list[bytes],
    msa_sequences: list[bytes],
    msa: npt.NDArray[np.uint8],
    theta: float,
    max_tokens: int,
    max_similarity: float,
    structure_in_context: bool,  # NB: unused here as structures are fetched later
    inverse_folding_query: bool,
    seed: int,
    sampling_weights_cache_dir: Path,
) -> list[Protein]:
    alphabet = Alphabet()
    sampler = MSASampler(
        method=NeighborsSampler(can_use_torch=False, theta=theta),
        force_include_first=inverse_folding_query,
        max_similarity=max_similarity,
    )
    sample_idxs = sampler.get_sample_idxs(
        msa=msa,
        gap_token=alphabet.gap_token,
        seed=seed,
        result_cache_dir=sampling_weights_cache_dir,
    )
    if inverse_folding_query:
        # NB: don't include wt in context
        assert sample_idxs[0] == 0
        sample_idxs = sample_idxs[1:]
    prompt: list[Protein] = []
    total_tokens = 0
    for idx in cast(Sequence[int], sample_idxs):
        sequence = msa_sequences[idx].upper().translate(None, delete=b"-")
        this_n_tokens = len(sequence) + 2
        if this_n_tokens + total_tokens > max_tokens:
            break
        prompt.append(Protein(name=msa_names[idx], sequence=sequence))
        total_tokens += this_n_tokens
    # shuffle order
    rng = np.random.RandomState(get_numpy_seed(f"{seed+1}"))
    return [prompt[i] for i in rng.permutation(len(prompt))]


def _fetch_structure(name: str, cache_dir: Path) -> str:
    cache_path = cache_dir / f"{name}.cif.gz"
    if cache_path.is_file():
        return name
    if NO_FETCH_STRUCTURE:
        raise ValueError(
            f"fetch structure disabled, but structure not cached at {cache_path}"
        )

    url = f"https://alphafold.ebi.ac.uk/files/AF-{name}-F1-model_v4.cif"
    response = requests.get(url)
    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            cache_path.touch()
            return name
        else:
            raise
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=cache_dir) as tmp:
        with gzip.GzipFile(fileobj=tmp, mode="wb") as gz:
            gz.write(response.content)
        temp_path = Path(tmp.name)
    try:
        temp_path.rename(cache_path)
    except FileExistsError:
        # Another process beat us to writing the file
        temp_path.unlink()  # Clean up our temp file
    return name


def fetch_structures(proteins: list[Protein], cache_dir: Path) -> dict[str, Protein]:
    proteins_with_structure: dict[str, Protein] = {}
    with ThreadPoolExecutor(max_workers=FETCH_NUM_WORKERS) as executor:
        seen_names: set[str] = set()
        futures: list[Future] = []
        for protein in proteins:
            assert protein.name is not None
            if protein.name.startswith("UPI"):
                continue
            if protein.name in seen_names:
                continue
            future = executor.submit(
                _fetch_structure, name=protein.name, cache_dir=cache_dir
            )
            seen_names.add(protein.name)
            futures.append(future)
        with ProcessPoolExecutor(max_workers=PARSE_NUM_WORKERS) as pool:
            parse_futures = []
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                mininterval=1.0,
                desc="Downloading structures...",
            ):
                name = future.result()
                cache_path = cache_dir / f"{name}.cif.gz"
                if cache_path.stat().st_size == 0:
                    continue
                future = pool.submit(
                    Protein.from_filepath, path=cache_path, chain_id="A"
                )
                parse_futures.append(future)
            for future in tqdm(
                as_completed(parse_futures),
                total=len(parse_futures),
                mininterval=1.0,
                desc="Parsing structures...",
            ):
                protein = future.result()
                assert isinstance(protein, Protein)
                assert protein.name is not None
                assert protein.name.endswith(".cif")
                name = protein.name.removesuffix(".cif")
                protein.name = name
                proteins_with_structure[name] = protein
    return proteins_with_structure


def load_query(
    DMS_structure_folder: Path, pdb_filename: str, wt_sequence: str
) -> Protein:
    query = Protein.from_filepath(DMS_structure_folder / pdb_filename, chain_id="A")
    assert query.name is not None
    assert not query.name.endswith(".pdb")
    wt_sequence_encoded = wt_sequence.encode()
    if pdb_filename == "CAS9_STRP1.pdb":
        assert len(query.sequence) > len(wt_sequence_encoded)
        assert (
            wt_sequence_encoded in query.sequence
        ), "for CAS9_STRP1, wt sequence is a subsequence of pdb sequence instead"
        assert query.sequence.startswith(wt_sequence_encoded)
        print(
            f"Recreating query {pdb_filename=} {len(wt_sequence)=} {len(query.sequence)=}"
        )
        query = query[: len(wt_sequence_encoded)]
    elif (
        pdb_filename == "P53_HUMAN.pdb"
        and wt_sequence_encoded
        == b"MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPRVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"
    ):
        # NB: there is a known sequence mismatch in this case...
        assert query.sequence != wt_sequence_encoded
    else:
        assert (
            query.sequence in wt_sequence_encoded
        ), f"pdb sequence must be subsequence of wt {pdb_filename=}"
        if query.sequence != wt_sequence_encoded:
            assert pdb_filename.endswith(
                ".pdb"
            ), "expect this only for DMS, not clinical"
            idx = wt_sequence_encoded.index(query.sequence)
            print(
                f"Recreating query {pdb_filename=} {len(wt_sequence)=} {len(query.sequence)=} {idx=}"
            )
            new_query = Protein(sequence=wt_sequence_encoded, name=query.name)
            new_query.coordinates[idx : idx + len(query)] = query.coordinates
            new_query.plddt[idx : idx + len(query)] = query.plddt
            query = new_query
    assert len(query.sequence) == len(wt_sequence_encoded)
    query.sequence = b"X" * len(query)
    return query


def download_s3_object(s3_path: str) -> Path:
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path format. Must start with 's3://'.")
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc
    s3_key = parsed_url.path.lstrip("/")
    if not bucket_name or not s3_key:
        raise ValueError(
            f"Invalid S3 path: '{s3_path}'. Could not parse bucket or key."
        )
    # Determine the local filename (just the basename of the S3 key)
    local_filename = os.path.basename(s3_key)
    current_directory = os.getcwd()
    local_filepath = os.path.join(current_directory, local_filename)
    # Check if the file already exists locally
    if os.path.exists(local_filepath):
        return Path(local_filepath)
    # If not, download it
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket_name, s3_key, local_filepath)
    return Path(local_filepath)


class InferenceWorker:
    def __init__(
        self,
        checkpoint_path: Path | str,
        prompts: dict[
            tuple[int, float, bool, bool, int], tuple[list[Protein], Protein | None]
        ],
        variants: list[str],
        batch_size: int,
    ):
        import poet_2.models.poet_2_helpers as helpers

        self.prompts: dict[
            tuple[int, float, bool, bool, int], tuple[list[Protein], Protein | None]
        ] = ray.get(
            prompts  # type: ignore
        )
        self.variants: list[str] = ray.get(variants)  # type: ignore
        self.prompt_idx_to_key = {
            idx: prompt_key for idx, prompt_key in enumerate(self.prompts.keys())
        }
        self.batch_size = batch_size

        self.device = torch.device("cuda")
        if isinstance(checkpoint_path, str):
            assert checkpoint_path.startswith("s3://")
            checkpoint_path = download_s3_object(checkpoint_path)
        self.checkpoint_path = checkpoint_path
        self.model = helpers.load_model(path=self.checkpoint_path, device=self.device)

        self._current_prompt_idx: int | None = None
        self._current_prompt: tuple[
            list[PackedTensorSequences], torch.Tensor, Protein | None
        ]

    def _set_current_prompt(self, prompt_idx: int):
        import poet_2.models.poet_2_helpers as helpers

        if self._current_prompt_idx == prompt_idx:
            return self._current_prompt
        context, query = self.prompts[self.prompt_idx_to_key[prompt_idx]]
        inputs_: list[helpers.NamedInput] = []
        if query is not None:
            inputs_.append(
                helpers.NamedInput(
                    sequence=query.sequence,
                    plddt=query.plddt.copy(),
                    atomx=query.coordinates[:, :3].copy(),
                ).enforce()
            )
        for protein in context:
            inputs_.append(
                helpers.NamedInput(
                    sequence=protein.sequence,
                    plddt=protein.plddt.copy(),
                    atomx=protein.coordinates[:, :3].copy(),
                ).enforce()
            )
        memory, ys_ref_values = helpers.compute_memory(
            model=self.model, prompts=[inputs_]
        )
        self._current_prompt_idx = prompt_idx
        self._current_prompt = (memory, ys_ref_values, query)
        return self._current_prompt

    def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        import poet_2.models.poet_2_helpers as helpers

        prompt_idx: int
        variants_start_idx: int
        variants_end_idx: int
        prompt_idx, variants_start_idx, variants_end_idx = row["item"]
        with torch.inference_mode():
            memory, ys_ref_values, query = self._set_current_prompt(
                prompt_idx=prompt_idx
            )
            result = helpers.score_sequences(
                model=self.model,
                memory=memory,
                sequences=self.variants[variants_start_idx:variants_end_idx],
                ys_ref_values=ys_ref_values if query is not None else None,
                self_prompt=query.sequence if query is not None else None,
                batch_size=self.batch_size,
            )
        row["result"] = result.cpu().float().numpy()
        return row


@ray.remote(num_cpus=0.08, max_retries=-1)
def compute(
    prompts_lst: list[
        dict[tuple[int, float, bool, bool, int], tuple[list[Protein], Protein | None]]
    ],
    variants_to_score_lst: list[list[str]],
    n_prompts: int,
    n_variants_to_score: int,
    tasks: list[tuple[int, int, int]],
    checkpoint_path: Path | str,
    batch_size: int,
    num_workers: int,
) -> npt.NDArray[np.float32]:
    while True:
        try:
            # NB: these are passed in as lists of length 1 to make them object refs
            prompts, variants_to_score = prompts_lst[0], variants_to_score_lst[0]
            scores = np.empty((n_variants_to_score, n_prompts), dtype=np.float32)
            for row in (
                ray.data.from_items(tasks, override_num_blocks=len(tasks))
                .map(
                    fn=InferenceWorker,  # type: ignore
                    fn_constructor_kwargs=dict(
                        checkpoint_path=checkpoint_path,
                        prompts=prompts,
                        variants=variants_to_score,
                        batch_size=batch_size,
                    ),
                    num_gpus=1,
                    concurrency=(num_workers if NUM_WORKERS == 0 else 1, num_workers),
                    max_restarts=-1,
                    max_task_retries=-1,
                )
                .iter_rows()
            ):
                prompt_idx, start_idx, end_idx = row["item"]
                scores[start_idx:end_idx, prompt_idx] = row["result"]
            return scores
        except ray.exceptions.GetTimeoutError:
            time.sleep(1)
            print(traceback.format_exc())
            continue


def main():
    num_workers = NUM_WORKERS or torch.cuda.device_count()
    if num_workers == 0:
        raise ValueError(f"{num_workers=}")
    args = parse_args()
    is_indels = "indels" in args.DMS_reference_file_path
    if is_indels:
        assert args.inverse_folding_query == [0]
    # get variants to score
    ref_series = pd.read_csv(args.DMS_reference_file_path).iloc[args.DMS_index]
    if not pd.isna(ref_series["MSA_start"]):
        msa_start = int(ref_series["MSA_start"])
    else:
        msa_start = 1
    if not pd.isna(ref_series["MSA_end"]):
        msa_end = int(ref_series["MSA_end"])
    else:
        msa_end = len(ref_series["target_seq"])
    wt_sequence: str = ref_series["target_seq"]
    wt_sequence_short = wt_sequence[msa_start - 1 : msa_end]
    variants_filename: str = ref_series["DMS_filename"]
    if (args.output_scores_folder / variants_filename).is_file():
        if VERBOSE:
            print(
                f"Output score file {args.output_scores_folder / variants_filename} already exists, skipping..."
            )
        return
    if not LOAD_QUERY_ONLY and not SAMPLE_PROMPTS_ONLY:
        ray.init()
    variants_df = pd.read_csv(args.DMS_data_folder / variants_filename)
    if N_SUBSAMPLE > 0:
        variants_df = variants_df.sample(
            n=min(N_SUBSAMPLE, len(variants_df)),
            replace=False,
            random_state=get_numpy_seed(
                (args.DMS_data_folder / variants_filename).stem
            ),
        )
    if "mutated_sequence" in variants_df.columns:
        variant_sequences = variants_df["mutated_sequence"].values
    elif "mutant" in variants_df.columns:
        variant_sequences = variants_df["mutant"].values
    else:
        raise ValueError("did not find sequence column in DMS file")
    variant_sequences = cast(list[str], list(variant_sequences))
    if not is_indels:
        assert "substitutions" in args.DMS_reference_file_path
        assert all(len(s) == len(wt_sequence) for s in variant_sequences)
    # load msa
    old_variants_filename = (
        variants_filename.replace("F7YBW8_MESOW_Ding_2023", "F7YBW7_MESOW_Ding_2023")
        .replace("PICP2", "SYNP2")
        .replace("Q6WV12", "Q6WV13")
    )
    msa_filepath = (args.MSA_folder / old_variants_filename).with_suffix(".a3m.zst")
    msa_names, msa_sequences = get_names_and_seqs_from_fastalike(msa_filepath)
    msa_names = [n.split(b"\t", 1)[0].split(b"_")[-1] for n in msa_names]
    if msa_sequences[0].decode() != wt_sequence_short:
        assert (
            ("clinical" in args.DMS_reference_file_path)
            and is_indels
            and (args.DMS_index in (705, 1150))
        )
    msa = get_encoded_msa_from_a3m_seqs(
        msa_sequences=msa_sequences, alphabet=Alphabet()
    )
    # load query
    if not is_indels:
        if "DMS" in args.DMS_reference_file_path:
            query = load_query(
                DMS_structure_folder=args.DMS_structure_folder,
                pdb_filename=ref_series["pdb_file"],
                wt_sequence=wt_sequence,
            )
        elif "clinical" in args.DMS_reference_file_path:
            query = load_query(
                DMS_structure_folder=args.DMS_structure_folder,
                pdb_filename=ref_series["DMS_id"] + ".cif.gz",
                wt_sequence=wt_sequence,
            )
        else:
            raise ValueError(args.DMS_reference_file_path)
    else:
        query = None
    # generate prompts
    prompts: dict[
        tuple[int, float, bool, bool, int], tuple[list[Protein], Protein | None]
    ] = {}
    for (
        max_tokens,
        max_similarity,
        structure_in_context,
        inverse_folding_query,
    ) in itertools.product(
        args.context_length,
        args.max_similarity,
        args.structure_in_context,
        args.inverse_folding_query,
    ):
        seed_offset = 0
        while (
            key := (
                max_tokens,
                max_similarity,
                structure_in_context,
                inverse_folding_query,
                seed_offset,
            )
        ) in prompts.keys():
            seed_offset += 1
        name = f"{max_tokens}_{max_similarity}_{structure_in_context}_{inverse_folding_query}"
        name = f"{name}_seed={args.seed + seed_offset}"
        context = sample_context(
            msa_names=msa_names,
            msa_sequences=msa_sequences,
            msa=msa,
            theta=args.theta,
            max_tokens=max_tokens,
            max_similarity=max_similarity,
            structure_in_context=structure_in_context,
            inverse_folding_query=inverse_folding_query,
            seed=get_numpy_seed(name),
            sampling_weights_cache_dir=args.MSA_folder,
        )
        prompts[key] = (context, query if inverse_folding_query else None)
        if VERBOSE:
            context_hash = hash_of_list([c.sequence for c in context])
            print(f"Sampled prompt {name}: {context_hash}")
    if LOAD_QUERY_ONLY:
        return
    # fetch structures
    need_structure_proteins: list[Protein] = []
    for (_, _, structure_in_context, _, _), (context, _) in prompts.items():
        if not structure_in_context:
            continue
        need_structure_proteins.extend(context)
    proteins_with_structure = fetch_structures(
        need_structure_proteins, cache_dir=args.AF2_cache_folder
    )
    # replace context proteins with proteins with structure as needed
    for (
        max_tokens,
        max_similarity,
        structure_in_context,
        inverse_folding_query,
        seed_offset,
    ), (context, _) in prompts.items():
        if not structure_in_context:
            continue
        n_replaced = 0
        for i in range(len(context)):
            name = context[i].name
            assert name is not None
            if name in proteins_with_structure:
                n_replaced += 1
            context[i] = proteins_with_structure.get(name, context[i])
        if VERBOSE:
            name = f"{max_tokens}_{max_similarity}_{structure_in_context}_{inverse_folding_query}"
            name = f"{name}_seed={args.seed + seed_offset}"
            context_hash = hash_of_list([c.sequence for c in context])
            print(
                f"Sampled prompt {name} ({n_replaced}/{len(context)}): {context_hash}"
            )
    # enforce max tokens again, as number of tokens may have changed after the above
    print("Reenforcing max tokens...")
    context_hashes: dict[tuple[int, float, bool, bool, int], str] = {}
    for key, (context, query) in prompts.items():
        (
            max_tokens,
            max_similarity,
            structure_in_context,
            inverse_folding_query,
            seed_offset,
        ) = key
        if structure_in_context:
            new_context: list[Protein] = []
            total_tokens = 0
            for protein in context:
                this_n_tokens = len(protein) + 2
                if this_n_tokens + total_tokens > max_tokens:
                    break
                new_context.append(protein)
                total_tokens += this_n_tokens
            n_old_context, n_new_context = len(context), len(new_context)
            context = new_context
            del new_context
            prompts[key] = (context, query)
        else:
            n_old_context, n_new_context = len(context), len(context)
        # save context hash
        context_hash = hash_of_list([c.sequence for c in context])
        context_hashes[key] = context_hash
        if VERBOSE:
            name = f"{max_tokens}_{max_similarity}_{structure_in_context}_{inverse_folding_query}"
            name = f"{name}_seed={args.seed + seed_offset}"
            print(
                f"Sampled prompt {name} ({n_new_context}/{n_old_context}): {context_hash}"
            )
    print("Trimming coordinates to backbone only to save memory...")
    for context, query in prompts.values():
        if query is not None:
            query.coordinates = query.coordinates[:, :3]
        for protein in context:
            protein.coordinates = protein.coordinates[:, :3]
    if SAMPLE_PROMPTS_ONLY:
        return

    # score variants
    variants_to_score = variant_sequences
    if args.relative_to_wt:
        variants_to_score = variant_sequences + [wt_sequence]
    # distribute work and gather results
    max_tasks = 1000
    min_metabatch_size = len(prompts) * len(variants_to_score) // max_tasks
    metabatch_size = max(min_metabatch_size, 1024)
    tasks = []
    for prompt_idx in range(len(prompts)):
        for start_idx in range(0, len(variants_to_score), metabatch_size):
            tasks.append((prompt_idx, start_idx, start_idx + metabatch_size))
    print(f"{len(prompts)=} {len(variants_to_score)=} {metabatch_size=} {len(tasks)=}")
    scores = ray.get(
        compute.remote(
            [ray.put(prompts)],
            [ray.put(variants_to_score)],
            len(prompts),
            len(variants_to_score),
            tasks,
            args.checkpoint,
            args.batch_size,
            num_workers,
        )
    )
    if is_indels:
        lengths = np.array([len(s) for s in variants_to_score], dtype=np.float32)
        scores = scores + 1.96 * lengths[:, np.newaxis]
    if args.relative_to_wt:
        scores = scores[:-1] - scores[-1]
    scores = np.concat((scores, scores.mean(axis=1)[:, np.newaxis]), axis=1)
    # save results
    df = pd.DataFrame(
        data=scores,
        columns=[
            "_".join([str(x) for x in prompt_key] + [context_hashes[prompt_key]])
            for prompt_key in prompts.keys()
        ]
        + ["PoET-2"],
    )
    if "DMS_score" in variants_df.columns:
        print(
            stats.spearmanr(
                df["PoET-2"].to_numpy(), variants_df["DMS_score"].to_numpy()
            )
        )
    df.to_csv(args.output_scores_folder / variants_filename, index=False)


if __name__ == "__main__":
    main()
