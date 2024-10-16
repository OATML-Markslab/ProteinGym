import gc
import logging
import os
from pathlib import Path

import rich.progress as progress
import torch
import torch.multiprocessing as mp
import torch.optim.lr_scheduler
import typer
import wandb
from dvc.api import params_show
from typing_extensions import Annotated

from vespag.utils import get_device, get_precision, load_model, setup_logger

from .dataset import PerResidueDataset
from .trainer import Trainer


def capitalize_embedding_type(embedding_type: str) -> str:
    return {"prott5": "ProtT5", "esm2": "ESM2"}[embedding_type]


def capitalize_dataset(dataset: str) -> str:
    return {
        "human": "Human",
        "ecoli": "Ecoli",
        "bodo-saltans": "Bodo-Saltans",
        "all": "All",
        "virus": "Virus",
        "droso": "Droso",
    }[dataset]


def train(
    model_config_key: Annotated[str, typer.Option("--model")],
    datasets: Annotated[list[str], typer.Option("--dataset")],
    output_dir: Annotated[Path, typer.Option("--output-dir", "-o")],
    embedding_type: Annotated[str, typer.Option("--embedding-type", "-e")],
    compute_full_train_loss: Annotated[bool, typer.Option("--full-train-loss")] = False,
    sampling_strategy: Annotated[str, typer.Option("--sampling-strategy")] = "basic",
    wandb_config: Annotated[tuple[str, str], typer.Option("--wandb")] = None,
    limit_cache: Annotated[bool, typer.Option("--limit-cache")] = False,
    use_full_dataset: Annotated[bool, typer.Option("--use-full-dataset")] = False,
):
    logger = setup_logger()
    wandb_logger = logging.getLogger("wandb")
    wandb_logger.setLevel(logging.INFO)
    device = get_device()
    precision = get_precision()
    logger.info(f"Using device {str(device)} with precision {precision}")

    params = params_show()

    torch.manual_seed(params["random"]["seed"])
    training_parameters = params["models"][model_config_key]["training_parameters"]
    training_batch_size = training_parameters["batch_size"]["training"]
    validation_batch_size = training_parameters["batch_size"]["validation"]
    learning_rate = training_parameters["learning_rate"]
    epochs = training_parameters["epochs"]
    val_every_epoch = training_parameters["val_every_epoch"]
    checkpoint_every_epoch = training_parameters["checkpoint_every_epoch"] or 999999
    dataset_parameters = params["datasets"]

    logger.info("Loading training data")
    max_len = 4096 if embedding_type == "esm2" else 99999
    train_datasets = {
        dataset: PerResidueDataset(
            dataset_parameters["train"][dataset]["embeddings"][embedding_type],
            dataset_parameters["train"][dataset]["gemme"],
            (
                dataset_parameters["train"][dataset]["splits"]["train"]
                if not use_full_dataset
                else dataset_parameters["train"][dataset]["splits"]["full"]
            ),
            precision,
            device,
            max_len,
            limit_cache,
        )
        for dataset in datasets
    }
    big_train_dataset = torch.utils.data.ConcatDataset(list(train_datasets.values()))

    if sampling_strategy == "basic":
        train_dl = torch.utils.data.DataLoader(
            big_train_dataset, batch_size=training_batch_size, shuffle=True
        )
    else:
        # TODO implement properly
        # TODO factor out
        epoch_size = len(big_train_dataset)  # TODO read from config if provided
        train_weights = [
            1 / (len(dataset) / len(big_train_dataset)) * (1 / row["cluster_size"])
            for dataset in big_train_dataset.datasets
            for row in dataset.cluster_df.rows(named=True)
            for aa in row["seq"][:max_len]
        ]
        train_dl = torch.utils.data.DataLoader(
            big_train_dataset,
            batch_size=training_batch_size,
            sampler=torch.utils.data.WeightedRandomSampler(
                train_weights, epoch_size, replacement=True
            ),
            shuffle=True,
        )

    train_eval_dls = (
        {
            name: torch.utils.data.DataLoader(
                dataset, batch_size=validation_batch_size, shuffle=False
            )
            for name, dataset in train_datasets.items()
        }
        if not use_full_dataset
        else None
    )

    logger.info("Loading validation data")
    if not use_full_dataset:
        val_datasets = {
            dataset: PerResidueDataset(
                dataset_parameters[dataset]["embeddings"][embedding_type],
                dataset_parameters[dataset]["gemme_dir"],
                dataset_parameters[dataset]["splits"]["val"],
                precision,
                device,
                max_len,
                limit_cache,
            )
            for dataset in datasets
        }
        val_dls = {
            name: torch.utils.data.DataLoader(
                dataset, batch_size=validation_batch_size, shuffle=False
            )
            for name, dataset in val_datasets.items()
        }
    else:
        val_dls = None

    architecture = params["models"][model_config_key]["architecture"]
    model_parameters = params["models"][model_config_key]["model_parameters"]
    model = load_model(architecture, model_parameters, embedding_type)

    model = model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=epochs / 25, factor=0.33
    )

    if wandb_config:
        logger.info("Setting up WandB")
        config = {"datasets": datasets, **params["models"][model_config_key]}
        run_name = f"{'+'.join([capitalize_dataset(dataset) for dataset in datasets])} {model_config_key.upper()} {capitalize_embedding_type(embedding_type)}"
        run = wandb.init(
            entity=wandb_config[0],
            project=wandb_config[1],
            config=config,
            name=run_name,
        )
        logger.info(f"Saving WandB run ID to {output_dir}/wandb_run_id.txt")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "wandb_run_id.txt", "w+") as f:
            f.write(run.id)

        wandb.define_metric("step")
        wandb.define_metric("train/batch_loss", step_metric="step")
        wandb.define_metric("batch")

        wandb.define_metric("epoch")
        wandb.define_metric("learning_rate", step_metric="epoch")
        for dataset in datasets + ["overall"]:
            wandb.define_metric(
                f"train/{dataset}/loss", step_metric="epoch", summary="min"
            )
            wandb.define_metric(
                f"train/{dataset}/spearman", step_metric="epoch", summary="max"
            )
            wandb.define_metric(
                f"val/{dataset}/loss", step_metric="epoch", summary="min"
            )
            wandb.define_metric(
                f"val/{dataset}/spearman", step_metric="epoch", summary="max"
            )
        wandb_cache_dir = Path.cwd() / ".wandb/cache"
        wandb_cache_dir.mkdir(exist_ok=True, parents=True)
        os.environ["WANDB_CACHE_DIR"] = str(wandb_cache_dir)

    threads = mp.cpu_count()
    mp.set_start_method("spawn", force=True)

    with progress.Progress(
        *progress.Progress.get_default_columns(), progress.TimeElapsedColumn()
    ) as pbar, mp.Pool(threads) as pool:
        print()
        progress_task_id = pbar.add_task("Training", total=epochs)
        trainer = Trainer(
            run.id,
            model,
            device,
            pool,
            train_dl,
            train_eval_dls,
            val_dls,
            optimizer,
            lr_scheduler,
            criterion,
            pbar,
            output_dir,
            logger,
            use_wandb=True if wandb_config else False,
        )
        trainer.on_train_start()
        for epoch in range(epochs):
            trainer.train_epoch()
            if (epoch + 1) % val_every_epoch == 0 and not use_full_dataset:
                trainer.val_epoch()
                if compute_full_train_loss:
                    trainer.train_eval_epoch()
            if (epoch + 1) % checkpoint_every_epoch == 0:
                trainer.save_state_dict(f"epoch-{epoch}")
            pbar.advance(progress_task_id)
        trainer.on_train_end()

    gc.collect()
    torch.cuda.empty_cache()
    wandb.finish()


if __name__ == "__main__":
    typer.run(train)
