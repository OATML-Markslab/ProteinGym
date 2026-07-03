from __future__ import annotations

import json
import math
import os
import re
import time
from json import JSONDecodeError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(AA_ALPHABET)}

# Simple biochemical scales are enough for audit/features without external dependencies.
AA_HYDROPATHY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

AA_VOLUME = {
    "A": 88.6,
    "C": 108.5,
    "D": 111.1,
    "E": 138.4,
    "F": 189.9,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "K": 168.6,
    "L": 166.7,
    "M": 162.9,
    "N": 114.1,
    "P": 112.7,
    "Q": 143.8,
    "R": 173.4,
    "S": 89.0,
    "T": 116.1,
    "V": 140.0,
    "W": 227.8,
    "Y": 193.6,
}

MUTATION_PATTERN = re.compile(r"([A-Z*])(\d+)([A-Z*])")


@dataclass
class FoldMetrics:
    fold: int
    train_n: int
    test_n: int
    mse: float
    r2: float
    spearman: float
    elapsed_seconds: float
    raw_mse: float | None = None
    target_train_mean: float | None = None
    target_train_std: float | None = None


@dataclass
class ReproductionResult:
    assay: str
    model: str
    fold_column: str
    feature_source: str
    n_rows: int
    feature_dim: int
    fold_scores: list[FoldMetrics]
    mean_mse: float
    std_mse: float
    mean_r2: float
    std_r2: float
    mean_spearman: float
    std_spearman: float
    mse_scale: str = "train_fold_standardized"


def discover_assay_csvs(csv_root: Path) -> list[Path]:
    """Return ProteinGym CV CSV files while tolerating either root or root/cv layouts."""
    csv_root = Path(csv_root)
    if (csv_root / "cv").is_dir():
        files = sorted((csv_root / "cv").glob("*.csv"))
    else:
        files = sorted(csv_root.glob("*.csv"))
    return [p for p in files if p.name != ".DS_Store"]


def fold_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("fold_")]


def parse_mutations(mutant: str) -> list[tuple[str, int, str]]:
    return [(wt, int(pos), alt) for wt, pos, alt in MUTATION_PATTERN.findall(str(mutant))]


def _aa_composition(sequence: str) -> np.ndarray:
    counts = np.zeros(len(AA_ALPHABET), dtype=np.float32)
    if not isinstance(sequence, str) or not sequence:
        return counts
    for aa in sequence:
        idx = AA_TO_INDEX.get(aa)
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


def build_mutation_features(frame: pd.DataFrame) -> np.ndarray:
    """Build deterministic, lightweight mutation features from ProteinGym CSV columns.

    These features are mainly for local smoke tests and ablations. For faithful reproduction
    of the stored protein-side results, prefer the released `X_float16.dat` feature arrays.
    """
    rows: list[np.ndarray] = []
    for mutant, sequence in zip(frame["mutant"], frame["mutated_sequence"], strict=False):
        muts = parse_mutations(mutant)
        seq_len = len(sequence) if isinstance(sequence, str) else 0
        denom = max(seq_len, 1)
        n_mut = len(muts)
        positions = np.array([pos for _, pos, _ in muts], dtype=np.float32)
        wt_counts = np.zeros(len(AA_ALPHABET), dtype=np.float32)
        alt_counts = np.zeros(len(AA_ALPHABET), dtype=np.float32)
        hyd_deltas: list[float] = []
        vol_deltas: list[float] = []
        for wt, pos, alt in muts:
            if wt in AA_TO_INDEX:
                wt_counts[AA_TO_INDEX[wt]] += 1.0
            if alt in AA_TO_INDEX:
                alt_counts[AA_TO_INDEX[alt]] += 1.0
            hyd_deltas.append(AA_HYDROPATHY.get(alt, 0.0) - AA_HYDROPATHY.get(wt, 0.0))
            vol_deltas.append(AA_VOLUME.get(alt, 0.0) - AA_VOLUME.get(wt, 0.0))
        if n_mut:
            wt_counts /= n_mut
            alt_counts /= n_mut
        summary = np.array(
            [
                seq_len,
                n_mut,
                positions.mean() / denom if n_mut else 0.0,
                positions.min() / denom if n_mut else 0.0,
                positions.max() / denom if n_mut else 0.0,
                float(np.mean(hyd_deltas)) if hyd_deltas else 0.0,
                float(np.mean(np.abs(hyd_deltas))) if hyd_deltas else 0.0,
                float(np.mean(vol_deltas)) if vol_deltas else 0.0,
                float(np.mean(np.abs(vol_deltas))) if vol_deltas else 0.0,
            ],
            dtype=np.float32,
        )
        rows.append(np.concatenate([summary, wt_counts, alt_counts, _aa_composition(sequence)]))
    return np.vstack(rows).astype(np.float32)


def infer_memmap_shape(x_path: Path, n_rows: int, dtype: np.dtype = np.dtype("float16")) -> tuple[int, int]:
    n_items = Path(x_path).stat().st_size // dtype.itemsize
    if n_items % n_rows != 0:
        raise ValueError(f"Cannot infer rectangular shape for {x_path}: {n_items=} {n_rows=}")
    return n_rows, n_items // n_rows


def load_feature_matrix(
    frame: pd.DataFrame,
    assay: str,
    feature_source: str,
    feature_root: Path | None = None,
) -> tuple[np.ndarray, str]:
    if feature_source not in {"auto", "memmap", "csv_mutation"}:
        raise ValueError(f"Unsupported feature source: {feature_source}")
    if feature_source in {"auto", "memmap"} and feature_root is not None:
        assay_dir = Path(feature_root) / assay
        x_path = assay_dir / "X_float16.dat"
        if x_path.exists():
            shape = infer_memmap_shape(x_path, len(frame))
            return np.memmap(x_path, dtype=np.float16, mode="r", shape=shape), "memmap"
        if feature_source == "memmap":
            raise FileNotFoundError(f"Missing feature memmap for assay {assay}: {x_path}")
    return build_mutation_features(frame), "csv_mutation"


def build_regressor(
    model_name: str,
    seed: int,
    tabicl_kwargs: dict | None = None,
    tabpfn3_kwargs: dict | None = None,
):
    key = model_name.lower()
    if key in {"ridge", "linear_ridge"}:
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=seed))
    if key in {"histgradientboostingregressor", "hgb", "hist_gradient_boosting"}:
        return HistGradientBoostingRegressor(random_state=seed, max_iter=300, learning_rate=0.05)
    if key in {"rbf_sampler", "rbf"}:
        return make_pipeline(
            StandardScaler(),
            RBFSampler(gamma=1.0, n_components=512, random_state=seed),
            Ridge(alpha=1.0, random_state=seed),
        )
    if key == "tabicl":
        try:
            try:
                from tabicl import TabICLRegressor
            except ImportError:
                from tabicl.sklearn.regressor import TabICLRegressor
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TabICLRegressor is unavailable. Activate the project environment or install tabicl."
            ) from exc
        device = os.environ.get("TABICL_DEVICE_OVERRIDE") or "cuda"
        if tabicl_kwargs and tabicl_kwargs.get("device") not in {None, "", "auto"}:
            device = tabicl_kwargs["device"]
        elif tabicl_kwargs and tabicl_kwargs.get("device") == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        kwargs = {
            "n_estimators": 8,
            "batch_size": 8,
            "random_state": seed,
            "device": device,
            "verbose": False,
        }
        if tabicl_kwargs:
            kwargs.update(
                {
                    k: v
                    for k, v in tabicl_kwargs.items()
                    if v is not None and k not in {"device", "prediction_chunk_size"}
                }
            )
        disk_offload_dir = kwargs.get("disk_offload_dir")
        offload_mode = kwargs.get("offload_mode")
        if disk_offload_dir and offload_mode == "disk" and "inference_config" not in kwargs:
            try:
                import torch

                disk_dtype = torch.float16
            except Exception:  # pragma: no cover - TabICL imports torch in real runs.
                disk_dtype = None
            # The high-level offload flags only affect column-wise embedding in TabICL.
            # ProteinGym tail assays need all inference managers to spill to NVME.
            manager_config = {
                "offload": "disk",
                "disk_offload_dir": str(disk_offload_dir),
                "disk_flush_mb": 512.0,
                "disk_min_free_mb": 1024.0,
                "disk_cleanup": True,
                "disk_safety_factor": 0.90,
                "max_pinned_memory_mb": 0.0,
            }
            if disk_dtype is not None:
                manager_config["disk_dtype"] = disk_dtype
            kwargs["inference_config"] = {
                "COL_CONFIG": dict(manager_config),
                "ROW_CONFIG": dict(manager_config),
                "ICL_CONFIG": dict(manager_config),
            }
        return TabICLRegressor(**kwargs)
    if key in {"tabpfn3", "tabpfn3_esmc600m"}:
        try:
            from tabpfn import TabPFNRegressor
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "TabPFNRegressor is unavailable. Activate the project environment or install tabpfn."
            ) from exc
        kwargs = {
            "model_path": os.environ.get("TABPFN3_REGRESSOR_MODEL_PATH") or None,
            "device": os.environ.get("TABPFN3_DEVICE_OVERRIDE") or "auto",
            "n_estimators": int(os.environ.get("TABPFN3_N_ESTIMATORS_OVERRIDE", "8")),
            "fit_mode": os.environ.get("TABPFN3_FIT_MODE_OVERRIDE") or "fit_preprocessors",
            "memory_saving_mode": os.environ.get("TABPFN3_MEMORY_SAVING_MODE_OVERRIDE") or "auto",
            "inference_precision": os.environ.get("TABPFN3_INFERENCE_PRECISION_OVERRIDE") or "auto",
            "ignore_pretraining_limits": True,
            "random_state": seed,
            "n_preprocessing_jobs": int(os.environ.get("TABPFN3_N_PREPROCESSING_JOBS_OVERRIDE", "1")),
            "show_progress_bar": False,
        }
        env_ignore_limits = os.environ.get("TABPFN3_IGNORE_PRETRAINING_LIMITS_OVERRIDE")
        if env_ignore_limits is not None:
            kwargs["ignore_pretraining_limits"] = env_ignore_limits.lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if tabpfn3_kwargs:
            kwargs.update(
                {
                    k: v
                    for k, v in tabpfn3_kwargs.items()
                    if v is not None and k != "prediction_chunk_size"
                }
            )
        return TabPFNRegressor(**kwargs)
    raise ValueError(f"Unsupported protein reproduction model: {model_name}")


def predict_regression(
    model,
    x: np.ndarray,
    prediction_chunk_size: int | None = None,
    progress_prefix: str | None = None,
) -> np.ndarray:
    if not prediction_chunk_size or prediction_chunk_size <= 0 or len(x) <= prediction_chunk_size:
        if progress_prefix:
            print(f"{progress_prefix}: predict all n={len(x)}", flush=True)
        return np.asarray(model.predict(x), dtype=np.float32)
    chunks: list[np.ndarray] = []
    total_chunks = int(math.ceil(len(x) / prediction_chunk_size))
    for start in range(0, len(x), prediction_chunk_size):
        stop = min(start + prediction_chunk_size, len(x))
        chunk_idx = len(chunks) + 1
        if progress_prefix:
            print(
                f"{progress_prefix}: predict chunk {chunk_idx}/{total_chunks} "
                f"rows={start}:{stop}",
                flush=True,
            )
        chunks.append(np.asarray(model.predict(x[start:stop]), dtype=np.float32))
        if progress_prefix:
            print(
                f"{progress_prefix}: finished chunk {chunk_idx}/{total_chunks}",
                flush=True,
            )
    return np.concatenate(chunks, axis=0)


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return math.nan
    return float(spearmanr(y_true, y_pred, nan_policy="omit").statistic)


def regression_fold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
        "spearman": _spearman(y_true, y_pred),
    }


def train_fold_standardized_mse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return MSE on the training-fold z-scale plus raw-scale diagnostics.

    The retained ProteinGym manuscript summaries report MSE on a normalized
    target scale. Spearman is insensitive to this, but raw-scale MSE can explode
    for assays whose DMS scores are measured on very large numerical ranges.
    """
    train_mean = float(np.nanmean(y_train))
    train_std = float(np.nanstd(y_train))
    if not np.isfinite(train_std) or train_std <= 0:
        train_std = 1.0
    raw_mse = float(mean_squared_error(y_true, y_pred))
    y_true_z = (y_true - train_mean) / train_std
    y_pred_z = (y_pred - train_mean) / train_std
    return float(mean_squared_error(y_true_z, y_pred_z)), raw_mse, train_mean, train_std


def run_proteingym_assay(
    csv_path: Path,
    out_dir: Path,
    model_name: str,
    fold_column: str,
    feature_source: str = "auto",
    feature_root: Path | None = None,
    feature_pca_components: int | None = None,
    fold_values: Iterable[int] | None = None,
    seed: int = 0,
    force: bool = False,
    tabicl_kwargs: dict | None = None,
    tabpfn3_kwargs: dict | None = None,
) -> ReproductionResult:
    csv_path = Path(csv_path)
    assay = csv_path.stem
    selected_fold_values = None
    if fold_values is not None:
        selected_fold_values = tuple(sorted(int(fold) for fold in fold_values))

    if selected_fold_values:
        fold_token = "fold_" + "_".join(str(fold) for fold in selected_fold_values)
        out_root = Path(out_dir) / assay / fold_column / fold_token / model_name
    else:
        out_root = Path(out_dir) / assay / fold_column / model_name
    summary_path = out_root / "summary.json"
    prediction_path = out_root / "per_mutant_predictions.csv"
    if summary_path.exists() and prediction_path.exists() and not force:
        try:
            return result_from_json(summary_path)
        except (JSONDecodeError, TypeError, KeyError, ValueError):
            # Older Slurm manifests accidentally used summary.json as the done marker.
            # If that happened, rerun and replace the corrupted cache.
            pass

    out_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_path)
    if fold_column not in frame.columns:
        raise ValueError(f"{csv_path} has no fold column {fold_column}")
    x, actual_feature_source = load_feature_matrix(frame, assay, feature_source, feature_root)
    output_feature_source = actual_feature_source
    output_feature_dim = int(x.shape[1])
    if feature_pca_components is not None:
        if feature_pca_components < 1:
            raise ValueError("feature_pca_components must be >= 1")
        if feature_pca_components >= int(x.shape[1]):
            raise ValueError(
                f"feature_pca_components={feature_pca_components} must be smaller than "
                f"feature_dim={int(x.shape[1])}"
            )
        output_feature_source = f"{actual_feature_source}_pca{feature_pca_components}"
        output_feature_dim = int(feature_pca_components)
    y = frame["DMS_score"].to_numpy(dtype=np.float32)
    folds = sorted(frame[fold_column].dropna().astype(int).unique().tolist())
    if selected_fold_values is not None:
        wanted = set(selected_fold_values)
        folds = [fold for fold in folds if fold in wanted]
    if not folds:
        raise ValueError(f"No folds selected for {assay} {fold_column}")

    predictions = np.full(len(frame), np.nan, dtype=np.float32)
    fold_scores: list[FoldMetrics] = []
    for fold in folds:
        start = time.time()
        test_mask = frame[fold_column].astype(int).to_numpy() == fold
        train_mask = ~test_mask
        progress_prefix = f"{assay}/{model_name}/{fold_column}/fold={int(fold)}"
        print(
            f"{progress_prefix}: start train_n={int(train_mask.sum())} "
            f"test_n={int(test_mask.sum())} feature_dim={int(x.shape[1])} "
            f"feature_source={actual_feature_source}",
            flush=True,
        )
        model = build_regressor(
            model_name,
            seed + int(fold),
            tabicl_kwargs=tabicl_kwargs,
            tabpfn3_kwargs=tabpfn3_kwargs,
        )
        x_train = np.asarray(x[train_mask], dtype=np.float32)
        x_test = np.asarray(x[test_mask], dtype=np.float32)
        if feature_pca_components is not None:
            print(
                f"{progress_prefix}: fitting PCA n_components={int(feature_pca_components)}",
                flush=True,
            )
            pca = PCA(
                n_components=int(feature_pca_components),
                svd_solver="randomized",
                iterated_power=3,
                random_state=seed + int(fold),
            )
            x_train = pca.fit_transform(x_train).astype(np.float32, copy=False)
            x_test = pca.transform(x_test).astype(np.float32, copy=False)
            print(f"{progress_prefix}: finished PCA", flush=True)
        print(f"{progress_prefix}: fitting model", flush=True)
        model.fit(x_train, y[train_mask])
        print(f"{progress_prefix}: finished fit", flush=True)
        chunk_size = None
        if model_name.lower() == "tabicl" and tabicl_kwargs:
            chunk_size = tabicl_kwargs.get("prediction_chunk_size")
        if model_name.lower() in {"tabpfn3", "tabpfn3_esmc600m"} and tabpfn3_kwargs:
            chunk_size = tabpfn3_kwargs.get("prediction_chunk_size")
        fold_pred = predict_regression(
            model,
            x_test,
            prediction_chunk_size=chunk_size,
            progress_prefix=progress_prefix,
        )
        predictions[test_mask] = fold_pred
        metrics = regression_fold_metrics(y[test_mask], fold_pred)
        standardized_mse, raw_mse, train_mean, train_std = train_fold_standardized_mse(
            y[test_mask],
            fold_pred,
            y[train_mask],
        )
        fold_scores.append(
            FoldMetrics(
                fold=int(fold),
                train_n=int(train_mask.sum()),
                test_n=int(test_mask.sum()),
                mse=standardized_mse,
                r2=metrics["r2"],
                spearman=metrics["spearman"],
                elapsed_seconds=float(time.time() - start),
                raw_mse=raw_mse,
                target_train_mean=train_mean,
                target_train_std=train_std,
            )
        )
        print(
            f"{progress_prefix}: finished fold elapsed_seconds={time.time() - start:.1f}",
            flush=True,
        )

    pred_frame = frame[["mutant", "mutated_sequence", "DMS_score"]].copy()
    pred_frame[fold_column] = frame[fold_column]
    pred_frame[f"{model_name}_score"] = predictions
    pred_frame["assay"] = assay
    pred_frame["model"] = model_name
    pred_frame["feature_source"] = output_feature_source
    pred_frame.to_csv(prediction_path, index=False)

    result = summarize_result(
        assay=assay,
        model=model_name,
        fold_column=fold_column,
        feature_source=output_feature_source,
        n_rows=len(frame),
        feature_dim=output_feature_dim,
        fold_scores=fold_scores,
    )
    summary_path.write_text(json.dumps(result_to_jsonable(result), indent=2) + "\n")
    return result


def summarize_result(
    assay: str,
    model: str,
    fold_column: str,
    feature_source: str,
    n_rows: int,
    feature_dim: int,
    fold_scores: list[FoldMetrics],
) -> ReproductionResult:
    def mean_std(values: list[float]) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    mean_mse, std_mse = mean_std([score.mse for score in fold_scores])
    mean_r2, std_r2 = mean_std([score.r2 for score in fold_scores])
    mean_spearman, std_spearman = mean_std([score.spearman for score in fold_scores])
    return ReproductionResult(
        assay=assay,
        model=model,
        fold_column=fold_column,
        feature_source=feature_source,
        n_rows=n_rows,
        feature_dim=feature_dim,
        fold_scores=fold_scores,
        mean_mse=mean_mse,
        std_mse=std_mse,
        mean_r2=mean_r2,
        std_r2=std_r2,
        mean_spearman=mean_spearman,
        std_spearman=std_spearman,
        mse_scale="train_fold_standardized",
    )


def result_to_jsonable(result: ReproductionResult) -> dict:
    payload = asdict(result)
    payload["fold_scores"] = [asdict(score) for score in result.fold_scores]
    return payload


def result_from_json(path: Path) -> ReproductionResult:
    payload = json.loads(Path(path).read_text())
    payload["fold_scores"] = [FoldMetrics(**score) for score in payload["fold_scores"]]
    payload.setdefault("mse_scale", "train_fold_standardized")
    return ReproductionResult(**payload)


def aggregate_summaries(out_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(out_dir).glob("**/summary.json")):
        result = result_from_json(path)
        base = {
            "assay": result.assay,
            "model": result.model,
            "fold_column": result.fold_column,
            "feature_source": result.feature_source,
            "n_rows": result.n_rows,
            "feature_dim": result.feature_dim,
            "mean_mse": result.mean_mse,
            "std_mse": result.std_mse,
            "mean_r2": result.mean_r2,
            "std_r2": result.std_r2,
            "mean_spearman": result.mean_spearman,
            "std_spearman": result.std_spearman,
            "mse_scale": result.mse_scale,
            "summary_path": str(path),
        }
        rows.append(base)
    return pd.DataFrame(rows)
