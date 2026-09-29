"""vrh: entropy-weighted native-preference ρ, then scale-map to α.

ρ = (1 - H̄) ρ_π + H̄ ρ_rot
α = ρ s_P / (ρ s_P + (1-ρ) s_M)
CCD uses β = 1-α (no global calibration constant).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import torch

AA20 = "ACDEFGHIKLMNPQRSTVWY"


def aa20_ids(vocab: dict[str, int]) -> torch.Tensor:
    return torch.tensor([vocab[aa] for aa in AA20 if aa in vocab], dtype=torch.long)


def site_mean_corr(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.to(dtype=torch.float64)
    y = right.to(dtype=torch.float64)
    xc = x - x.mean(dim=1, keepdim=True)
    yc = y - y.mean(dim=1, keepdim=True)
    num = (xc * yc).sum(dim=1)
    den = torch.sqrt((xc * xc).sum(dim=1) * (yc * yc).sum(dim=1)).clamp_min(1e-12)
    corr = torch.nan_to_num(num / den, nan=0.0, posinf=0.0, neginf=0.0)
    return float(corr.mean().item())


def site_mean_entropy(logits_aa: torch.Tensor) -> float:
    logp = torch.log_softmax(logits_aa.to(dtype=torch.float64), dim=1)
    entropy = float((-(logp.exp() * logp).sum(dim=1)).mean().item())
    return float(np.clip(entropy / math.log(len(AA20)), 0.0, 1.0))


def entropy_weighted_rho(rho_pi: float, hbar: float, rho_rot: float) -> float:
    return float(np.clip((1.0 - hbar) * rho_pi + hbar * rho_rot, 0.0, 1.0))


def beta_to_raw_alpha(rho: float, scale_plm: float, scale_msa: float) -> float:
    rho = float(np.clip(rho, 0.0, 1.0))
    numerator = rho * float(scale_plm)
    denominator = numerator + (1.0 - rho) * float(scale_msa)
    if denominator <= 0:
        raise ValueError("non-positive scale-mapping denominator")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def native_preference_features(
    raw_logits: torch.Tensor,
    count_matrix: torch.Tensor,
    aln_start: int,
    aln_end: int,
    sequence: str,
    vocab: dict[str, int],
) -> dict[str, float]:
    """DMS-free protein-wise features. ρ uses only the WT-marginal PLL."""
    ids = aa20_ids(vocab)
    if ids.numel() < 2:
        raise ValueError("vocab is missing standard amino acids")
    span = max(0, int(aln_end) - int(aln_start))
    n = min(span, count_matrix.size(0), max(0, raw_logits.size(0) - int(aln_start)))
    if n <= 0:
        raise ValueError("empty aligned span")
    start = int(aln_start)
    raw_aa = raw_logits[start : start + n, ids].to(dtype=torch.float64)
    count_aa = count_matrix[:n, ids].to(dtype=torch.float64)
    plm_pref = torch.softmax(raw_aa, dim=1)
    col_of = {aa: i for i, aa in enumerate(AA20) if aa in vocab}
    wt_index = torch.tensor(
        [col_of.get(aa, -1) for aa in sequence[start : start + n]],
        dtype=torch.long,
    )
    valid = wt_index >= 0
    if not bool(valid.any()):
        raise ValueError("aligned span contains no standard wild-type residues")
    rows = torch.arange(n, dtype=torch.long)[valid]
    wt = wt_index[valid]
    plm_native = plm_pref[rows, wt].clamp_min(1e-12)
    pll_plm = float(plm_native.log().mean())
    x = raw_aa[valid] - raw_aa[valid, wt].unsqueeze(1)
    y = count_aa[valid] - count_aa[valid, wt].unsqueeze(1)
    non_wt = torch.ones_like(x, dtype=torch.bool)
    non_wt[torch.arange(len(rows)), wt] = False
    scale_plm = float(x[non_wt].std(unbiased=True))
    scale_msa = float(y[non_wt].std(unbiased=True))
    if not np.isfinite(scale_plm) or not np.isfinite(scale_msa) or scale_msa <= 0:
        raise ValueError("invalid native-margin score scale")
    rho_pi = float(np.clip((-pll_plm) / math.log(len(AA20)), 0.0, 1.0))
    hbar = site_mean_entropy(raw_aa[valid])
    corr_rm = site_mean_corr(raw_aa[valid], count_aa[valid])
    rho_rot = 0.5 * (1.0 - corr_rm)
    rho = entropy_weighted_rho(rho_pi, hbar, rho_rot)
    return {
        "n": int(n),
        "n_native_sites": int(valid.sum()),
        "pll_plm": pll_plm,
        "rho_pi": rho_pi,
        "hbar": hbar,
        "corr_rm": corr_rm,
        "rho_rot": rho_rot,
        "rho": rho,
        "scale_plm": scale_plm,
        "scale_msa": scale_msa,
        "alpha_pi": beta_to_raw_alpha(rho_pi, scale_plm, scale_msa),
        "alpha": beta_to_raw_alpha(rho, scale_plm, scale_msa),
    }


def parse_alpha_arg(value: Any) -> Any:
    if value is None:
        return "entropy"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"entropy", "auto", "adaptive"}:
        return "entropy"
    return float(text)


def parse_background_weight_arg(value: Any) -> Any:
    if value is None:
        return "one_minus_alpha"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().lower()
    if text in {"one_minus_alpha", "1-alpha", "auto", "dynamic"}:
        return "one_minus_alpha"
    return float(text)


def resolve_mix_weights(
    raw_logits: torch.Tensor,
    count_matrix: Optional[torch.Tensor],
    aln_start: int,
    aln_end: int,
    sequence: str,
    vocab: dict[str, int],
    alpha_spec: Any,
    background_spec: Any,
) -> tuple[float, float, dict[str, float]]:
    """Return (alpha, background_weight, feature dict)."""
    features: dict[str, float] = {}
    alpha_spec = parse_alpha_arg(alpha_spec)
    if alpha_spec == "entropy":
        if count_matrix is None:
            raise ValueError("entropy α requires an MSA count matrix")
        features = native_preference_features(
            raw_logits, count_matrix, aln_start, aln_end, sequence, vocab
        )
        alpha = float(features["alpha"])
    else:
        alpha = float(alpha_spec)
    background_spec = parse_background_weight_arg(background_spec)
    if background_spec == "one_minus_alpha":
        background_weight = 1.0 - float(alpha)
    else:
        background_weight = float(background_spec)
    features["alpha"] = float(alpha)
    features["background_weight"] = float(background_weight)
    return float(alpha), float(background_weight), features
