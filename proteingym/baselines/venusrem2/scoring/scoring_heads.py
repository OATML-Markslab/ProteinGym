import math

import torch

# Standard amino-acid alphabet used for CCD background z-scoring / consistency.
_AA20 = tuple("ACDEFGHIKLMNPQRSTVWY")


def _aa_vocab_ids(vocab):
    ids = []
    for aa in _AA20:
        idx = vocab.get(aa, -1)
        if idx is not None and idx >= 0:
            ids.append(idx)
    return ids


def build_calibration_terms(
    logits,
    sequence,
    vocab,
    scoring_mode,
    raw_logits=None,
    rsa_weights=None,
    plddt_weights=None,
    disable_adaptive_ccd=False,
):
    """Build CCD calibration terms.

    CCD terms (matches vrh calibrated_margin):
      background_z[v] = zscore_AA(logsumexp_i(source[:, v]) - log L)
      bg_consistency = mean_i pearson(source[i, AA], background[AA])
      bg_scale = 1.0 if disable_adaptive_ccd else max(0, bg_consistency)

    calibrated_margin:
      score = Δ - bg_scale * β * (background_z[mt] - background_z[wt])
    """
    terms = {}
    if scoring_mode not in {
        "calibrated_margin",
        "ccd_exact",
        "temp_scaled_log_odds",
        "entropy_adaptive_margin",
        "venusrem2_confidence_gate",
        "rsa_modulated_ccd",
    }:
        # Still attach RSA/pLDDT stats when requested for decay-only log_odds paths.
        if rsa_weights is None and plddt_weights is None:
            return terms

    source = raw_logits if raw_logits is not None else logits
    if scoring_mode in {
        "calibrated_margin",
        "ccd_exact",
        "temp_scaled_log_odds",
        "entropy_adaptive_margin",
        "venusrem2_confidence_gate",
        "rsa_modulated_ccd",
    }:
        probs = source.exp()
        background = torch.logsumexp(source, dim=0) - math.log(max(source.size(0), 1))
        terms["background"] = background
        terms["entropy"] = -(probs * source).sum(dim=-1)
        terms["wt_confidence"] = torch.zeros(source.size(0), device=source.device, dtype=source.dtype)
        terms["wt_log_probability"] = torch.zeros(
            source.size(0), device=source.device, dtype=source.dtype
        )
        wt_vocab_ids = torch.tensor([vocab.get(aa, -1) for aa in sequence], device=source.device, dtype=torch.long)
        valid_mask = wt_vocab_ids >= 0
        if valid_mask.any():
            valid_positions = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            wt_logits = source[valid_positions, wt_vocab_ids[valid_positions]]
            terms["wt_log_probability"][valid_positions] = wt_logits
            # Position-wise z-score then sigmoid (historical CCD; avoids ProtSSN saturation).
            wt_std = wt_logits.std()
            if float(wt_std) > 1e-8:
                wt_z = (wt_logits - wt_logits.mean()) / wt_std
            else:
                wt_z = torch.zeros_like(wt_logits)
            terms["wt_confidence"][valid_positions] = torch.sigmoid(wt_z)
        terms["calibrated_on_raw"] = raw_logits is not None

        # Z-scored background over the 20 AA columns + optional adaptive bg_scale.
        aa_ids = _aa_vocab_ids(vocab)
        background_z = torch.zeros_like(background)
        bg_consistency = 0.0
        if len(aa_ids) >= 2:
            aa_ids_t = torch.tensor(aa_ids, device=source.device, dtype=torch.long)
            bg_aa = background[aa_ids_t]
            bg_std = bg_aa.std()
            if float(bg_std) > 1e-8:
                bg_aa_z = (bg_aa - bg_aa.mean()) / bg_std
            else:
                bg_aa_z = torch.zeros_like(bg_aa)
            background_z[aa_ids_t] = bg_aa_z

            # bg_consistency: mean pearson(local AA logits, global AA background).
            local_aa = source[:, aa_ids_t].to(dtype=torch.float64)
            bg_f = bg_aa.to(dtype=torch.float64)
            bg_c = bg_f - bg_f.mean()
            bg_den = torch.sqrt((bg_c * bg_c).sum())
            if float(bg_den) > 1e-12:
                loc_c = local_aa - local_aa.mean(dim=1, keepdim=True)
                loc_den = torch.sqrt((loc_c * loc_c).sum(dim=1)).clamp_min(1e-12)
                corr = (loc_c * bg_c.unsqueeze(0)).sum(dim=1) / (loc_den * bg_den)
                corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                bg_consistency = float(corr.mean().item())
        terms["background_z"] = background_z
        terms["bg_consistency"] = bg_consistency
        if disable_adaptive_ccd:
            terms["bg_scale"] = 1.0
        else:
            terms["bg_scale"] = max(0.0, bg_consistency)

    if rsa_weights is not None:
        rsa = rsa_weights.squeeze(-1).clamp(0.0, 1.0)
        terms["rsa"] = rsa
        valid = rsa[rsa > 0]
        terms["rsa_mean"] = valid.mean().item() if len(valid) > 0 else 0.0
        terms["rsa_std"] = valid.std().item() if len(valid) > 1 else 0.0
    if plddt_weights is not None:
        plddt = plddt_weights.squeeze(-1).clamp(0.0, 1.0)
        terms["plddt"] = plddt
        valid_p = plddt[plddt > 0]
        terms["plddt_mean"] = valid_p.mean().item() if len(valid_p) > 0 else 0.0
        disorder = 1.0 - plddt[plddt > 0] if len(plddt[plddt > 0]) > 0 else plddt
        terms["plddt_disorder_std"] = disorder.std().item() if len(disorder) > 1 else 0.0
    return terms


def score_sub_mutation(
    sub_mutant,
    logits,
    sequence,
    vocab,
    scoring_mode,
    background_weight,
    calibration_terms,
    uncertainty_weight=0.0,
    score_temperature=1.0,
    entropy_adaptive_power=1.0,
    gate_center=0.5,
    gate_sharpness=8.0,
    use_rsa_decay=False,
    rsa_decay_mode="raw",
    use_plddt_decay=False,
    plddt_decay_mode="above_mean",
    plddt_mode="off",
    task_type="default",
):
    from .mutant_parse import parse_substitution

    wt, idx, mt = parse_substitution(sub_mutant, sequence, vocab)
    mt_id = vocab[mt]
    wt_id = vocab[wt]
    delta_log_odds = logits[idx, mt_id] - logits[idx, wt_id]

    safe_temperature = max(float(score_temperature), 1e-6)
    plddt_gate = 1.0
    if plddt_mode in ("gate_ccd", "gate_all") and "plddt" in calibration_terms:
        plddt_gate = calibration_terms["plddt"][idx].clamp(0.0, 1.0)

    if scoring_mode == "calibrated_margin":
        bg = calibration_terms.get("background_z", calibration_terms["background"])
        bg_scale = float(calibration_terms.get("bg_scale", 1.0))
        background_penalty = bg_scale * background_weight * (bg[mt_id] - bg[wt_id])
        uncertainty_penalty = uncertainty_weight * calibration_terms.get(
            "entropy", torch.zeros((), device=logits.device, dtype=logits.dtype)
        )
        if isinstance(uncertainty_penalty, torch.Tensor) and uncertainty_penalty.ndim > 0:
            uncertainty_penalty = uncertainty_penalty[idx]
        score = delta_log_odds - plddt_gate * (background_penalty + uncertainty_penalty)
    elif scoring_mode == "ccd_exact":
        entropy_i = calibration_terms["entropy"][idx] if "entropy" in calibration_terms else 0.0
        score = (
            logits[idx, mt_id]
            - background_weight * calibration_terms["background"][mt_id]
            - uncertainty_weight * entropy_i
        )
    elif scoring_mode == "temp_scaled_log_odds":
        score = delta_log_odds / safe_temperature
    elif scoring_mode == "entropy_adaptive_margin":
        wt_confidence = calibration_terms["wt_confidence"][idx]
        adaptive_uncertainty = (
            calibration_terms["entropy"][idx]
            * torch.pow(1.0 - wt_confidence, float(entropy_adaptive_power))
        )
        background_penalty = background_weight * (
            calibration_terms["background"][mt_id] - calibration_terms["background"][wt_id]
        )
        score = (
            delta_log_odds / safe_temperature
            - background_penalty
            - uncertainty_weight * adaptive_uncertainty
        )
    elif scoring_mode == "rsa_modulated_ccd":
        background_penalty_raw = background_weight * (
            calibration_terms["background"][mt_id] - calibration_terms["background"][wt_id]
        )
        uncertainty_penalty_raw = uncertainty_weight * calibration_terms["entropy"][idx]
        if "rsa" in calibration_terms:
            rsa_i = calibration_terms["rsa"][idx].clamp(0.0, 1.0)
            background_penalty = (1.0 - rsa_i) * background_penalty_raw
            uncertainty_penalty = rsa_i * uncertainty_penalty_raw
        else:
            background_penalty = background_penalty_raw
            uncertainty_penalty = uncertainty_penalty_raw
        score = delta_log_odds - background_penalty - uncertainty_penalty
    elif scoring_mode == "venusrem2_confidence_gate":
        wt_confidence = calibration_terms["wt_confidence"][idx]
        gate = torch.sigmoid((wt_confidence - float(gate_center)) * float(gate_sharpness))
        score = gate * (delta_log_odds / safe_temperature)
    else:
        score = delta_log_odds

    if use_rsa_decay and "rsa" in calibration_terms:
        rsa_val = calibration_terms["rsa"][idx].clamp(0.0, 1.0)
        if rsa_decay_mode == "above_mean":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            decay = max(0.0, float(rsa_val) - mean_rsa)
        elif rsa_decay_mode == "std_scaled":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            std_rsa = calibration_terms.get("rsa_std", 0.0)
            decay = std_rsa * max(0.0, float(rsa_val) - mean_rsa)
        elif rsa_decay_mode == "adaptive":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            strength = max(0.0, 1.0 - 2.0 * mean_rsa)
            decay = strength * float(rsa_val)
        else:
            decay = float(rsa_val)
        if plddt_mode in ("gate_rsa", "gate_all") and "plddt" in calibration_terms:
            decay = decay * float(calibration_terms["plddt"][idx].clamp(0.0, 1.0))
        if task_type == "surface":
            score = score * (1.0 + decay)
        else:
            score = score * (1.0 - decay)

    # Preferred pLDDT decay API (VenusMutHub / hparam scripts).
    if use_plddt_decay and "plddt" in calibration_terms:
        p_i = float(calibration_terms["plddt"][idx].clamp(0.0, 1.0))
        disorder_i = 1.0 - p_i
        mean_disorder = 1.0 - calibration_terms.get("plddt_mean", 0.0)
        q_i = max(0.0, disorder_i - mean_disorder)
        if plddt_decay_mode == "std_scaled":
            q_i = q_i * calibration_terms.get("plddt_disorder_std", 0.0)
        score = score * (1.0 - q_i)
    elif plddt_mode in ("above_mean_decay", "std_scaled_decay") and "plddt" in calibration_terms:
        p_i = float(calibration_terms["plddt"][idx].clamp(0.0, 1.0))
        disorder_i = 1.0 - p_i
        mean_disorder = 1.0 - calibration_terms.get("plddt_mean", 0.0)
        q_i = max(0.0, disorder_i - mean_disorder)
        if plddt_mode == "std_scaled_decay":
            q_i = q_i * calibration_terms.get("plddt_disorder_std", 0.0)
        score = score * (1.0 - q_i)

    return score.item()


def score_mutations_batch(
    mutants,
    logits,
    sequence,
    vocab,
    scoring_mode,
    background_weight,
    calibration_terms,
    uncertainty_weight=0.0,
    use_rsa_decay=False,
    rsa_decay_mode="raw",
    use_plddt_decay=False,
    plddt_decay_mode="above_mean",
    task_type="default",
):
    """Batched mutant scoring with VenusREM2 CCD (z-scored background + optional adaptive bg_scale)."""
    n_mutants = len(mutants)
    if n_mutants == 0:
        return []

    mut_indices = []
    positions = []
    wt_ids = []
    mt_ids = []

    from .mutant_parse import parse_substitution

    for i, mutant in enumerate(mutants):
        for sub in str(mutant).split(":"):
            wt, idx, mt = parse_substitution(sub, sequence, vocab)
            mut_indices.append(i)
            positions.append(idx)
            wt_ids.append(vocab[wt])
            mt_ids.append(vocab[mt])

    device = logits.device
    mut_indices = torch.tensor(mut_indices, dtype=torch.long, device=device)
    positions = torch.tensor(positions, dtype=torch.long, device=device)
    wt_ids = torch.tensor(wt_ids, dtype=torch.long, device=device)
    mt_ids = torch.tensor(mt_ids, dtype=torch.long, device=device)

    def _ct(key):
        v = calibration_terms[key]
        return v.to(device) if isinstance(v, torch.Tensor) and v.device != device else v

    delta_log_odds = logits[positions, mt_ids] - logits[positions, wt_ids]

    if scoring_mode == "calibrated_margin":
        bg = _ct("background_z") if "background_z" in calibration_terms else _ct("background")
        bg_scale = float(calibration_terms.get("bg_scale", 1.0))
        bg_penalty = bg_scale * background_weight * (bg[mt_ids] - bg[wt_ids])
        unc = 0.0
        if uncertainty_weight and "entropy" in calibration_terms:
            unc = uncertainty_weight * _ct("entropy")[positions]
        sub_scores = delta_log_odds - bg_penalty - unc
    elif scoring_mode == "ccd_exact":
        bg = _ct("background")
        unc = 0.0
        if uncertainty_weight and "entropy" in calibration_terms:
            unc = uncertainty_weight * _ct("entropy")[positions]
        sub_scores = (
            logits[positions, mt_ids]
            - background_weight * bg[mt_ids]
            - unc
        )
    elif scoring_mode == "rsa_modulated_ccd":
        bg = _ct("background")
        bg_penalty_raw = background_weight * (bg[mt_ids] - bg[wt_ids])
        unc_raw = 0.0
        if uncertainty_weight and "entropy" in calibration_terms:
            unc_raw = uncertainty_weight * _ct("entropy")[positions]
        if "rsa" in calibration_terms:
            rsa_i = _ct("rsa")[positions].clamp(0.0, 1.0)
            bg_penalty = (1.0 - rsa_i) * bg_penalty_raw
            unc = rsa_i * unc_raw if isinstance(unc_raw, torch.Tensor) else unc_raw
        else:
            bg_penalty = bg_penalty_raw
            unc = unc_raw
        sub_scores = delta_log_odds - bg_penalty - unc
    else:
        sub_scores = delta_log_odds

    if use_rsa_decay and "rsa" in calibration_terms:
        rsa_vals = _ct("rsa")[positions].clamp(0.0, 1.0)
        if rsa_decay_mode == "above_mean":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            decay = (rsa_vals - mean_rsa).clamp_min(0.0)
        elif rsa_decay_mode == "std_scaled":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            std_rsa = calibration_terms.get("rsa_std", 0.0)
            decay = std_rsa * (rsa_vals - mean_rsa).clamp_min(0.0)
        elif rsa_decay_mode == "adaptive":
            mean_rsa = calibration_terms.get("rsa_mean", 0.0)
            strength = max(0.0, 1.0 - 2.0 * mean_rsa)
            decay = strength * rsa_vals
        else:
            decay = rsa_vals
        if task_type == "surface":
            sub_scores = sub_scores * (1.0 + decay)
        else:
            sub_scores = sub_scores * (1.0 - decay)

    if use_plddt_decay and "plddt" in calibration_terms:
        p_vals = _ct("plddt")[positions].clamp(0.0, 1.0)
        disorder = 1.0 - p_vals
        mean_disorder = 1.0 - calibration_terms.get("plddt_mean", 0.0)
        q = (disorder - mean_disorder).clamp_min(0.0)
        if plddt_decay_mode == "std_scaled":
            q = q * calibration_terms.get("plddt_disorder_std", 0.0)
        sub_scores = sub_scores * (1.0 - q)

    scores = torch.zeros(n_mutants, dtype=sub_scores.dtype, device=device)
    scores.scatter_add_(0, mut_indices, sub_scores)

    return scores.tolist()
