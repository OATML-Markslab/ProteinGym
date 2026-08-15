# Copyright 2026 Travis Smith
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pure CPCWE (Constraint-Propagated Confidence-Weighted Ensemble)

DYNAMIC RANGE EXPANSION: Power transformation (x^0.7) to expand ensemble output tails.
Testing whether ensemble compression is limiting performance.

Mechanism:
1. Quantile calibration (2.3× harmful expansion)
2. Z-score normalization
3. Confidence-weighted ensemble with model-specific scaling
4. Residual propagation (3 iterations)
5. Power transformation (x^0.7) to expand dynamic range
6. Structure-based penalties with assay-specific multipliers
7. GEMME conservation modulation (Shannon entropy)
"""

import numpy as np
from collections import Counter
from proteingym_data import get_model_scores, get_residue_structure, get_protein_info

AA_ALPHABET = set('ACDEFGHIKLMNPQRSTVWY')

# ── Tunable parameters ──────────────────────────────────────────────────────
_PARAMS = {
    # Model-specific confidence scales
    "confidence_venus":   1.0,
    "confidence_s3f":     1.0,
    "confidence_esm":      0.5,
    "confidence_gemme":    0.5,
    "confidence_prosst":   1.5,

    # Base ensemble weights — high MSA (msa_depth >= 500)
    "weight_high_venus":   0.314,
    "weight_high_s3f":     0.236,
    "weight_high_esm":     0.236,
    "weight_high_gemme":   0.125,
    "weight_high_prosst":  0.089,

    # Base ensemble weights — low MSA (msa_depth < 500)
    "weight_low_venus":    0.471,
    "weight_low_s3f":      0.236,
    "weight_low_esm":      0.078,
    "weight_low_gemme":    0.125,
    "weight_low_prosst":   0.089,

    # Quantile calibration
    "quantile_expansion_harmful": 2.3,

    # Residual propagation
    "residual_iterations": 3,
    "residual_damping":    0.3,
    "residual_position_window": 5,

    # Power transformation
    "power_transform": 0.7,

    # RSA buried threshold
    "rsa_buried_threshold": 0.2,

    # GEMME conservation range
    "gemme_conservation_range": 1.5,

    # Assay penalty multipliers (5 assay types × 5 property types)
    "penalty_stability_charge":    0.90,
    "penalty_stability_volume":    0.95,
    "penalty_stability_hydro":     0.95,
    "penalty_stability_gp":        0.95,
    "penalty_stability_aromatic":  0.95,

    "penalty_binding_charge":      0.95,
    "penalty_binding_volume":      1.00,
    "penalty_binding_hydro":       1.00,
    "penalty_binding_gp":          1.00,
    "penalty_binding_aromatic":    1.00,

    "penalty_activity_charge":     0.90,
    "penalty_activity_volume":     0.95,
    "penalty_activity_hydro":      0.95,
    "penalty_activity_gp":         0.95,
    "penalty_activity_aromatic":   0.95,

    "penalty_expression_charge":  0.925,
    "penalty_expression_volume":   0.975,
    "penalty_expression_hydro":    0.975,
    "penalty_expression_gp":       0.975,
    "penalty_expression_aromatic": 0.975,

    "penalty_default_charge":      0.90,
    "penalty_default_volume":      0.95,
    "penalty_default_hydro":       0.95,
    "penalty_default_gp":          0.95,
    "penalty_default_aromatic":    0.95,
}


def set_params(params_dict):
    """Update tunable parameters in-place. Raises ValueError on unknown keys."""
    unknown = set(params_dict) - set(_PARAMS)
    if unknown:
        raise ValueError(f"Unknown parameter(s): {', '.join(sorted(unknown))}")
    _PARAMS.update(params_dict)
    _refresh_confidence_scales()


# Backward-compatible module-level constants (reference _PARAMS)
CONFIDENCE_SCALES = {
    'venus':  _PARAMS["confidence_venus"],
    's3f':    _PARAMS["confidence_s3f"],
    'esm':    _PARAMS["confidence_esm"],
    'gemme':  _PARAMS["confidence_gemme"],
    'prosst': _PARAMS["confidence_prosst"],
}


def _refresh_confidence_scales():
    """Sync the module-level CONFIDENCE_SCALES dict with current _PARAMS."""
    CONFIDENCE_SCALES['venus']   = _PARAMS["confidence_venus"]
    CONFIDENCE_SCALES['s3f']     = _PARAMS["confidence_s3f"]
    CONFIDENCE_SCALES['esm']     = _PARAMS["confidence_esm"]
    CONFIDENCE_SCALES['gemme']   = _PARAMS["confidence_gemme"]
    CONFIDENCE_SCALES['prosst']  = _PARAMS["confidence_prosst"]


def score_mutations(sequences, protein_id, wild_type, mutations, msa=None):
    """
    Pure CPCWE with power transformation for dynamic range expansion.
    """
    model_scores = get_model_scores(protein_id, mutations)
    protein_info = get_protein_info(protein_id)

    # Get assay type for penalty modulation
    assay_type = protein_info.get("assay_type", "").lower() if protein_info else ""

    # Determine assay-specific penalty multipliers
    if "stability" in assay_type:
        charge_multiplier   = _PARAMS["penalty_stability_charge"]
        volume_multiplier   = _PARAMS["penalty_stability_volume"]
        hydro_multiplier    = _PARAMS["penalty_stability_hydro"]
        gp_multiplier      = _PARAMS["penalty_stability_gp"]
        aromatic_multiplier = _PARAMS["penalty_stability_aromatic"]
    elif "binding" in assay_type:
        charge_multiplier   = _PARAMS["penalty_binding_charge"]
        volume_multiplier   = _PARAMS["penalty_binding_volume"]
        hydro_multiplier    = _PARAMS["penalty_binding_hydro"]
        gp_multiplier      = _PARAMS["penalty_binding_gp"]
        aromatic_multiplier = _PARAMS["penalty_binding_aromatic"]
    elif "activity" in assay_type:
        charge_multiplier   = _PARAMS["penalty_activity_charge"]
        volume_multiplier   = _PARAMS["penalty_activity_volume"]
        hydro_multiplier    = _PARAMS["penalty_activity_hydro"]
        gp_multiplier      = _PARAMS["penalty_activity_gp"]
        aromatic_multiplier = _PARAMS["penalty_activity_aromatic"]
    elif "expression" in assay_type:
        charge_multiplier   = _PARAMS["penalty_expression_charge"]
        volume_multiplier   = _PARAMS["penalty_expression_volume"]
        hydro_multiplier    = _PARAMS["penalty_expression_hydro"]
        gp_multiplier      = _PARAMS["penalty_expression_gp"]
        aromatic_multiplier = _PARAMS["penalty_expression_aromatic"]
    else:
        charge_multiplier   = _PARAMS["penalty_default_charge"]
        volume_multiplier   = _PARAMS["penalty_default_volume"]
        hydro_multiplier    = _PARAMS["penalty_default_hydro"]
        gp_multiplier      = _PARAMS["penalty_default_gp"]
        aromatic_multiplier = _PARAMS["penalty_default_aromatic"]

    # Extract model scores
    venus_scores = []
    s3f_scores = []
    esm_scores = []
    gemme_scores = []
    prosst_scores = []

    for mut in mutations:
        ms = model_scores.get(mut, {})
        venus = ms.get("venus_rem", 0.0)
        s3f = ms.get("s3f_msa", 0.0)
        esm = ms.get("esm2_15b", 0.0)
        gemme = ms.get("gemme", 0.0)
        prosst = ms.get("prosst_2048", 0.0)

        venus_scores.append(venus if venus is not None else 0.0)
        s3f_scores.append(s3f if s3f is not None else 0.0)
        esm_scores.append(esm if esm is not None else 0.0)
        gemme_scores.append(gemme if gemme is not None else 0.0)
        prosst_scores.append(prosst if prosst is not None else 0.0)

    venus_arr = np.array(venus_scores, dtype=np.float64)
    s3f_arr = np.array(s3f_scores, dtype=np.float64)
    esm_arr = np.array(esm_scores, dtype=np.float64)
    gemme_arr = np.array(gemme_scores, dtype=np.float64)
    prosst_arr = np.array(prosst_scores, dtype=np.float64)

    # Get MSA depth for adaptive ensemble weights
    msa_depth = protein_info.get("msa_size", 0) if protein_info else 0

    # Base ensemble weights (same as CPCWE)
    if msa_depth >= 500:
        base_w_venus  = _PARAMS["weight_high_venus"]
        base_w_s3f    = _PARAMS["weight_high_s3f"]
        base_w_esm    = _PARAMS["weight_high_esm"]
        base_w_gemme  = _PARAMS["weight_high_gemme"]
        base_w_prosst = _PARAMS["weight_high_prosst"]
    else:
        base_w_venus  = _PARAMS["weight_low_venus"]
        base_w_s3f    = _PARAMS["weight_low_s3f"]
        base_w_esm    = _PARAMS["weight_low_esm"]
        base_w_gemme  = _PARAMS["weight_low_gemme"]
        base_w_prosst = _PARAMS["weight_low_prosst"]

    # Apply quantile calibration (same as CPCWE)
    def _quantile_calibrate(scores, expansion_harmful, expansion_benign):
        if len(scores) < 10:
            return scores

        quantiles = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
        pred_quantiles = np.quantile(scores, quantiles)

        median_idx = np.argmin(np.abs(quantiles - 0.5))
        median_pred = pred_quantiles[median_idx]
        target_quantiles = pred_quantiles.copy()

        for i in range(median_idx):
            dist_from_median = median_pred - pred_quantiles[i]
            target_quantiles[i] = median_pred - dist_from_median * expansion_harmful

        for i in range(median_idx + 1, len(quantiles)):
            dist_from_median = pred_quantiles[i] - median_pred
            target_quantiles[i] = median_pred + dist_from_median * expansion_benign

        calibrated = np.zeros_like(scores)

        for i, score in enumerate(scores):
            if score <= pred_quantiles[0]:
                calibrated[i] = target_quantiles[0] + (score - pred_quantiles[0])
            elif score >= pred_quantiles[-1]:
                calibrated[i] = target_quantiles[-1] + (score - pred_quantiles[-1])
            else:
                j = 0
                while j < len(pred_quantiles) - 1 and pred_quantiles[j + 1] <= score:
                    j += 1

                t = (score - pred_quantiles[j]) / (pred_quantiles[j + 1] - pred_quantiles[j] + 1e-9)
                calibrated[i] = target_quantiles[j] + t * (target_quantiles[j + 1] - target_quantiles[j])

        return calibrated

    expansion_harmful = _PARAMS["quantile_expansion_harmful"]
    calib_venus  = _quantile_calibrate(venus_arr, expansion_harmful, 1.0)
    calib_s3f    = _quantile_calibrate(s3f_arr, expansion_harmful, 1.0)
    calib_esm    = _quantile_calibrate(esm_arr, expansion_harmful, 1.0)
    calib_gemme  = _quantile_calibrate(gemme_arr, expansion_harmful, 1.0)
    calib_prosst = _quantile_calibrate(prosst_arr, expansion_harmful, 1.0)

    # Z-score normalize
    def zscore(arr):
        mean = np.mean(arr)
        std = np.std(arr)
        if std < 1e-9:
            return arr
        return (arr - mean) / std

    z_venus = zscore(calib_venus)
    z_s3f = zscore(calib_s3f)
    z_esm = zscore(calib_esm)
    z_gemme = zscore(calib_gemme)
    z_prosst = zscore(calib_prosst)

    # Compute confidence
    def compute_confidence(arr):
        max_abs = np.max(np.abs(arr)) if len(arr) > 0 else 1.0
        if max_abs < 1e-9:
            return np.ones_like(arr) * 0.5
        return np.abs(arr) / max_abs

    conf_venus = compute_confidence(z_venus)
    conf_s3f = compute_confidence(z_s3f)
    conf_esm = compute_confidence(z_esm)
    conf_gemme = compute_confidence(z_gemme)
    conf_prosst = compute_confidence(z_prosst)

    # Parse mutations
    mutation_data = _parse_mutations_with_properties_and_wt(mutations, model_scores)

    # Get structure data
    structure_data = get_residue_structure(protein_id)

    # CPCWE residual propagation
    n_mutations = len(mutations)
    current_venus = z_venus.copy()
    current_s3f = z_s3f.copy()
    current_esm = z_esm.copy()
    current_gemme = z_gemme.copy()
    current_prosst = z_prosst.copy()

    curr_conf_venus = conf_venus.copy()
    curr_conf_s3f = conf_s3f.copy()
    curr_conf_esm = conf_esm.copy()
    curr_conf_gemme = conf_gemme.copy()
    curr_conf_prosst = conf_prosst.copy()

    residual_iterations    = _PARAMS["residual_iterations"]
    residual_damping       = _PARAMS["residual_damping"]
    residual_pos_window    = _PARAMS["residual_position_window"]
    rsa_buried_threshold   = _PARAMS["rsa_buried_threshold"]
    gemme_conservation_range = _PARAMS["gemme_conservation_range"]

    _refresh_confidence_scales()

    for iteration in range(residual_iterations):
        # Confidence-weighted ensemble with model-specific scaling
        blended = np.zeros(n_mutations)

        for i in range(n_mutations):
            w_venus  = base_w_venus  * (1.0 + CONFIDENCE_SCALES['venus']   * curr_conf_venus[i])
            w_s3f    = base_w_s3f    * (1.0 + CONFIDENCE_SCALES['s3f']     * curr_conf_s3f[i])
            w_esm    = base_w_esm    * (1.0 + CONFIDENCE_SCALES['esm']     * curr_conf_esm[i])
            w_gemme  = base_w_gemme  * (1.0 + CONFIDENCE_SCALES['gemme']   * curr_conf_gemme[i])
            w_prosst = base_w_prosst * (1.0 + CONFIDENCE_SCALES['prosst']  * curr_conf_prosst[i])

            total_w = w_venus + w_s3f + w_esm + w_gemme + w_prosst
            if total_w > 1e-9:
                blended[i] = (w_venus * current_venus[i] +
                             w_s3f * current_s3f[i] +
                             w_esm * current_esm[i] +
                             w_gemme * current_gemme[i] +
                             w_prosst * current_prosst[i]) / total_w
            else:
                blended[i] = 0.0

        # Confidence-weighted residuals
        residual_venus  = (current_venus  - blended) / (curr_conf_venus  + 0.1)
        residual_s3f    = (current_s3f    - blended) / (curr_conf_s3f    + 0.1)
        residual_esm    = (current_esm    - blended) / (curr_conf_esm    + 0.1)
        residual_gemme  = (current_gemme  - blended) / (curr_conf_gemme  + 0.1)
        residual_prosst = (current_prosst - blended) / (curr_conf_prosst + 0.1)

        # Organize by position
        pos_to_indices = {}
        for i, mut_data in enumerate(mutation_data):
            if mut_data:
                pos = mut_data[0][0] if mut_data else 0
                if pos not in pos_to_indices:
                    pos_to_indices[pos] = []
                pos_to_indices[pos].append(i)

        # Per-position residuals
        pos_residual_venus = {}
        pos_residual_s3f = {}
        pos_residual_esm = {}
        pos_residual_gemme = {}
        pos_residual_prosst = {}

        pos_conf_venus = {}
        pos_conf_s3f = {}
        pos_conf_esm = {}
        pos_conf_gemme = {}
        pos_conf_prosst = {}

        for pos, indices in pos_to_indices.items():
            if len(indices) > 0:
                weights_venus  = curr_conf_venus[indices]  + 0.1
                weights_s3f    = curr_conf_s3f[indices]    + 0.1
                weights_esm    = curr_conf_esm[indices]    + 0.1
                weights_gemme  = curr_conf_gemme[indices]  + 0.1
                weights_prosst = curr_conf_prosst[indices] + 0.1

                pos_residual_venus[pos]  = np.sum(residual_venus[indices]  * weights_venus)  / np.sum(weights_venus)
                pos_residual_s3f[pos]    = np.sum(residual_s3f[indices]    * weights_s3f)    / np.sum(weights_s3f)
                pos_residual_esm[pos]    = np.sum(residual_esm[indices]    * weights_esm)    / np.sum(weights_esm)
                pos_residual_gemme[pos]  = np.sum(residual_gemme[indices]  * weights_gemme)  / np.sum(weights_gemme)
                pos_residual_prosst[pos] = np.sum(residual_prosst[indices] * weights_prosst) / np.sum(weights_prosst)

                pos_conf_venus[pos]  = np.mean(curr_conf_venus[indices])
                pos_conf_s3f[pos]    = np.mean(curr_conf_s3f[indices])
                pos_conf_esm[pos]    = np.mean(curr_conf_esm[indices])
                pos_conf_gemme[pos]  = np.mean(curr_conf_gemme[indices])
                pos_conf_prosst[pos] = np.mean(curr_conf_prosst[indices])

        # Propagate residuals
        pos_correction_venus = {}
        pos_correction_s3f = {}
        pos_correction_esm = {}
        pos_correction_gemme = {}
        pos_correction_prosst = {}

        for pos in pos_residual_venus.keys():
            similar_positions = []
            for other_pos in pos_residual_venus.keys():
                if abs(other_pos - pos) <= residual_pos_window:
                    if structure_data and pos in structure_data and other_pos in structure_data:
                        rsa_pos = structure_data[pos].get("rsa", 0.5)
                        rsa_other = structure_data[other_pos].get("rsa", 0.5)
                        rsa_diff = abs(rsa_pos - rsa_other)
                        weight = np.exp(-rsa_diff / 0.3)
                    else:
                        weight = 1.0

                    conf_ratio = (pos_conf_venus.get(other_pos, 0.5) /
                                 (pos_conf_venus.get(pos, 0.5) + 0.1))
                    weight *= conf_ratio

                    similar_positions.append((other_pos, weight))

            if similar_positions:
                total_weight = sum(w for _, w in similar_positions)
                pos_correction_venus[pos]  = sum(pos_residual_venus[p]  * w for p, w in similar_positions) / total_weight
                pos_correction_s3f[pos]    = sum(pos_residual_s3f[p]    * w for p, w in similar_positions) / total_weight
                pos_correction_esm[pos]    = sum(pos_residual_esm[p]    * w for p, w in similar_positions) / total_weight
                pos_correction_gemme[pos]  = sum(pos_residual_gemme[p]  * w for p, w in similar_positions) / total_weight
                pos_correction_prosst[pos] = sum(pos_residual_prosst[p] * w for p, w in similar_positions) / total_weight
            else:
                pos_correction_venus[pos]  = pos_residual_venus[pos]
                pos_correction_s3f[pos]    = pos_residual_s3f[pos]
                pos_correction_esm[pos]    = pos_residual_esm[pos]
                pos_correction_gemme[pos]  = pos_residual_gemme[pos]
                pos_correction_prosst[pos] = pos_residual_prosst[pos]

        # Apply corrections
        damping = residual_damping / (iteration + 1)
        for i, mut_data in enumerate(mutation_data):
            if mut_data:
                pos = mut_data[0][0] if mut_data else 0
                if pos in pos_correction_venus:
                    conf_factor = curr_conf_venus[i]
                    correction_venus  = damping * pos_correction_venus[pos]  * (0.5 + conf_factor)
                    correction_s3f    = damping * pos_correction_s3f[pos]    * (0.5 + curr_conf_s3f[i])
                    correction_esm    = damping * pos_correction_esm[pos]    * (0.5 + curr_conf_esm[i])
                    correction_gemme  = damping * pos_correction_gemme[pos]  * (0.5 + curr_conf_gemme[i])
                    correction_prosst = damping * pos_correction_prosst[pos] * (0.5 + curr_conf_prosst[i])

                    current_venus[i]  -= correction_venus
                    current_s3f[i]    -= correction_s3f
                    current_esm[i]    -= correction_esm
                    current_gemme[i]  -= correction_gemme
                    current_prosst[i] -= correction_prosst

    # Position conservation for GEMME weight modulation
    position_conservation = _compute_position_conservation(msa)

    gemme_weights = np.zeros(n_mutations)
    for i, mut_data in enumerate(mutation_data):
        if mut_data and len(mut_data) > 0:
            pos = mut_data[0][0]
            if pos < len(position_conservation):
                norm_entropy = position_conservation[pos]
                gemme_weight = base_w_gemme * (gemme_conservation_range - norm_entropy)
            else:
                gemme_weight = base_w_gemme
            gemme_weights[i] = gemme_weight
        else:
            gemme_weights[i] = base_w_gemme

    # Final ensemble with conservation-modulated weights and structural penalties
    final_scores = np.zeros(n_mutations)
    for i in range(n_mutations):
        w_venus  = base_w_venus  * (1.0 + CONFIDENCE_SCALES['venus']   * curr_conf_venus[i])
        w_s3f    = base_w_s3f    * (1.0 + CONFIDENCE_SCALES['s3f']     * curr_conf_s3f[i])
        w_esm    = base_w_esm    * (1.0 + CONFIDENCE_SCALES['esm']     * curr_conf_esm[i])
        w_gemme  = gemme_weights[i] * (1.0 + CONFIDENCE_SCALES['gemme'] * curr_conf_gemme[i])
        w_prosst = base_w_prosst * (1.0 + CONFIDENCE_SCALES['prosst']  * curr_conf_prosst[i])

        total_w = w_venus + w_s3f + w_esm + w_gemme + w_prosst
        if total_w > 1e-9:
            final_scores[i] = (w_venus * current_venus[i] +
                              w_s3f * current_s3f[i] +
                              w_esm * current_esm[i] +
                              w_gemme * current_gemme[i] +
                              w_prosst * current_prosst[i]) / total_w
        else:
            final_scores[i] = 0.0

        # Apply structural penalties
        if structure_data and i < len(mutation_data):
            mut_data = mutation_data[i]
            if mut_data and len(mut_data) > 0:
                product_penalty = 1.0
                for pos, delta_charge, delta_volume, delta_hydro, wt_aa in mut_data:
                    if pos in structure_data:
                        rsa = structure_data[pos].get("rsa", 1.0)
                    else:
                        rsa = 1.0

                    if rsa < rsa_buried_threshold:
                        penalty = 1.0

                        if delta_charge != 0:
                            penalty *= charge_multiplier

                        if abs(delta_volume) > 50.0:
                            penalty *= volume_multiplier

                        if abs(delta_hydro) > 2.0:
                            penalty *= hydro_multiplier

                        if wt_aa in ('G', 'P'):
                            penalty *= gp_multiplier

                        if wt_aa in ('F', 'W', 'Y'):
                            penalty *= aromatic_multiplier

                        product_penalty *= penalty

                final_scores[i] *= product_penalty

    # DYNAMIC RANGE EXPANSION: Apply power transformation to expand tails
    # Power < 1 expands tails (makes extreme values more extreme)
    # Power > 1 compresses tails (makes extreme values less extreme)
    power = _PARAMS["power_transform"]
    final_scores = np.sign(final_scores) * np.power(np.abs(final_scores), power)

    return final_scores.tolist()


def _parse_mutations_with_properties_and_wt(mutations, model_scores):
    """
    Parse mutation strings to extract positions, physicochemical properties, and wild-type amino acid.
    """
    mutation_data = []

    for mut in mutations:
        components = mut.split(":")

        mut_positions = []
        for comp in components:
            pos = _extract_position(comp)
            wt_aa = comp[0] if len(comp) > 0 else None

            ms = model_scores.get(comp, {})
            delta_charge = ms.get("delta_charge", 0.0) or 0.0
            delta_volume = ms.get("delta_volume", 0.0) or 0.0
            delta_hydro = ms.get("delta_hydro", 0.0) or 0.0

            mut_positions.append((pos, delta_charge, delta_volume, delta_hydro, wt_aa))

        mutation_data.append(mut_positions)

    return mutation_data


def _extract_position(mutant):
    """
    Extract numeric position from mutation string.
    """
    pos_str = ''
    for char in mutant:
        if char.isdigit():
            pos_str += char
        elif pos_str:
            break
    return int(pos_str) if pos_str else 0


def _compute_position_conservation(msa):
    """
    Compute position-specific Shannon entropy from MSA.
    """
    if not msa or len(msa) == 0:
        return []

    msa_len = len(msa[0]) if msa else 0
    if msa_len == 0:
        return []

    max_entropy = np.log2(20)
    normalized_entropy = np.zeros(msa_len)

    for pos in range(msa_len):
        aa_counts = Counter()
        for seq in msa:
            if pos < len(seq):
                aa = seq[pos]
                if aa in AA_ALPHABET:
                    aa_counts[aa] += 1

        total_count = sum(aa_counts.values())
        if total_count == 0:
            normalized_entropy[pos] = 1.0
        else:
            entropy = 0.0
            for count in aa_counts.values():
                p = count / total_count
                if p > 0:
                    entropy -= p * np.log2(p)

            normalized_entropy[pos] = entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy

# CMA-ES optimized parameters
set_params({"confidence_esm": 0.7766273, "confidence_gemme": 0.5535163, "confidence_prosst": 1.9355505, "confidence_s3f": 0.76348445, "confidence_venus": 1.02675538, "gemme_conservation_range": 1.75732338, "power_transform": 0.86625609, "quantile_expansion_harmful": 2.62323591, "residual_damping": 0.36981465, "residual_iterations": 4, "residual_position_window": 5, "rsa_buried_threshold": 0.31058009, "weight_high_esm": 0.25293733, "weight_high_gemme": 0.28817229, "weight_high_prosst": 0.29902311, "weight_high_s3f": 0.4948601, "weight_high_venus": 0.09663942, "weight_low_esm": 0.03612469, "weight_low_gemme": 0.09558981, "weight_low_prosst": 0.28656554, "weight_low_s3f": 0.15944371, "weight_low_venus": 0.10615476})
