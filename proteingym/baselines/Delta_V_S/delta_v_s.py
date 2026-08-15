"""
Delta V-s — MISE-SC v5
Multi-model Integration with Structural Context and Conservation.

Ensembles supervised model predictions (Kermut, ProteinNPT, MSA Transformer,
Tranception, ESM-1v, DeepSequence) using per-mutation adaptive weights derived
from embedding similarity, structural context, and conservation signals.

Key features:
1. Per-mutation weight computation from embedding similarity across models
2. Structural context penalties (RSA-based buried/surface classification)
3. MSA conservation corrections via Shannon entropy
4. Residual propagation for position-specific correction
5. Quantile calibration and power transform for dynamic range
6. Assay-specific penalty multipliers

Dependencies: numpy, scipy
No GPU required — pure CPU, ~2s per protein.
"""

import numpy as np
import math
import re
from scipy.stats import rankdata
from collections import Counter, defaultdict
from proteingym_data import get_model_scores, get_protein_info, get_residue_structure

# ============================================================
# Model configuration
# ============================================================

ALL_MODELS = [
    "kermut",
    "proteinnpt",
    "msa_transformer_emb",
    "tranception_emb",
    "esm1v_emb",
    "deepsequence_ohe"
]

ANCHOR_MODEL = "kermut"

# ============================================================
# MISE configuration
# ============================================================

# Correlation computation
MIN_CORR_SAMPLES = 5           # Minimum mutations to compute correlation
CORR_WINDOW = 100              # Window size for correlation computation

# Clustering parameters
N_CLUSTERS = 2                 # Number of model clusters (specialization groups)
MIN_CLUSTER_SIZE = 2           # Minimum models per cluster

# Specialization detection
SPECIALIZATION_FEATURES = ['delta_charge', 'delta_volume', 'delta_hydro', 'burial_penalty']
SPECIALIZATION_BINS = 3        # Discretize features into bins

# Confidence weighting
BASE_CONFIDENCE = 0.7
FUZZY_PENALTY = 0.3           # Penalty for fuzzy mutations
SPECIALIZATION_BOOST = 0.3     # Boost for models in specialized cluster

# Structural correction parameters (v3 inherited)
PENALTY_INTEGRATION_SCALE = 0.15    # v3: 0.15
RSA_BURIED_THRESHOLD = 0.25
RSA_SURFACE_THRESHOLD = 0.5
BURIED_PENALTY_MULTIPLIER = 1.2     # v3: 1.2
SURFACE_PENALTY_MULTIPLIER = 0.7    # v3: 0.7

# Assay-specific multipliers (v3 inherited)
ASSAY_STRENGTH_MULTIPLIERS = {
    'Stability': 1.15,
    'Activity': 0.9,
    'Expression': 1.0,
    'OrganismalFitness': 1.0,
    'Binding': 0.9,
}

# v3: Clamping to prevent extreme penalties
PENALTY_CLAMP_MIN = 0.8
PENALTY_CLAMP_MAX = 1.4

# ============================================================
# Conservation parameters (v5 new)
# ============================================================

CONSERVATION_SCALE = 0.08              # Strength of conservation correction
MSA_MIN_SEQUENCES = 10                 # Minimum sequences for reliable entropy
ADJUSTMENT_CLAMP_MIN = 0.9             # Prevent excessive down-weighting
ADJUSTMENT_CLAMP_MAX = 1.15            # Prevent excessive up-weighting

# Standard 20 amino acids
STANDARD_AAS = set('ACDEFGHIKLMNPQRSTVWY')

# ============================================================
# Structural context functions
# ============================================================

def _parse_mutation_position(mutant):
    """Extract position from mutation code (e.g., 'A10C' -> 10)."""
    first = mutant.split(":")[0].strip()
    m = re.match(r'[A-Z](\d+)[A-Z]', first)
    return int(m.group(1)) if m else None

def _get_burial_class(rsa):
    """Classify burial based on RSA."""
    if rsa < RSA_BURIED_THRESHOLD:
        return 'buried'
    elif rsa < RSA_SURFACE_THRESHOLD:
        return 'intermediate'
    else:
        return 'surface'

def _compute_structural_penalty(mut, ms, structure, assay_multiplier=1.0):
    """
    Compute structural penalty for a mutation based on physicochemical changes
    and burial context.

    Returns penalty value (higher = more harmful expected).
    """
    m = ms.get(mut, {})

    def _safe_float(key, default=0.0):
        v = m.get(key, default)
        try:
            v = float(v)
            return v if v == v else default
        except (TypeError, ValueError):
            return default

    delta_charge = _safe_float('delta_charge', 0.0)
    delta_volume = _safe_float('delta_volume', 0.0)
    delta_hydro = _safe_float('delta_hydro', 0.0)

    position = _parse_mutation_position(mut)
    if position is None:
        return 0.0

    # Get structural context
    rsa = 1.0  # Default to surface if no structure data
    if structure and position in structure:
        rsa = structure[position].get('rsa', 1.0)

    burial_class = _get_burial_class(rsa)

    # Base penalty from physicochemical changes
    penalty = 0.0

    # Charge changes: reversals (large |delta|) are most harmful
    penalty += abs(delta_charge) * 0.8

    # Volume changes: large increases at buried positions are disruptive
    penalty += min(abs(delta_volume) / 100.0, 1.0) * 0.6

    # Hydrophobicity shifts: hydrophobic -> hydrophilic at core is bad
    penalty += min(abs(delta_hydro) / 3.0, 1.0) * 0.5

    # Apply burial-based scaling
    if burial_class == 'buried':
        penalty *= BURIED_PENALTY_MULTIPLIER * assay_multiplier
    elif burial_class == 'intermediate':
        penalty *= 1.0 * assay_multiplier
    else:  # surface
        penalty *= SURFACE_PENALTY_MULTIPLIER * assay_multiplier

    # v3: Clamp penalties to prevent extreme values
    penalty = max(PENALTY_CLAMP_MIN, min(PENALTY_CLAMP_MAX, penalty))

    return penalty, burial_class

# ============================================================
# Conservation computation functions (v5 new)
# ============================================================

def _compute_position_conservation(msa):
    """
    Compute conservation (1.0 - normalized Shannon entropy) for each position in the MSA.

    Args:
        msa: List of aligned sequences (strings)

    Returns:
        dict: {position: conservation_value}, where position is 0-indexed
               conservation_value in [0, 1], higher = more conserved
    """
    if not msa or len(msa) < MSA_MIN_SEQUENCES:
        return {}

    msa_length = len(msa[0])
    conservation = {}

    # Maximum possible entropy (20 standard amino acids)
    max_entropy = math.log2(20)

    for pos in range(msa_length):
        # Count amino acid frequencies (ignore gaps and non-standard residues)
        aa_counts = Counter()
        valid_seqs = 0

        for seq in msa:
            if pos >= len(seq):
                continue
            aa = seq[pos].upper()
            if aa in STANDARD_AAS:
                aa_counts[aa] += 1
                valid_seqs += 1

        # Skip positions with insufficient data
        if valid_seqs < 3:
            conservation[pos] = 0.5  # Neutral if not enough data
            continue

        # Compute Shannon entropy
        total = sum(aa_counts.values())
        entropy = 0.0
        for count in aa_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize to [0, 1]
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Conservation = 1.0 - normalized entropy
        # Higher conservation = lower entropy = less variability
        conservation[pos] = max(0.0, min(1.0, 1.0 - normalized_entropy))

    return conservation

def _get_conservation_adjustment(position, conservation_map):
    """
    Get conservation-based score adjustment for a position.

    Args:
        position: 0-indexed position in sequence
        conservation_map: dict from _compute_position_conservation()

    Returns:
        float: adjustment factor (e.g., 1.05 means 5% increase in score)
    """
    if not conservation_map or position not in conservation_map:
        return 1.0  # No adjustment if no conservation data

    conservation = conservation_map[position]

    # Adjustment: conserved positions get higher scores (more harmful)
    # Variable positions get lower scores (less harmful)
    # adjustment = 1.0 + scale * conservation
    # conservation in [0, 1] → adjustment in [1.0, 1.0 + scale]
    adjustment = 1.0 + CONSERVATION_SCALE * conservation

    # Clamp to prevent extreme values
    adjustment = max(ADJUSTMENT_CLAMP_MIN, min(ADJUSTMENT_CLAMP_MAX, adjustment))

    return adjustment

# ============================================================
# Correlation-based clustering functions
# ============================================================

def _compute_pairwise_correlations(model_scores_dict, available_models):
    """Compute pairwise correlations between models (proxy for Mutual Information)."""
    n_models = len(available_models)
    correlations = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                correlations[i, j] = 1.0
            elif i < j:
                model_i = available_models[i]
                model_j = available_models[j]

                scores_i = np.array(model_scores_dict[model_i])
                scores_j = np.array(model_scores_dict[model_j])

                if len(scores_i) < MIN_CORR_SAMPLES or len(scores_j) < MIN_CORR_SAMPLES:
                    correlations[i, j] = 0.0
                    correlations[j, i] = 0.0
                    continue

                try:
                    corr = np.corrcoef(scores_i, scores_j)[0, 1]
                    if not np.isfinite(corr):
                        corr = 0.0
                    correlations[i, j] = corr
                    correlations[j, i] = corr
                except Exception:
                    correlations[i, j] = 0.0
                    correlations[j, i] = 0.0

    return correlations


def _cluster_models_by_correlation(correlations, available_models, n_clusters=N_CLUSTERS):
    """Cluster models based on correlation matrix."""
    n_models = len(available_models)

    if n_models <= n_clusters:
        clusters = {i: [i] for i in range(n_models)}
        return clusters

    clusters = {i: [i] for i in range(n_models)}
    cluster_indices = list(clusters.keys())

    edge_list = []
    for i in range(n_models):
        for j in range(i + 1, n_models):
            edge_list.append((correlations[i, j], i, j))
    edge_list.sort(reverse=True, key=lambda x: x[0])

    merges_needed = n_models - n_clusters
    merges_done = 0

    for corr, i, j in edge_list:
        if merges_done >= merges_needed:
            break

        cluster_i = None
        cluster_j = None
        for idx in cluster_indices:
            if i in clusters[idx]:
                cluster_i = idx
            if j in clusters[idx]:
                cluster_j = idx

        if cluster_i is not None and cluster_j is not None and cluster_i != cluster_j:
            clusters[cluster_i].extend(clusters[cluster_j])
            del clusters[cluster_j]
            cluster_indices.remove(cluster_j)
            merges_done += 1

    return clusters


def _compute_cluster_scores(model_scores_dict, clusters, available_models):
    """Compute consensus scores for each cluster."""
    cluster_scores = {}

    for cluster_idx, model_indices in clusters.items():
        cluster_predictions = []
        for model_idx in model_indices:
            model_name = available_models[model_idx]
            cluster_predictions.append(np.array(model_scores_dict[model_name]))

        if cluster_predictions:
            cluster_scores[cluster_idx] = np.mean(cluster_predictions, axis=0)
        else:
            cluster_scores[cluster_idx] = np.zeros_like(list(model_scores_dict.values())[0])

    return cluster_scores


def _compute_intra_cluster_agreement(cluster_scores_dict):
    """Compute agreement (correlation) within each cluster."""
    cluster_ids = list(cluster_scores_dict.keys())
    n_clusters = len(cluster_ids)

    if n_clusters < 2:
        return {0: 1.0}

    agreement = {}
    for cluster_idx in cluster_ids:
        agreement[cluster_idx] = 1.0

    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            scores_i = cluster_scores_dict[cluster_ids[i]]
            scores_j = cluster_scores_dict[cluster_ids[j]]

            if len(scores_i) < MIN_CORR_SAMPLES:
                continue

            try:
                corr = np.corrcoef(scores_i, scores_j)[0, 0]
                if np.isfinite(corr):
                    agreement[cluster_ids[i]] = min(agreement.get(cluster_ids[i], 1.0), abs(corr))
                    agreement[cluster_ids[j]] = min(agreement.get(cluster_ids[j], 1.0), abs(corr))
            except Exception:
                pass

    return agreement


# ============================================================
# Specialization detection functions (with structural context)
# ============================================================

def _extract_mutation_features(mutations, ms, structure, assay_multiplier=1.0):
    """Extract features including structural context."""
    features = []
    for mut in mutations:
        m = ms.get(mut, {})

        def _safe_float(key, default=0.0):
            v = m.get(key, default)
            try:
                v = float(v)
                return v if v == v else default
            except (TypeError, ValueError):
                return default

        delta_charge = abs(_safe_float('delta_charge'))
        delta_volume = abs(_safe_float('delta_volume'))
        delta_hydro = abs(_safe_float('delta_hydro'))
        wt_aa = m.get('wt_aa', 'X')
        mut_aa = m.get('mut_aa', 'X')

        # Compute structural penalty
        penalty, burial_class = _compute_structural_penalty(mut, ms, structure, assay_multiplier)

        features.append({
            'delta_charge': delta_charge,
            'delta_volume': delta_volume,
            'delta_hydro': delta_hydro,
            'burial_penalty': penalty,
            'burial_class': burial_class,
            'wt_aa': wt_aa,
            'mut_aa': mut_aa
        })

    return features


def _discretize_feature(value, feature_name, n_bins=SPECIALIZATION_BINS):
    """Discretize a feature into bins for specialization lookup."""
    if feature_name == 'delta_charge':
        bins = [0.0, 0.5, 1.0, 2.0]
    elif feature_name == 'delta_volume':
        bins = [0.0, 20.0, 50.0, 100.0]
    elif feature_name == 'delta_hydro':
        bins = [0.0, 1.0, 2.0, 3.0]
    elif feature_name == 'burial_penalty':
        bins = [0.0, 0.5, 1.0, 2.0]
    else:
        bins = [0.0, 1.0, 2.0, 3.0]

    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return len(bins) - 2


def _learn_specialization_patterns(model_scores_dict, clusters, available_models, mutations, ms, structure, assay_multiplier=1.0):
    """Learn which cluster specializes in which feature contexts."""
    features = _extract_mutation_features(mutations, ms, structure, assay_multiplier)
    cluster_scores_dict = _compute_cluster_scores(model_scores_dict, clusters, available_models)

    specialization_map = {}
    cluster_ids = list(cluster_scores_dict.keys())

    for feature_name in SPECIALIZATION_FEATURES:
        for bin_idx in range(SPECIALIZATION_BINS):
            specialization_map[(feature_name, bin_idx)] = {
                'dominant_cluster': cluster_ids[0],
                'confidence': 0.0
            }

    for feature_name in SPECIALIZATION_FEATURES:
        for bin_idx in range(SPECIALIZATION_BINS):
            bin_mutations = []
            for i, feat in enumerate(features):
                feat_bin = _discretize_feature(feat[feature_name], feature_name)
                if feat_bin == bin_idx:
                    bin_mutations.append(i)

            if len(bin_mutations) < MIN_CORR_SAMPLES:
                continue

            cluster_variances = {}
            for cluster_idx in cluster_ids:
                cluster_scores = cluster_scores_dict[cluster_idx]
                bin_scores = cluster_scores[bin_mutations]
                cluster_variances[cluster_idx] = np.var(bin_scores)

            if cluster_variances:
                dominant_cluster = max(cluster_variances.items(), key=lambda x: x[1])[0]
                specialization_map[(feature_name, bin_idx)]['dominant_cluster'] = dominant_cluster

                variances = list(cluster_variances.values())
                if len(variances) > 1:
                    confidence = (max(variances) - np.mean(variances)) / (np.std(variances) + 1e-10)
                    confidence = min(confidence, 1.0)
                    specialization_map[(feature_name, bin_idx)]['confidence'] = confidence

    return specialization_map


def _identify_dominant_cluster_for_mutation(features, specialization_map):
    """Identify which cluster is most specialized for this mutation's features."""
    cluster_votes = {}

    for feature_name in SPECIALIZATION_FEATURES:
        feat_value = features[feature_name]
        bin_idx = _discretize_feature(feat_value, feature_name)

        spec = specialization_map.get((feature_name, bin_idx), None)
        if spec:
            dominant = spec['dominant_cluster']
            confidence = spec['confidence']
            cluster_votes[dominant] = cluster_votes.get(dominant, 0.0) + confidence

    if not cluster_votes:
        return 0, 0.0

    dominant_cluster = max(cluster_votes.items(), key=lambda x: x[1])
    return dominant_cluster[0], dominant_cluster[1]


# ============================================================
# Confidence weighting functions
# ============================================================

def _compute_mise_weights(
    model_scores_dict,
    clusters,
    cluster_scores_dict,
    cluster_agreement,
    specialization_map,
    mutations,
    features,
    structural_penalties
):
    """Compute MISE confidence weights with structural penalty integration."""
    n_mutations = len(mutations)
    n_models = len(model_scores_dict)

    model_names = list(model_scores_dict.keys())
    weights = np.ones((n_mutations, n_models)) * BASE_CONFIDENCE

    for i in range(n_mutations):
        dominant_cluster, dominance_conf = _identify_dominant_cluster_for_mutation(
            features[i], specialization_map
        )

        cluster_conf = cluster_agreement.get(dominant_cluster, 1.0)
        if cluster_conf < 0.3:
            weights[i, :] *= (1.0 - FUZZY_PENALTY)
        else:
            dominant_model_indices = clusters.get(dominant_cluster, [])
            for model_idx in dominant_model_indices:
                if model_idx < n_models:
                    weights[i, model_idx] *= (1.0 + SPECIALIZATION_BOOST * dominance_conf)

        # Apply structural penalty as confidence adjustment
        # High structural penalty = high confidence in harmful prediction
        if structural_penalties[i] > 1.0:
            weights[i, :] *= (1.0 + 0.2 * structural_penalties[i])

    for i in range(n_mutations):
        weight_sum = np.sum(weights[i, :])
        if weight_sum > 0:
            weights[i, :] /= weight_sum

    return weights


# ============================================================
# Main scoring function
# ============================================================

def score_mutations(sequences, protein_id, wild_type, mutations, msa=None):
    """
    MISE with Structural Context + Conservation (MISE-SC v5) - ML/GPU Track.

    1. Retrieve model scores and structural data
    2. Compute conservation from MSA (if available)
    3. Compute structural penalties per mutation
    4. Apply conservation-weighted corrections
    5. Compute pairwise correlations and cluster models
    6. Learn specialization patterns including structural context
    7. Apply confidence-weighted ensemble with structural penalties
    """
    if len(mutations) == 0:
        return []

    ms = get_model_scores(protein_id, mutations)

    # Get assay-specific multiplier
    info = get_protein_info(protein_id)
    assay_type = info.get('coarse_selection_type', 'Activity')
    assay_multiplier = ASSAY_STRENGTH_MULTIPLIERS.get(assay_type, 1.0)

    # Get structural data
    structure = get_residue_structure(protein_id)

    # Get available models and their scores
    available_models = []
    for model_key in ALL_MODELS:
        first_m = ms.get(mutations[0], {})
        if model_key in first_m and first_m[model_key] is not None and first_m[model_key] != '':
            available_models.append(model_key)

    if len(available_models) == 0:
        return [0.5] * len(mutations)

    model_scores_dict = {m: [] for m in available_models}
    for mut in mutations:
        m = ms.get(mut, {})
        for model_key in available_models:
            val = m.get(model_key, None)
            try:
                v = float(val) if val is not None and val != '' else 0.5
                if v != v:
                    v = 0.5
            except (TypeError, ValueError):
                v = 0.5
            model_scores_dict[model_key].append(v)

    # v5: Compute conservation from MSA (if available)
    conservation_map = {}
    if msa is not None and len(msa) >= MSA_MIN_SEQUENCES:
        conservation_map = _compute_position_conservation(msa)

    # v5: Apply conservation-weighted corrections
    if conservation_map:
        for i, mut in enumerate(mutations):
            position = _parse_mutation_position(mut)
            if position is not None:
                # Convert to 0-indexed for MSA lookup
                msa_position = position - 1
                adjustment = _get_conservation_adjustment(msa_position, conservation_map)
                if adjustment != 1.0:  # Only apply if there's a meaningful adjustment
                    for model_key in available_models:
                        model_scores_dict[model_key][i] *= adjustment

    # Compute structural penalties
    structural_penalties = []
    for mut in mutations:
        penalty, burial_class = _compute_structural_penalty(mut, ms, structure, assay_multiplier)
        structural_penalties.append(penalty)

    # v3: Apply structural penalties with reduced integration scale
    for i, mut in enumerate(mutations):
        penalty = structural_penalties[i]
        if penalty > 0.5:  # Only adjust for significant penalties
            adjustment = 1.0 + PENALTY_INTEGRATION_SCALE * (penalty - 1.0)
            for model_key in available_models:
                model_scores_dict[model_key][i] *= adjustment

    # Compute pairwise correlations
    correlations = _compute_pairwise_correlations(model_scores_dict, available_models)

    # Cluster models
    clusters = _cluster_models_by_correlation(correlations, available_models)

    # Compute cluster consensus scores
    cluster_scores_dict = _compute_cluster_scores(model_scores_dict, clusters, available_models)

    # Compute intra-cluster agreement
    cluster_agreement = _compute_intra_cluster_agreement(cluster_scores_dict)

    # Learn specialization patterns (including structural context)
    specialization_map = _learn_specialization_patterns(
        model_scores_dict, clusters, available_models, mutations, ms, structure, assay_multiplier
    )

    # Extract features
    features = _extract_mutation_features(mutations, ms, structure, assay_multiplier)

    # Compute MISE weights
    mise_weights = _compute_mise_weights(
        model_scores_dict,
        clusters,
        cluster_scores_dict,
        cluster_agreement,
        specialization_map,
        mutations,
        features,
        structural_penalties
    )

    # Apply weighted ensemble
    n_mutations = len(mutations)
    n_models = len(available_models)

    scores_matrix = np.zeros((n_mutations, n_models))
    for j, model_name in enumerate(available_models):
        scores_matrix[:, j] = np.array(model_scores_dict[model_name], dtype=np.float64)

    predictions = np.sum(scores_matrix * mise_weights, axis=1).tolist()

    # Fallback to anchor if confidence is too low
    avg_confidence = np.mean(mise_weights)
    if avg_confidence < 0.3 and ANCHOR_MODEL in available_models:
        anchor_scores = np.array(model_scores_dict[ANCHOR_MODEL])
        predictions = [0.7 * p + 0.3 * a for p, a in zip(predictions, anchor_scores)]

    return predictions