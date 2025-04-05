import torch
from tqdm import tqdm
import numpy as np
import re
import os
from Bio.PDB import PDBParser, Selection
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import attr
import pandas as pd
from scipy.stats import spearmanr

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain


def score_mutations_with_pdb(pdb_path, mutations, chain_id=None, model=None, window_size=1024):
    """
    Score mutations using the masked-marginals approach with ESM3 using sequence from PDB.

    Args:
        pdb_path (str): Path to the PDB file
        mutations (list): List of mutations in format "A25G" (wt, position, mutant)
        chain_id (str, optional): Chain ID to use from PDB. If None, uses first chain.
        model: ESM3 model (optional, will load if not provided)
        window_size (int): Size of window for long sequences

    Returns:
        dict: Dictionary of mutation scores
    """
    # Load model if not provided
    if model is None:
        model = ESM3.from_pretrained("esm3-open").to("cuda")

    # Make sure we're in eval mode
    model.eval()

    # Load PDB file using ProteinChain
    protein_chain = ProteinChain.from_pdb(pdb_path, chain_id=chain_id)

    # Create ESMProtein from ProteinChain - sequence only, no structure
    protein = ESMProtein(sequence=protein_chain.sequence)

    sequence = protein_chain.sequence

    if len(sequence) == 0:
        raise ValueError("Could not extract valid sequence from PDB file")

    print(f"Extracted sequence of length {len(sequence)} from PDB")

    # Create a mapping from PDB residue numbers to sequence indices
    # Using residue_index from ProteinChain
    residue_id_to_idx = {}
    for i, residue_id in enumerate(protein_chain.residue_index):
        # Handle insertion codes if present
        if hasattr(protein_chain, 'insertion_code') and protein_chain.insertion_code is not None:
            ins_code = protein_chain.insertion_code[i]
            if ins_code and ins_code.strip():  # If there's a non-empty insertion code
                residue_id = f"{residue_id}{ins_code}"

        residue_id_to_idx[int(residue_id)] = i

    # Parse mutations and validate them against the sequence
    parsed_mutations = []
    for mutation in mutations:
        # Handle multiple mutations separated by colon
        if ":" in mutation:
            sub_mutations = mutation.split(":")
            multi_wt, multi_mt = "", ""
            multi_pos = []
            multi_seq_pos = []
            valid_multi = True

            for sub_mutation in sub_mutations:
                # Parse single mutation format (e.g., "A25G")
                match = re.match(r"([A-Z])(\d+)([A-Z])", sub_mutation)
                if not match:
                    print(
                        f"Warning: Could not parse mutation {sub_mutation}, skipping")
                    valid_multi = False
                    break

                wt, pos_str, mt = match.groups()
                pos = int(pos_str)

                # Check if position exists in PDB
                if pos not in residue_id_to_idx:
                    print(
                        f"Warning: Position {pos} not found in PDB, skipping")
                    valid_multi = False
                    break

                # Map PDB position to sequence index
                seq_pos = residue_id_to_idx[pos]

                if sequence[seq_pos] != wt:
                    print(
                        f"Warning: Wild-type {wt} at position {pos} doesn't match sequence {sequence[seq_pos]} at index {seq_pos}, skipping")
                    valid_multi = False
                    break

                multi_wt += wt
                multi_mt += mt
                multi_pos.append(pos)
                multi_seq_pos.append(seq_pos)

            if valid_multi:
                # Add combined mutation
                parsed_mutations.append(
                    (multi_wt, multi_pos, multi_mt, multi_seq_pos, mutation))
        else:
            # Parse single mutation format (e.g., "A25G")
            match = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
            if not match:
                print(
                    f"Warning: Could not parse mutation {mutation}, skipping")
                continue

            wt, pos_str, mt = match.groups()
            pos = int(pos_str)

            # Check if position exists in PDB
            if pos not in residue_id_to_idx:
                print(f"Warning: Position {pos} not found in PDB, skipping")
                continue

            # Map PDB position to sequence index
            seq_pos = residue_id_to_idx[pos]

            if sequence[seq_pos] != wt:
                print(
                    f"Warning: Wild-type {wt} at position {pos} doesn't match sequence {sequence[seq_pos]} at index {seq_pos}, skipping")
                continue

            parsed_mutations.append((wt, [pos], mt, [seq_pos], mutation))

    if not parsed_mutations:
        print("No valid mutations to score")
        return {}

    # Map amino acids to tokens
    aa_to_token = {}
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")

    # Build a mapping of amino acids to token IDs
    for aa in amino_acids:
        encoded_protein = ESMProtein(sequence=aa)
        tensor = model.encode(encoded_protein)
        # Get the token ID (skipping special tokens)
        aa_to_token[aa] = tensor.sequence[1].item()

    # Encode the protein to get tensor representation
    protein_tensor = model.encode(protein)

    # Get sequence tokens
    sequence_tokens = protein_tensor.sequence

    # Store mutation scores
    mutation_scores = {}

    # For each position, mask it and get the logits
    # Map from sequence position to position in all_token_probs
    position_map = {}
    all_token_probs = []

    # Get all unique positions that need to be scored
    all_positions = []
    for _, _, _, seq_positions, _ in parsed_mutations:
        if isinstance(seq_positions, list):
            all_positions.extend(seq_positions)
        else:
            all_positions.append(seq_positions)
    positions_to_score = sorted(set(all_positions))

    # Check if sequence is too long for model context
    seq_len = len(sequence)
    needs_windowing = seq_len > window_size - 2  # Account for special tokens

    if needs_windowing:
        print(
            f"Sequence length {seq_len} exceeds model window size {window_size-2}, using windowing approach")

    for i, seq_pos in enumerate(tqdm(positions_to_score, desc="Scoring mutations")):
        # Map the sequence position to the index in all_token_probs
        position_map[seq_pos] = i

        # Calculate token position (add 1 for BOS token)
        token_pos = seq_pos + 1

        # Handle long sequences with windowing
        if needs_windowing:
            # Determine optimal window around mutation position
            window_half = (window_size - 2) // 2  # Excluding special tokens
            start_pos = max(0, seq_pos - window_half)
            end_pos = min(seq_len, start_pos + window_size - 2)

            # Adjust start if end is at sequence boundary
            if end_pos == seq_len:
                start_pos = max(0, seq_len - (window_size - 2))

            # Get the window
            window_seq = sequence[start_pos:end_pos]

            # Create new protein for the window - sequence only
            window_protein = ESMProtein(sequence=window_seq)

            # Encode window
            window_tensor = model.encode(window_protein)

            # Calculate new position in window
            window_token_pos = seq_pos - start_pos + 1  # +1 for BOS

            # Clone tokens and apply mask
            masked_tokens = window_tensor.sequence.clone()
            masked_tokens[window_token_pos] = model.tokenizers.sequence.mask_token_id

            # Create masked protein tensor
            masked_protein_tensor = attr.evolve(
                window_tensor, sequence=masked_tokens)
        else:
            # Clone tokens and apply mask for full sequence
            masked_tokens = sequence_tokens.clone()
            masked_tokens[token_pos] = model.tokenizers.sequence.mask_token_id

            # Create a new protein tensor with masked tokens
            masked_protein_tensor = attr.evolve(
                protein_tensor, sequence=masked_tokens)
            window_token_pos = token_pos  # Use original position

        # Get logits at the masked position
        with torch.no_grad():
            logits_output = model.logits(
                masked_protein_tensor,
                LogitsConfig(sequence=True)
            )

        # Calculate log probabilities
        if needs_windowing:
            token_logits = logits_output.logits.sequence[0, window_token_pos]
        else:
            token_logits = logits_output.logits.sequence[0, token_pos]

        token_probs = torch.log_softmax(token_logits, dim=-1)
        all_token_probs.append(token_probs)

    # Score mutations
    for wt, pos_list, mt, seq_pos_list, mutation_name in parsed_mutations:
        # Calculate score based on ESM2's label_row approach
        score = 0.0

        # Handle multiple mutations
        if isinstance(seq_pos_list, list) and len(seq_pos_list) > 1:
            for i, seq_pos in enumerate(seq_pos_list):
                # Get single wt and mt for this position
                single_wt = wt[i]
                single_mt = mt[i]

                # Get token indices for amino acids
                wt_idx = aa_to_token[single_wt]
                mt_idx = aa_to_token[single_mt]

                # Get the index in all_token_probs
                probs_idx = position_map[seq_pos]

                # Extract probabilities
                token_probs = all_token_probs[probs_idx]

                # Add to cumulative score (mt_prob - wt_prob)
                score += (token_probs[mt_idx] - token_probs[wt_idx]).item()
        else:
            # Handle single mutation
            if isinstance(seq_pos_list, list):
                seq_pos = seq_pos_list[0]
            else:
                seq_pos = seq_pos_list

            # Get token indices for amino acids
            wt_idx = aa_to_token[wt]
            mt_idx = aa_to_token[mt]

            # Get the index in all_token_probs
            probs_idx = position_map[seq_pos]

            # Extract probabilities
            token_probs = all_token_probs[probs_idx]

            # Compute score (mt_prob - wt_prob)
            score = (token_probs[mt_idx] - token_probs[wt_idx]).item()

        # Store score
        mutation_scores[mutation_name] = score

    return mutation_scores


def process_csv_and_score_mutations(csv_path, pdb_path, chain_id=None, output_path=None):
    """
    Process a CSV file with mutations and calculate Spearman correlation with DMS scores.

    Args:
        csv_path (str): Path to CSV file with mutations
        pdb_path (str): Path to PDB file
        chain_id (str, optional): Chain ID to use
        output_path (str, optional): Path to save output CSV

    Returns:
        float: Spearman correlation coefficient
    """
    # Load CSV file
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} mutations from CSV file")

    # Extract mutation strings from CSV
    mutations = df['mutant'].tolist()

    # Initialize model
    print("Loading ESM3 model")
    model = ESM3.from_pretrained("esm3-open").to("cuda")

    # Score mutations - now with only sequence-based scoring
    print("Scoring mutations")
    mutation_scores = score_mutations_with_pdb(
        pdb_path,
        mutations,
        chain_id,
        model,
        window_size=1024
    )

    # Add scores to dataframe
    df['ESM3_score'] = df['mutant'].map(
        lambda x: mutation_scores.get(x, np.nan))

    # Calculate Spearman correlation
    valid_data = df.dropna(subset=['ESM3_score', 'DMS_score'])
    if len(valid_data) > 0:
        correlation, p_value = spearmanr(
            valid_data['DMS_score'], valid_data['ESM3_score'])
        print(
            f"\nSpearman correlation: {correlation:.4f} (p-value: {p_value:.4e})")
        print(
            f"Based on {len(valid_data)} valid mutations (out of {len(df)} total)")
    else:
        correlation = np.nan
        print("No valid mutations for correlation calculation")

    # Save results to CSV
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return correlation


def process_assays_from_file(input_list_csv, base_dms_dir, base_pdb_dir, output_dir):
    """
    Process multiple assays from a CSV file with DMS_id column.

    Args:
        input_list_csv (str): Path to CSV file with list of assays (containing DMS_id column)
        base_dms_dir (str): Base directory containing DMS CSV files
        base_pdb_dir (str): Base directory containing PDB files
        output_dir (str): Directory to save output files

    Returns:
        dict: Dictionary of assay IDs and their correlation values
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV file with assay list
    assay_list_df = pd.read_csv(input_list_csv)

    if 'DMS_id' not in assay_list_df.columns:
        raise ValueError("Input CSV must contain a 'DMS_id' column")

    results = {}

    # Process each assay
    for idx, row in assay_list_df.iterrows():
        assay = row['DMS_id']

        # Skip assays with sequence length > 2000 if the column exists
        if 'seq_len' in row and row['seq_len'] > 2000:
            print(
                f"\n=== Skipping assay {assay} due to sequence length {row['seq_len']} > 2000 ===")
            results[assay] = np.nan
            continue

        print(
            f"\n=== Processing assay {assay} ({idx+1}/{len(assay_list_df)}) ===")

        # Extract protein name for PDB file (first two parts separated by underscore)
        pdb_file = "_".join(assay.split("_")[0:2])

        # Construct paths
        input_csv = os.path.join(base_dms_dir, f"{assay}.csv")
        input_pdb = os.path.join(base_pdb_dir, f"{pdb_file}.pdb")
        output_csv = os.path.join(output_dir, f"{assay}_scored.csv")

        # Check if files exist
        if not os.path.exists(input_csv):
            print(f"Error: Input CSV file {input_csv} not found, skipping")
            continue

        if not os.path.exists(input_pdb):
            print(f"Error: PDB file {input_pdb} not found, skipping")
            continue

        # Process the assay
        try:
            correlation = process_csv_and_score_mutations(
                input_csv,
                input_pdb,
                "A",  # Default chain ID
                output_csv
            )

            results[assay] = correlation
        except Exception as e:
            print(f"Error processing {assay}: {str(e)}")
            results[assay] = np.nan

    # Create summary file with correlations
    summary_df = pd.DataFrame(
        {'assay': list(results.keys()), 'correlation': list(results.values())}
    )
    summary_df.to_csv(os.path.join(
        output_dir, "correlation_summary.csv"), index=False)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process DMS assays with ESM3 scoring")
    parser.add_argument("--reference_csv", required=True,
                        help="CSV file with DMS_id column")
    parser.add_argument("--dms_dir", required=True,
                        help="Directory containing DMS CSV files")
    parser.add_argument("--pdb_dir", required=True,
                        help="Directory containing PDB structures")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save output files")

    args = parser.parse_args()

    # Process all assays from the input list
    results = process_assays_from_file(
        args.reference_csv,
        args.dms_dir,
        args.pdb_dir,
        args.output_dir
    )

    # Print summary of results
    print("\n=== Summary of Results ===")
    for assay, correlation in results.items():
        if not np.isnan(correlation):
            print(f"{assay}: Spearman correlation = {correlation:.4f}")
        else:
            print(f"{assay}: Failed to calculate correlation")
