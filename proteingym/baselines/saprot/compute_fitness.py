import os
import argparse

from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss
from foldseek_util import get_struc_seq
from tqdm import tqdm

foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"


def predict_mut(model, tokenizer, seq: str, mut_info: str) -> float:
    """
    Predict the mutational effect of a given mutation
    Args:
        seq: The wild type sequence

        mut_info: The mutation information in the format of "A123B", where A is the original amino acid, 123 is the
                  position and B is the mutated amino acid. If multiple mutations are provided, they should be
                  separated by colon, e.g. "A123B:C124D".

    Returns:
        The predicted mutational effect
    """
    tokens = tokenizer.tokenize(seq)
    for single in mut_info.split(":"):
        pos = int(single[1:-1])
        tokens[pos - 1] = "#" + tokens[pos - 1][-1]
    
    mask_seq = " ".join(tokens)
    inputs = tokenizer(mask_seq, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = logits.softmax(dim=-1)
        
        score = 0
        for single in mut_info.split(":"):
            ori_aa, pos, mut_aa = single[0], int(single[1:-1]), single[-1]
            ori_st = tokenizer.get_vocab()[ori_aa + foldseek_struc_vocab[0]]
            mut_st = tokenizer.get_vocab()[mut_aa + foldseek_struc_vocab[0]]
            
            ori_prob = probs[0, pos, ori_st: ori_st + len(foldseek_struc_vocab)].sum()
            mut_prob = probs[0, pos, mut_st: mut_st + len(foldseek_struc_vocab)].sum()
            
            score += torch.log(mut_prob / ori_prob)
    
    return score


def calc_fitness(foldseek_bin, model, DMS_data, tokenizer, mutation_col='mutant', target_seq=None, pdb_file=None, offset_idx=1):
    # Get 3Di sequence
    struc_seq = get_struc_seq(foldseek_bin, pdb_file, ["A"])["A"][1].lower()
    
    seq = "".join([a + b for a, b in zip(target_seq, struc_seq)])
    log_proba_list = []
    for mut_info in tqdm(DMS_data[mutation_col]):
        mutations = []
        for row in mut_info.split(":"):
            ori, pos, mut = row[0], row[1: -1], row[-1]
            pos = int(pos) - offset_idx + 1
            mutations.append(f"{ori}{pos}{mut}")

        all_mut = ":".join(mutations)
        score = predict_mut(model, tokenizer, seq, all_mut).item()
        log_proba_list.append(score)

    return np.array(log_proba_list)


def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"):
    """
    Helper function that mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).
    Mutation triplet are typically based on 1-indexing: start_idx is used for switching to 0-indexing.
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: " + str(mutation))
        relative_position = position - start_idx
        assert (from_AA == focus_seq[relative_position]), "Invalid from_AA or mutant position: " + str(
            mutation) + " from_AA: " + str(from_AA) + " relative pos: " + str(relative_position) + " focus_seq: " + str(
            focus_seq)
        assert (to_AA in AA_vocab), "Mutant to_AA is invalid: " + str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)


def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with SaProt.
    """
    parser = argparse.ArgumentParser(description='SaProt scoring')
    parser.add_argument('--foldseek_bin',
                        default="",
                        type=str, help='Path to foldseek binary file')
    parser.add_argument('--SaProt_model_name_or_path',
                        default="",
                        type=str, help='Name of or path to SaProt model')
    parser.add_argument('--DMS_reference_file_path',
                        default='/sujin/Datasets/ProteinGym/v1.0/DMS_substitutions.csv',
                        type=str, help='Path of DMS folder')
    parser.add_argument('--DMS_data_folder',
                        default='/sujin/Datasets/ProteinGym/v1.0/DMS_ProteinGym_substitutions',
                        type=str, help='Path of DMS folder')
    parser.add_argument('--structure_data_folder', default='', type=str, help='Path of structure folder')
    parser.add_argument('--DMS_index', type=int, help='Path of DMS folder')
    parser.add_argument('--output_scores_folder', default=None, type=str,
                        help='Name of folder to write model scores to')
    parser.add_argument('--indel_mode', action='store_true',
                        help='Whether to score sequences with insertions and deletions')
    args = parser.parse_args()
    model = AutoModelForMaskedLM.from_pretrained(args.SaProt_model_name_or_path, trust_remote_code=True)
    model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.SaProt_model_name_or_path)
    
    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id = list_DMS[args.DMS_index]
    print("Computing scores for: {} with SaProt: {}".format(DMS_id, args.SaProt_model_name_or_path))
    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
    target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].upper()
    
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
    DMS_data['mutated_sequence'] = DMS_data['mutant'].apply(
        lambda x: get_mutated_sequence(target_seq, x)) if not args.indel_mode else DMS_data['mutant']
    
    pdb_filenames = mapping_protein_seq_DMS["pdb_file"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[
        0].split('|')  # if sequence is large (eg., BRCA2_HUMAN) the structure is split in several chunks
    pdb_ranges = mapping_protein_seq_DMS["pdb_range"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].split(
        '|')
    model_scores = []
    for pdb_index, pdb_filename in enumerate(pdb_filenames):
        pdb_file = args.structure_data_folder + os.sep + pdb_filename
        pdb_range = [int(x) for x in pdb_ranges[pdb_index].split("-")]
        target_seq_split = target_seq[pdb_range[0] - 1:pdb_range[1]]  # pdb_range is 1-indexed
        DMS_data["mutated_position"] = DMS_data['mutant'].apply(lambda x: int(x.split(':')[0][1:-1])) #if multiple mutant, will extract position of first mutant
        filtered_DMS_data = DMS_data[
            (DMS_data["mutated_position"] >= pdb_range[0]) & (DMS_data["mutated_position"] <= pdb_range[1])]
        model_scores.append(calc_fitness(foldseek_bin=args.foldseek_bin, model=model, DMS_data=filtered_DMS_data,
                                         tokenizer=tokenizer, target_seq=target_seq_split,
                                         pdb_file=pdb_file, offset_idx=pdb_range[0]))
    
    model_scores = np.concatenate(model_scores)
    
    DMS_data['SaProt_score'] = model_scores
    scoring_filename = args.output_scores_folder + os.sep + DMS_id + '.csv'
    DMS_data[['mutant', 'SaProt_score', 'DMS_score']].to_csv(scoring_filename, index=False)


if __name__ == '__main__':
    main()