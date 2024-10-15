import numpy as np
import pandas as pd
import torch

AA_vocab = "ACDEFGHIKLMNPQRSTVWY"
unusual_AA ="OU" #Pyrrolysine O and selenocysteine U
indeterminate_AA = "BJXZ" #B = Asparagine or Aspartic acid; J = leucine or isoleucine; X = Any/Unknown ; Z = Glutamine or glutamic acid

def standardize(x, epsilon = 1e-10):
    return (x - x.mean()) / (x.std() + epsilon)
    
def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)

def nansum(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs)

def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab=AA_vocab):
    """
    Helper function that mutates an input sequence (focus_seq) via an input mutation triplet (substitutions only).
    Mutation triplet are typically based on 1-indexing: start_idx is used for switching to 0-indexing.
    """
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: "+str(mutation))
        relative_position = position - start_idx
        assert (from_AA==focus_seq[relative_position]), "Invalid from_AA or mutant position: "+str(mutation)+" from_AA: "+str(from_AA) + " relative pos: "+str(relative_position) + " focus_seq: "+str(focus_seq)
        assert (to_AA in AA_vocab) , "Mutant to_AA is invalid: "+str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)

def get_optimal_window(mutation_position_relative, seq_len_wo_special, model_window):
    half_model_window = model_window // 2
    if seq_len_wo_special <= model_window:
        return [0,seq_len_wo_special]
    elif mutation_position_relative < half_model_window:
        return [0,model_window]
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        return [seq_len_wo_special - model_window, seq_len_wo_special]
    else:
        return [max(0,mutation_position_relative-half_model_window), min(seq_len_wo_special,mutation_position_relative+half_model_window)]

def set_mutant_offset(mutant, MSA_start, mutant_delim=":"):
    """
    Adjusts the offset of a mutant sequence to match the MSA start and end positions
    """
    indiv_mutants = mutant.split(mutant_delim)
    new_mutants = []
    for indiv_mutant in indiv_mutants:
        wt, pos, sub = indiv_mutant[0], int(indiv_mutant[1:-1]), indiv_mutant[-1]
        shift_pos = pos - MSA_start + 1
        new_mutants.append(wt + str(int(shift_pos)) + sub)
    return mutant_delim.join(new_mutants)

def undo_mutant_offset(mutant, MSA_start, mutant_delim=","):
    """
    Undoes the offset adjustment of a mutant sequence to match the MSA start and end positions
    """
    indiv_mutants = mutant.split(mutant_delim)
    new_mutants = []
    for indiv_mutant in indiv_mutants:
        wt, pos, sub = indiv_mutant[0], int(indiv_mutant[1:-1]), indiv_mutant[-1]
        shift_pos = pos + MSA_start - 1
        new_mutants.append(wt + str(int(shift_pos)) + sub)
    return mutant_delim.join(new_mutants)