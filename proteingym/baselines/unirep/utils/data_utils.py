"""
Utilities for data processing.
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split


"""
File formatting note.
Data should be preprocessed as a sequence of comma-seperated ints with
sequences  /n seperated
"""

# Lookup tables
aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10, 'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24,
    'stop':25,
    '-':26,
}

int_to_aa = {value:key for key, value in aa_to_int.items()}

def get_aa_to_int():
    """
    Get the lookup table (for easy import)
    """
    return aa_to_int

def get_int_to_aa():
    """
    Get the lookup table (for easy import)
    """
    return int_to_aa

# Helper functions

def aa_seq_to_int(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]

def int_seq_to_aa(s):
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return "".join([int_to_aa[i] for i in s])


def nonpad_len(batch):
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)
    return lengths    


def format_seq(seq,stop=False):
    """
    Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
    Here, the default is to strip the stop symbol (stop=False) which would have 
    otherwise been added to the end of the sequence. If you are trying to generate
    a rep, do not include the stop. It is probably best to ignore the stop if you are
    co-tuning the babbler and a top model as well.
    """
    if stop:
        int_seq = aa_seq_to_int(seq.strip())
    else:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    return int_seq


def format_batch_seqs(seqs):
    maxlen = -1
    for s in seqs:
        if len(s) > maxlen:
            maxlen = len(s)
    formatted = []
    for seq in seqs:
        pad_len = maxlen - len(seq)
        padded = np.pad(format_seq(seq), (0, pad_len), 'constant', constant_values=0)
        formatted.append(padded)
    return np.stack(formatted)


def is_valid_seq(seq, max_len=4000):
    """
    True if seq is valid for the babbler, False otherwise.
    """
    l = len(seq)
    valid_aas = "MRHKDESTNQCUGPAVIFYWLO"
    if (l <= max_len) and set(seq) <= set(valid_aas):
        return True
    else:
        return False


def seqs_to_onehot(seqs):
    seqs = format_batch_seqs(seqs)
    X = np.zeros((seqs.shape[0], seqs.shape[1]*24), dtype=int)
    for i in range(seqs.shape[1]):
        for j in range(24):
            X[:, i*24+j] = (seqs[:, i] == j)
    return X


def seqs_to_binary_onehot(seqs, wt):
    seqs = np.array([list(s) for s in seqs])
    X = np.zeros((seqs.shape[0], seqs.shape[1]), dtype=int)
    for i in range(seqs.shape[1]):
        X[:, i] = (seqs[:, i] != wt[i])
    return X


def dict2str(d):
    return ';'.join([f'{k}={v}' for k, v in d.items()])


def seq2mutation(seq, model, return_str=False, ignore_gaps=False,
        sep=":", offset=1):
    mutations = []
    for pf, pm in model.index_map.items():
        if seq[pf-offset] != model.target_seq[pm]:
            if ignore_gaps and (
                    seq[pf-offset] == '-' or seq[pf-offset] not in model.alphabet):
                continue
            mutations.append((pf, model.target_seq[pm], seq[pf-offset]))
    if return_str:
        return sep.join([m[1] + str(m[0]) + m[2] for m in mutations])
    return mutations


def seq2mutation_fromwt(seq, wt, ignore_gaps=False, sep=':', offset=1,
        focus_only=True):
    mutations = []
    for i in range(offset, offset+len(seq)):
        if ignore_gaps and ( seq[i-offset] == '-'):
            continue
        if wt[i-offset].islower() and focus_only:
            continue
        if seq[i-offset].upper() != wt[i-offset].upper():
            mutations.append((i, wt[i-offset].upper(), seq[i-offset].upper()))
    return mutations


def seqs2subs(seqs, wt, ignore_gaps=False):
    pos = []
    subs = []
    for s in seqs:
        p = []
        su = []
        for j in range(len(wt)):
            if s[j] != wt[j]:
                if ignore_gaps and (s[j] == '-' or s[j] == 'X'):
                    continue
                p.append(j)
                su.append(s[j])
        pos.append(np.array(p))
        subs.append(np.array(su))
    return pos, subs


def seq2effect(seqs, model, offset=1, ignore_gaps=False):
    effects = np.zeros(len(seqs))
    for i in range(len(seqs)):
        mutations = seq2mutation(seqs[i], model,
                ignore_gaps=ignore_gaps, offset=offset)
        dE, _, _ = model.delta_hamiltonian(mutations)
        effects[i] = dE
    return effects


def mutant2seq(mut, wt, offset):
    if mut.upper() == 'WT':
        return wt
    chars = list(wt)
    mut = mut.replace(':', ',')
    mut = mut.replace(';', ',')
    for m in mut.split(','):
        idx = int(m[1:-1])-offset
        assert wt[idx] == m[0]
        chars[idx] = m[-1]
    return ''.join(chars)

def get_blosum_scores(seqs, wt, matrix):
    scores = np.zeros(len(seqs))
    wt_score = 0
    for j in range(len(wt)):
        wt_score += matrix[wt[j], wt[j]]
    for i, s in enumerate(seqs):
        for j in range(len(wt)):
            if s[j] not in matrix.alphabet:
                print(f'unexpected AA {s[j]} (seq {i}, pos {j})')
            scores[i] += matrix[wt[j], s[j]]
    return scores - wt_score


def get_wt_seq(mutation_descriptions):
    wt_len = 0
    for m in mutation_descriptions:
        if m == 'WT':
            continue
        if int(m[1:-1]) > wt_len:
            wt_len = int(m[1:-1])
    wt = ['?' for _ in range(wt_len)]
    for m in mutation_descriptions:
        if m == 'WT':
            continue
        idx, wt_char = int(m[1:-1])-1, m[0]   # 1-index to 0-index
        if wt[idx] == '?':
            wt[idx] = wt_char
        else:
            assert wt[idx] == wt_char
    return ''.join(wt), wt_len