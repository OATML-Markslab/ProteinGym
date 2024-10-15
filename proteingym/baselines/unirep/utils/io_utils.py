import fileinput
import glob
import os

from Bio import SeqIO
import filelock
import numpy as np
import pandas as pd

from utils.data_utils import is_valid_seq, seqs_to_onehot


def merge_dfs(in_rgx, out_path, index_cols, groupby_cols, ignore_cols):
    """
    Merge multiple pandas DataFrames into one and provides a summary file.
    Args:
    - in_rgx: regex for input filepath
    - out_path: output path
    - index_cols: index column names for DataFrame
    - groupby_cols: groupby column names in the summary step
    - ignore_cols: columns to be ignored in the summary step
    """
    lock = filelock.FileLock(out_path + '.lock')
    with lock:
        frames = []
        for f in glob.glob(in_rgx):
            try:
                frames.append(pd.read_csv(f))
                os.remove(f)
            except pd.errors.EmptyDataError:
                continue
        df = pd.concat(frames, axis=0, sort=True).sort_values(index_cols)
        df.set_index(index_cols).to_csv(out_path, float_format='%.4f')

        #df = df.drop(columns=ignore_cols)
        #means = df.groupby(groupby_cols).mean()
        #stds = df.groupby(groupby_cols).std()
        #summary = pd.merge(means, stds, on=groupby_cols, suffixes=('_mean', '_std'))
        #summary = summary.sort_index(axis=1)
        #save_path = out_path.replace(".csv", "_summary.csv")
        #summary.to_csv(save_path, float_format='%.4f')
        #return summary


def parse_var(s):
    """
    Parse a key, value pair, separated by '='
    That's the reverse of ShellArgs.
    On the command line (argparse) a declaration will typically look like:
        foo=hello
    or
        foo="hello world"
    """
    items = s.split('=')
    key = items[0].strip() # we remove blanks around keys, as is logical
    if len(items) > 1:
        # rejoin the rest:
        value = '='.join(items[1:])
    return (key, value)


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            try:
                d[key] = float(value)
            except:
                d[key] = value
    return d


def load_data_split(dataset_name, split_id, seed=0, ignore_gaps=False):
    data_path = os.path.join('data', dataset_name, 'data.csv')
    # Sample shuffles the DataFrame.
    data_pre_split = pd.read_csv(data_path).sample(frac=1.0, random_state=seed)
    if not ignore_gaps:
        is_valid = data_pre_split['seq'].apply(is_valid_seq)
        data_pre_split = data_pre_split[is_valid]
    if split_id == -1:
        return data_pre_split
    return np.array_split(data_pre_split, 3)[split_id]


def get_wt_log_fitness(dataset_name):
    data_path = os.path.join('data', dataset_name, 'data.csv')
    data = pd.read_csv(data_path)
    try:
        return data[data.n_mut == 0].log_fitness.mean()
    except:
        return data.log_fitness.mean()


def get_log_fitness_cutoff(dataset_name):
    data_path = os.path.join('data', dataset_name, 'log_fitness_cutoff.npy')
    return np.loadtxt(data_path).item()


def count_rows(filename_glob_pattern):
    cnt = 0
    for f in sorted(glob.glob(filename_glob_pattern)):
        with open(f) as fp:
            for line in fp:
                cnt += 1
    return cnt


def load_rows_by_numbers(filename_glob_pattern, line_numbers):
    lns_sorted = sorted(line_numbers)
    lns_idx = np.argsort(line_numbers)
    n_rows = len(line_numbers)
    current_ln = 0   # current (accumulated) line number in opened file
    j = 0            # index in lns
    rows = None
    for f in sorted(glob.glob(filename_glob_pattern)):
        with open(f) as fp:
            for line in fp:
                while j < n_rows and lns_sorted[j] == current_ln:
                    thisrow = np.array([float(x) for x in line.split(' ')])
                    if rows is None:
                        rows = np.full((n_rows, len(thisrow)), np.nan)
                    rows[lns_idx[j], :] = thisrow
                    j += 1
                current_ln += 1
    assert j == n_rows, (f"Expected {n_rows} rows, found {j}. "
    f"Scanned {current_ln} lines from {filename_glob_pattern}.")
    return rows


def load(filename_glob_pattern):
    files = sorted(glob.glob(filename_glob_pattern))
    if len(files) == 0:
        print("No files found for", filename_glob_pattern)
    return np.loadtxt(fileinput.input(files))


def save(filename_pattern, data, entries_per_file=2000):
    n_files = int(data.shape[0] / entries_per_file)
    if data.shape[0] % entries_per_file > 0:
        n_files += 1
    for i in range(n_files):
        filename = filename_pattern + f'-{i:03d}-of-{n_files:03d}'
        l_idx = i * entries_per_file
        r_idx = min(l_idx + entries_per_file, data.shape[0])
        np.savetxt(filename, data[l_idx:r_idx])


def load_and_filter_seqs(data_filename, mode="ProteinGym"):
    """
    seqs_filename: file to write out filtered sequences
    """
    df = pd.read_csv(data_filename, low_memory=False)
    if mode=="ProteinGym":
        if "mutated_sequence" in df.columns.values:
            all_sequences = np.unique(df.mutated_sequence.values)
        # assumes "mutant" column contains full mutated sequence instead
        else:
            all_sequences = np.unique(df.mutant.values)
    elif 'Sequence' in df.columns.values:
        all_sequences = np.unique(df.Sequence.values)
    else:
        all_sequences = np.unique(df.seq.values)
    seqs = []
    stop_codon_cnt = 0
    for seq in all_sequences:
        seq = seq.strip('*')
        if is_valid_seq(seq):
            seqs.append(seq)
        else:
            if '*' in seq:
                stop_codon_cnt += 1
            else:
                print('Invalid seq', seq)
    print('Formatted %d sequences. Discarded %d with stop codon.' % (len(seqs), stop_codon_cnt))
    return seqs


def read_fasta(filename, return_ids=False):
    records = SeqIO.parse(filename, 'fasta')
    seqs = list()
    ids = list()
    for record in records:
        seqs.append(str(record.seq))
        ids.append(str(record.id))
    if return_ids:
        return seqs, ids
    else:
        return seqs