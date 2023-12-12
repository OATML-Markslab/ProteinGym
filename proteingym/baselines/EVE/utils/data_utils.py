import multiprocessing
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from utils.weights import map_from_alphabet, map_matrix, compute_sequence_weights, calc_weights_evcouplings

# TODOs: Can get rid of one_hot_encodings when only calculating weights right?

# constants
GAP = "-"
MATCH_GAP = GAP
INSERT_GAP = "."

ALPHABET_PROTEIN_NOGAP = "ACDEFGHIKLMNPQRSTVWY"
ALPHABET_PROTEIN_GAP = GAP + ALPHABET_PROTEIN_NOGAP


class MSA_processing:
    def __init__(self,
                 MSA_location="",
                 theta=0.2,
                 use_weights=True,
                 weights_location="./data/weights",
                 preprocess_MSA=True,
                 threshold_sequence_frac_gaps=0.5,
                 threshold_focus_cols_frac_gaps=0.3,
                 remove_sequences_with_indeterminate_AA_in_focus_cols=True,
                 num_cpus=1,
                 weights_calc_method="evcouplings",
                 overwrite_weights=False,
                 debug_only_weights=False,
                 ):

        """
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format:
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corresponding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that;
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) File to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        - num_cpus: (int) Number of CPUs to use for parallel weights calculation processing. If set to -1, all available CPUs are used. If set to 1, weights are computed in serial.
        - weights_calc_method: (str) Method to use for calculating sequence weights. Options: "evcouplings","eve" or "identity". (default "evcouplings")
        -   Note: For now the "evcouplings" method is modified to be equivalent to the "eve" method,
                but the "evcouplings" method is faster as it uses numba.
        - overwrite_weights: (bool) If True, calculate weights and overwrite weights file. If False, load weights from weights_location if it exists.
            TODO these weights options should be more like calc_weights=[True/False], and the weights_location should be a list of locations to load from/save to.
        """
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = ALPHABET_PROTEIN_NOGAP
        self.use_weights = use_weights
        self.overwrite_weights = overwrite_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols
        self.debug_only_weights = debug_only_weights
        self.weights_calc_method = weights_calc_method

        # Defined by gen_alignment
        self.aa_dict = {}
        self.focus_seq_name = ""
        self.seq_name_to_sequence = defaultdict(str)
        self.focus_seq, self.focus_cols, self.focus_seq_trimmed, self.seq_len, self.alphabet_size = [None] * 5
        self.focus_start_loc, self.focus_stop_loc = None, None
        self.uniprot_focus_col_to_wt_aa_dict, self.uniprot_focus_col_to_focus_idx = None, None
        self.one_hot_encoding, self.weights, self.Neff, self.num_sequences = [None] * 4

        # Defined by create_all_singles
        self.mutant_to_letter_pos_idx_focus_list = None
        self.all_single_mutations = None

        # Fill in the instance variables
        self.gen_alignment()
        self.calc_weights(num_cpus=num_cpus, method=weights_calc_method)
        
        if not self.debug_only_weights:
            print("Creating all single mutations")
            self.create_all_singles()

    def gen_alignment(self):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i, aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i == 0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line
        print("Number of sequences in MSA (before preprocessing):", len(self.seq_name_to_sequence))

        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            # Overwrite self.seq_name_to_sequence
            self.seq_name_to_sequence = self.preprocess_msa(
                seq_name_to_sequence=self.seq_name_to_sequence,
                focus_seq_name=self.focus_seq_name,
                threshold_sequence_frac_gaps=self.threshold_sequence_frac_gaps,
                threshold_focus_cols_frac_gaps=self.threshold_focus_cols_frac_gaps
            )

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s != '-']
        self.focus_seq_trimmed = "".join([self.focus_seq[ix] for ix in self.focus_cols])
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        focus_loc = self.focus_seq_name.split("/")[-1]
        start, stop = focus_loc.split("-")
        self.focus_start_loc = int(start)
        self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col + int(start): self.focus_seq[idx_col] for idx_col in self.focus_cols}
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col + int(start): idx_col for idx_col in self.focus_cols}

        # Move all letters to CAPS; keeps focus columns only
        for seq_name, sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".", "-")
            self.seq_name_to_sequence[seq_name] = "".join(
                [sequence[ix].upper() for ix in self.focus_cols])  # Makes a List[str] instead of str

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name, sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]
        
        print("Number of sequences after preprocessing:", len(self.seq_name_to_sequence))
        
        if self.debug_only_weights and self.weights_calc_method == "evcouplings":
            print("Weights-only mode with evcouplings: Skipping one-hot encodings")
        else:
            # Encode the sequences
            print("One-hot encoding sequences")
            self.one_hot_encoding = one_hot_3D(
                seq_keys=self.seq_name_to_sequence.keys(),  # Note: Dicts are unordered for python < 3.6
                seq_name_to_sequence=self.seq_name_to_sequence,
                alphabet=self.alphabet,
                seq_length=self.seq_len,
            )

    # Using staticmethod to keep this under the MSAProcessing namespace, but this is apparently not best practice
    @staticmethod
    def preprocess_msa(seq_name_to_sequence, focus_seq_name, threshold_sequence_frac_gaps, threshold_focus_cols_frac_gaps):
        """Remove inadequate columns and sequences from MSA, overwrite self.seq_name_to_sequence."""
        print("Pre-processing MSA to remove inadequate columns and sequences...")
        msa_df = pd.DataFrame.from_dict(seq_name_to_sequence, orient='index', columns=['sequence'])
        # Data clean up
        msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".", "-")).apply(
            lambda x: ''.join([aa.upper() for aa in x]))
        # Remove columns that would be gaps in the wild type
        non_gap_wt_cols = [aa != '-' for aa in msa_df.sequence[focus_seq_name]]
        msa_df['sequence'] = msa_df['sequence'].apply(
            lambda x: ''.join([aa for aa, non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
        assert 0.0 <= threshold_sequence_frac_gaps <= 1.0, "Invalid fragment filtering parameter"
        assert 0.0 <= threshold_focus_cols_frac_gaps <= 1.0, "Invalid focus position filtering parameter"
        print("Calculating proportion of gaps")
        msa_array = np.array([list(seq) for seq in msa_df.sequence])
        gaps_array = np.array(list(map(lambda seq: [aa == '-' for aa in seq], msa_array)))
        # Identify fragments with too many gaps
        seq_gaps_frac = gaps_array.mean(axis=1)
        seq_below_threshold = seq_gaps_frac <= threshold_sequence_frac_gaps
        print("Proportion of sequences dropped due to fraction of gaps: " + str(
            round(float(1 - seq_below_threshold.sum() / seq_below_threshold.shape) * 100, 2)) + "%")
        # Identify focus columns
        columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
        index_cols_below_threshold = columns_gaps_frac <= threshold_focus_cols_frac_gaps
        print("Proportion of non-focus columns removed: " + str(
            round(float(1 - index_cols_below_threshold.sum() / index_cols_below_threshold.shape) * 100, 2)) + "%")
        # Lower case non focus cols and filter fragment sequences
        def _lower_case_and_filter_fragments(seq):
            return ''.join([aa.lower() if aa_ix in index_cols_below_threshold else aa for aa_ix, aa in enumerate(seq)])
        msa_df['sequence'] = msa_df['sequence'].apply(
            lambda seq: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in
             zip(seq, index_cols_below_threshold)]))
        msa_df = msa_df[seq_below_threshold]
        # Overwrite seq_name_to_sequence with clean version
        seq_name_to_sequence = defaultdict(str)
        for seq_idx in range(len(msa_df['sequence'])):
            seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]

        return seq_name_to_sequence

    def calc_weights(self, num_cpus=1, method="evcouplings"):
        """
        If num_cpus == 1, weights are computed in serial.
        If num_cpus == -1, weights are computed in parallel using all available cores.
        Note: This will use multiprocessing.cpu_count() to get the number of available cores, which on clusters may
        return all cores, not just the number of cores available to the user.
        """
        # Refactored into its own function so that we can call it separately
        if self.use_weights:
            if os.path.isfile(self.weights_location) and not self.overwrite_weights:
                print("Loading sequence weights from disk")
                self.weights = np.load(file=self.weights_location)
            else:
                print("Computing sequence weights")
                if num_cpus == -1:
                    #multiprocessing.cpu_count()
                    num_cpus = get_num_cpus()

                if method == "evcouplings":
                    alphabet_mapper = map_from_alphabet(ALPHABET_PROTEIN_GAP, default=GAP)
                    arrays = []
                    for seq in self.seq_name_to_sequence.values():
                        arrays.append(np.array(list(seq)))
                    sequences = np.vstack(arrays)
                    sequences_mapped = map_matrix(sequences, alphabet_mapper)
                    print("Starting EVCouplings calculation")
                    start = time.perf_counter()
                    self.weights = calc_weights_evcouplings(sequences_mapped, identity_threshold=1 - self.theta,
                                                            empty_value=0, num_cpus=num_cpus)  # GAP = 0
                    end = time.perf_counter()
                    print(f"EVCouplings weights took {end - start:.2f} seconds")
                elif method == "eve":
                    list_seq = self.one_hot_encoding.numpy()
                    start = time.perf_counter()
                    self.weights = compute_sequence_weights(list_seq, self.theta, num_cpus=num_cpus)
                    end = time.perf_counter()
                    print(f"EVE weights took {end - start:.2f} seconds")
                elif method == "identity":
                    self.weights = np.ones(self.one_hot_encoding.shape[0])
                else:
                    raise ValueError(f"Unknown method: {method}. Must be either 'evcouplings', 'eve' or 'identity'.")
                print("Saving sequence weights to disk")
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.weights.shape[0]

        print("Neff =", str(self.Neff))
        
        if self.debug_only_weights and self.weights_calc_method == "evcouplings":
            print("Num sequences: ", self.num_sequences)
        else:
            print("Data Shape =", self.one_hot_encoding.shape)

        return self.weights

    def create_all_singles(self):
        start_idx = self.focus_start_loc
        focus_seq_index = 0
        self.mutant_to_letter_pos_idx_focus_list = {}
        list_valid_mutations = []
        # find all possible valid mutations that can be run with this alignment
        alphabet_set = set(list(self.alphabet))
        for i, letter in enumerate(self.focus_seq):
            if letter in alphabet_set and letter != "-":
                for mut in self.alphabet:
                    pos = start_idx + i
                    if mut != letter:
                        mutant = letter + str(pos) + mut
                        self.mutant_to_letter_pos_idx_focus_list[mutant] = [letter, pos, focus_seq_index]
                        list_valid_mutations.append(mutant)
                focus_seq_index += 1
        self.all_single_mutations = list_valid_mutations

    def save_all_singles(self, output_filename):
        with open(output_filename, "w") as output:
            output.write('mutations')
            for mutation in self.all_single_mutations:
                output.write('\n')
                output.write(mutation)


def generate_mutated_sequences(msa_data, list_mutations):
    """
    Copied from VAE_model.compute_evol_indices.

    Generate mutated sequences using a MSAProcessing data object and list of mutations of the form "A42T" where position
    42 on the wild type is changed from A to T.
    Multiple mutations are separated by colons e.g. "A42T:C9A"

    Returns a tuple (list_valid_mutations, valid_mutated_sequences),
    e.g. (['wt', 'A3T'], {'wt': 'AGAKLI', 'A3T': 'AGTKLI'})
    """
    list_valid_mutations = ['wt']
    valid_mutated_sequences = {}
    valid_mutated_sequences['wt'] = msa_data.focus_seq_trimmed  # first sequence in the list is the wild_type

    # Remove (multiple) mutations that are invalid
    for mutation in list_mutations:
        individual_substitutions = mutation.split(':')
        mutated_sequence = list(msa_data.focus_seq_trimmed)[:]
        fully_valid_mutation = True
        for mut in individual_substitutions:
            wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
            if pos not in msa_data.uniprot_focus_col_to_wt_aa_dict \
                    or msa_data.uniprot_focus_col_to_wt_aa_dict[pos] != wt_aa \
                    or mut not in msa_data.mutant_to_letter_pos_idx_focus_list:
                print("Not a valid mutant: " + mutation)
                fully_valid_mutation = False
                break
            else:
                wt_aa, pos, idx_focus = msa_data.mutant_to_letter_pos_idx_focus_list[mut]
                mutated_sequence[idx_focus] = mut_aa  # perform the corresponding AA substitution

        if fully_valid_mutation:
            list_valid_mutations.append(mutation)
            valid_mutated_sequences[mutation] = ''.join(mutated_sequence)

    return list_valid_mutations, valid_mutated_sequences


# Copied from VAE_model.compute_evol_indices
# One-hot encoding of sequences
def one_hot_3D(seq_keys, seq_name_to_sequence, alphabet, seq_length):
    """
    Take in a list of sequence names/keys and corresponding sequences, and generate a one-hot array according to an alphabet.
    """
    aa_dict = {letter: i for (i, letter) in enumerate(alphabet)}

    one_hot_out = np.zeros((len(seq_keys), seq_length, len(alphabet)))
    for i, seq_key in enumerate(tqdm(seq_keys, desc="One-hot encoding sequences", mininterval=1)):
        sequence = seq_name_to_sequence[seq_key]
        for j, letter in enumerate(sequence):
            if letter in aa_dict:
                k = aa_dict[letter]
                one_hot_out[i, j, k] = 1.0
    one_hot_out = torch.tensor(one_hot_out)
    return one_hot_out


def gen_one_hot_to_sequence(one_hot_tensor, alphabet):
    """Reverse of one_hot_3D. Need the msa_data again. Returns a list of sequences."""
    for seq_tensor in one_hot_tensor:  # iterate through outer dimension
        seq = ""
        letters_idx = seq_tensor.argmax(-1)

        for idx in letters_idx.tolist():  # Could also do map(di.get, letters_idx)
            letter = alphabet[idx]
            seq += letter
        yield seq


def one_hot_to_sequence_list(one_hot_tensor, alphabet):
    return list(gen_one_hot_to_sequence(one_hot_tensor, alphabet))

def get_one_hot_3D_fn(msa_data):
        aa_dict = {letter: i for (i, letter) in enumerate(msa_data.alphabet)}

        def fn(batch_seqs):
            one_hot_out = np.zeros((len(batch_seqs), msa_data.seq_len, len(msa_data.alphabet)))
            for i, sequence in enumerate(batch_seqs):
                for j, letter in enumerate(sequence):
                    if letter in aa_dict:
                        k = aa_dict[letter]
                        one_hot_out[i, j, k] = 1.0
            one_hot_out = torch.tensor(one_hot_out)
            return one_hot_out
        return fn
    
def get_num_cpus():
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        num_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
        print("SLURM_CPUS_PER_TASK:", os.environ['SLURM_CPUS_PER_TASK'])
        print("Using all available cores (calculated using SLURM_CPUS_PER_TASK):", num_cpus)
    else:
        num_cpus = len(os.sched_getaffinity(0)) 
        print("Using all available cores (calculated using len(os.sched_getaffinity(0))):", num_cpus)
    return num_cpus

class OneHotDataset(Dataset):
    def __init__(self, seq_keys, seq_name_to_sequence, alphabet, seq_length, total_length=None):
        self.seq_keys = list(seq_keys)
        self.seq_name_to_sequence = seq_name_to_sequence
        self.alphabet = alphabet
        self.seq_length = seq_length
        self.aa_dict = {letter: i for (i, letter) in enumerate(alphabet)}
        if total_length is None:
            self.total_length = len(self.seq_keys)
        else:
            self.total_length = int(total_length)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        seq_key = self.seq_keys[idx]
        sequence = self.seq_name_to_sequence[seq_key]
        return sequence

class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_loader = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iter_loader)
        except StopIteration:
            # If the inner DataLoader has exhausted the dataset, reset it
            self.iter_loader = super().__iter__()
            batch = next(self.iter_loader)
        return batch

def get_dataloader(msa_data: MSA_processing, batch_size, num_training_steps):
    print("Going to hackily set the length of the dataset to the number of training steps, not the actual number of sequences.")
    dataset = OneHotDataset(
        seq_keys=msa_data.seq_name_to_sequence.keys(), 
        seq_name_to_sequence=msa_data.seq_name_to_sequence, 
        alphabet=msa_data.alphabet, 
        seq_length=msa_data.seq_len) #, total_length=num_training_steps
    # This can take a ton of memory if the weights or num_training_steps*batch_size are large
    sampler = WeightedRandomSampler(weights=msa_data.weights, num_samples=num_training_steps*batch_size, replacement=True)
    num_cpus = 1 # get_num_cpus() # TODO test with only 1 CPU
        
    one_hot_fn = get_one_hot_3D_fn(msa_data)
    
    def collate_fn(batch_seqs):
        # Construct a batch of one-hot-encodings
        batch_seq_tensor = one_hot_fn(batch_seqs)
        return batch_seq_tensor
    
    
    # dataloader = DataLoader(dataset, 
    
    # Other option for avoiding the problem of the dataset running out: Wrap it with an iterable that refreshes it every time
    dataloader = InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch_size, 
        num_workers=num_cpus,  # collate_fn is not parallelized, so no speedup with multiple CPUs
        sampler=sampler, 
        collate_fn=collate_fn,) #pin_memory=True
    
    return dataloader