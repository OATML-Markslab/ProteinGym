import numpy as np
import pandas as pd
from collections import defaultdict
import random
import os
import torch
from Bio.Align.Applications import ClustalOmegaCommandline

def filter_msa(msa_data, num_sequences_kept=3):
    """
    Helper function to filter an input MSA msa_data (obtained via process_msa_data) and keep only num_sequences_kept aligned sequences.
    If the MSA already has fewer sequences than num_sequences_kept, we keep the MSA as is.
    If filtering, we always keep the first sequence of the MSA (ie. the wild type) by default.
    Sampling is done without replacement.
    """
    if len(list(msa_data.keys())) <= num_sequences_kept:
        return  msa_data
    filtered_msa = {}
    wt_name = next(iter(msa_data)) 
    filtered_msa[wt_name] = msa_data[wt_name]
    del msa_data[wt_name]
    sequence_names = list(msa_data.keys())
    sequence_names_sampled  = random.sample(sequence_names,k=num_sequences_kept-1)
    for seq in sequence_names_sampled:
        filtered_msa[seq] = msa_data[seq]
    return filtered_msa

def process_msa_data(MSA_data_file):
    """
    Helper function that takes as input a path to a MSA file (expects a2m format) and returns a dict mapping sequence ID to the corresponding AA sequence.
    """
    msa_data = defaultdict(str)
    sequence_name = ""
    with open(MSA_data_file, "r") as msa_file:
        for i, line in enumerate(msa_file):
            line = line.rstrip()
            if line.startswith(">"):
                sequence_name = line
            else:
                msa_data[sequence_name] += line.upper()
    return msa_data

def get_one_hot_sequences_dict(msa_data,MSA_start,MSA_end,vocab):
    vocab_size = len(vocab.keys())
    num_sequences_msa = len(msa_data.keys())
    one_hots = np.zeros((num_sequences_msa,MSA_end-MSA_start,vocab_size))
    for i,seq_name in enumerate(msa_data.keys()):
        sequence = msa_data[seq_name]
        for j,letter in enumerate(sequence):
            if letter in vocab: 
                k = vocab[letter]
                one_hots[i,j,k] = 1.0
    return one_hots

def one_hot(sequence_string,vocab):
    one_hots = np.zeros((len(sequence_string),len(vocab.keys())))
    for j,letter in enumerate(sequence_string):
        if letter in vocab: 
            k = vocab[letter]
            one_hots[j,k] = 1.0
    return one_hots.flatten()

def get_msa_prior(MSA_data_file, MSA_weight_file_name, MSA_start, MSA_end, len_target_seq, vocab, retrieval_aggregation_mode="aggregate_substitution", filter_MSA=True, verbose=False):
    """
    Function to enable retrieval inference mode, via computation of (weighted) pseudocounts of AAs at each position of the retrieved MSA.
    MSA_data_file: (string) path to MSA file (expects a2m format).
    MSA_weight_file_name: (string) path to sequence weights in MSA.
    MSA_start: (int) Sequence position that the MSA starts at (1-indexing).
    MSA_end: (int) Sequence position that the MSA ends at (1-indexing).
    len_target_seq: (int) Full length of sequence to be scored.
    vocab: (dict) Vocabulary of the tokenizer.
    retrieval_aggregation_mode: (string) Mode for retrieval inference (aggregate_substitution Vs aggregate_indel). If None, places a uniform prior over each token.
    filter_MSA: (bool) Whether to filter out sequences with very low hamming similarity (< 0.2) to the reference sequence in the MSA (first sequence).
    verbose: (bool) Whether to print to the console processing details along the way.
    """
    msa_data = process_msa_data(MSA_data_file)
    vocab_size = len(vocab.keys())
    if verbose: print("Target seq len is {}, MSA length is {}, start position is {}, end position is {} and vocab size is {}".format(len_target_seq,MSA_end-MSA_start,MSA_start,MSA_end,vocab_size))

    if filter_MSA:
        if verbose: print("Num sequences in MSA pre filtering: {}".format(len(msa_data.keys())))
        list_sequence_names = list(msa_data.keys())
        focus_sequence_name = list(msa_data.keys())[0]
        ref_sequence_hot = one_hot(msa_data[focus_sequence_name],vocab)
        for sequence_name in list_sequence_names:
            seq_hot = one_hot(msa_data[sequence_name],vocab)
            hamming_similarity_seq_ref = np.dot(ref_sequence_hot,seq_hot) / np.dot(ref_sequence_hot,ref_sequence_hot)
            if hamming_similarity_seq_ref < 0.2:
                del msa_data[sequence_name]
        if verbose: print("Num sequences in MSA post filtering: {}".format(len(msa_data.keys())))

    if MSA_weight_file_name is not None:
        if verbose: print("Using weights in {} for sequences in MSA.".format(MSA_weight_file_name))
        assert os.path.exists(MSA_weight_file_name), "Weights file not located on disk."
        MSA_EVE = MSA_processing(
                MSA_location=MSA_data_file,
                use_weights=True,
                weights_location=MSA_weight_file_name
        )
        #We scan through all sequences to see if we have a weight for them as per EVE pre-processing. We drop them otherwise.
        dropped_sequences=0
        list_sequence_names = list(msa_data.keys())
        MSA_weight=[]
        for sequence_name in list_sequence_names:
            if sequence_name not in MSA_EVE.seq_name_to_sequence:
                dropped_sequences +=1
                del msa_data[sequence_name]
            else:
                MSA_weight.append(MSA_EVE.seq_name_to_weight[sequence_name])
        if verbose: print("Dropped {} sequences from MSA due to absent sequence weights".format(dropped_sequences))
    else:
        MSA_weight = [1] *  len(list(msa_data.keys()))

    if retrieval_aggregation_mode=="aggregate_substitution" or retrieval_aggregation_mode=="aggregate_indel":
        one_hots = get_one_hot_sequences_dict(msa_data,MSA_start,MSA_end,vocab)
        MSA_weight = np.expand_dims(np.array(MSA_weight),axis=(1,2))
        base_rate = 1e-5
        base_rates = np.ones_like(one_hots) * base_rate
        weighted_one_hots = (one_hots + base_rates) * MSA_weight
        MSA_weight_norm_counts = weighted_one_hots.sum(axis=-1).sum(axis=0)
        MSA_weight_norm_counts = np.tile(MSA_weight_norm_counts.reshape(-1,1), (1,vocab_size))
        one_hots_avg = weighted_one_hots.sum(axis=0) / MSA_weight_norm_counts
        msa_prior = np.zeros((len_target_seq,vocab_size))
        print(MSA_start)
        print(MSA_end)
        print(one_hots_avg.shape)
        msa_prior[MSA_start:MSA_end,:]=one_hots_avg
    else:
        msa_prior = np.ones((len_target_seq,vocab_size)) / vocab_size
    
    if verbose:
        for idx, position in enumerate(msa_prior):
            if len(position)!=25:
                print("Size error")
            if not round(position.sum(),2)==1.0:
                print("Position at index {} does not add up to 1: {}".format(idx, position.sum()))
    
    return msa_prior


def update_retrieved_MSA_log_prior_indel(model, MSA_log_prior, MSA_start, MSA_end, mutated_sequence, clustal_hash):
    """
    Function to process MSA when scoring indels.
    To identify positions to add / remove in the retrieved MSA, we append and align the sequence to be scored to the original MSA for that protein family with Clustal Omega.
    If the original MSA is relatively deep (over 100k sequences), we sample (by default) 100k rows at random from that MSA to speed computations.
    MSA sampling is performed only once (for the first sequence to be scored). Subsequent scoring use the same MSA sample.
    """
    if not os.path.isdir(model.MSA_folder + os.sep + "Sampled"):
        os.mkdir(model.MSA_folder + os.sep + "Sampled")
    sampled_MSA_location = model.MSA_folder + os.sep + "Sampled" + os.sep + "Sampled_" + clustal_hash + "_" + model.MSA_filename.split(os.sep)[-1]
    
    if not os.path.exists(sampled_MSA_location):
        msa_data = process_msa_data(model.MSA_filename)
        msa_data_sampled = filter_msa(msa_data, num_sequences_kept=100000) #If MSA has less than 100k sequences, the sample is identical to original MSA
        with open(sampled_MSA_location, 'w') as sampled_write_location:
            for index, key in enumerate(msa_data_sampled):
                key_name = ">REFERENCE_SEQUENCE" if index==0 else key
                msa_data_sampled[key] = msa_data_sampled[key].upper()
                msa_data_sampled[key] = msa_data_sampled[key].replace(".","-")
                sampled_write_location.write(key_name+"\n"+"\n".join([msa_data_sampled[key][i:i+80] for i in range(0, len(msa_data_sampled[key]), 80)])+"\n")
    
    seq_to_align_location = model.MSA_folder + os.sep + "Sampled" + os.sep + "Seq_to_align_" + clustal_hash + "_" + model.MSA_filename.split(os.sep)[-1]
    sequence_text_split = [mutated_sequence[i:i+80] for i in range(0, len(mutated_sequence), 80)]
    sequence_text_split_split_join = "\n".join([">SEQ_TO_SCORE"]+sequence_text_split)
    os.system("echo '"+sequence_text_split_split_join+"' > "+seq_to_align_location)
    
    expanded_MSA_location = model.MSA_folder + os.sep + "Sampled" + os.sep + "Expanded_" + clustal_hash + "_" + model.MSA_filename.split(os.sep)[-1]
    clustalw_cline = ClustalOmegaCommandline(cmd=model.config.clustal_omega_location,
                                            profile1=sampled_MSA_location,
                                            profile2=seq_to_align_location,
                                            outfile=expanded_MSA_location,
                                            force=True)                                        
    stdout, stderr = clustalw_cline()
    msa_data = process_msa_data(expanded_MSA_location)
    aligned_seqA, aligned_seqB = msa_data[">SEQ_TO_SCORE"], msa_data[">REFERENCE_SEQUENCE"]
    try:
        keep_column=[]
        for column_index_pairwise_alignment in range(len(aligned_seqA)):
            if aligned_seqA[column_index_pairwise_alignment]=="-" and aligned_seqB[column_index_pairwise_alignment]=="-":
                continue  # Skips if both are gaps
            elif aligned_seqA[column_index_pairwise_alignment]=="-":
                keep_column.append(False)  # Skips if the query SEQ_TO_SCORE is a gap
            elif aligned_seqB[column_index_pairwise_alignment]=="-":
                MSA_log_prior=torch.cat((MSA_log_prior[:column_index_pairwise_alignment], torch.zeros(MSA_log_prior.shape[1]).view(1,-1).cuda(), MSA_log_prior[column_index_pairwise_alignment:]),dim=0)
                keep_column.append(True) #keep the zero column we just added
            else:
                keep_column.append(True)
        MSA_log_prior = MSA_log_prior[keep_column]
        MSA_end = MSA_start + len(MSA_log_prior)
        if len(MSA_log_prior) == 0:
            print(f"TMP Lood: MSA_log_prior length is 0. aligned_seqA={aligned_seqA}, aligned_seqB={aligned_seqB}, keep_columns={keep_column}")
    except:
        print("Error when processing the following alignment: {}".format(expanded_MSA_location))
    return MSA_log_prior, MSA_start, MSA_end

class MSA_processing:
    def __init__(self,
        MSA_location="",
        theta=0.2,
        use_weights=True,
        weights_location="./data/weights",
        preprocess_MSA=True,
        threshold_sequence_frac_gaps=0.5,
        threshold_focus_cols_frac_gaps=1.0,
        remove_sequences_with_indeterminate_AA_in_focus_cols=True
        ):
        
        """
        This MSA_processing class is directly borrowed from the EVE codebase: https://github.com/OATML-Markslab/EVE
        
        Parameters:
        - msa_location: (path) Location of the MSA data. Constraints on input MSA format: 
            - focus_sequence is the first one in the MSA data
            - first line is structured as follows: ">focus_seq_name/start_pos-end_pos" (e.g., >SPIKE_SARS2/310-550)
            - corespondding sequence data located on following line(s)
            - then all other sequences follow with ">name" on first line, corresponding data on subsequent lines
        - theta: (float) Sequence weighting hyperparameter. Generally: Prokaryotic and eukaryotic families =  0.2; Viruses = 0.01
        - use_weights: (bool) If False, sets all sequence weights to 1. If True, checks weights_location -- if non empty uses that; 
            otherwise compute weights from scratch and store them at weights_location
        - weights_location: (path) Location to load from/save to the sequence weights
        - preprocess_MSA: (bool) performs pre-processing of MSA to remove short fragments and positions that are not well covered.
        - threshold_sequence_frac_gaps: (float, between 0 and 1) Threshold value to define fragments
            - sequences with a fraction of gap characters above threshold_sequence_frac_gaps are removed
            - default is set to 0.5 (i.e., fragments with 50% or more gaps are removed)
        - threshold_focus_cols_frac_gaps: (float, between 0 and 1) Threshold value to define focus columns
            - positions with a fraction of gap characters above threshold_focus_cols_pct_gaps will be set to lower case (and not included in the focus_cols)
            - default is set to 0.3 (i.e., focus positions are the ones with 30% of gaps or less, i.e., 70% or more residue occupancy)
        - remove_sequences_with_indeterminate_AA_in_focus_cols: (bool) Remove all sequences that have indeterminate AA (e.g., B, J, X, Z) at focus positions of the wild type
        """
        np.random.seed(2021)
        self.MSA_location = MSA_location
        self.weights_location = weights_location
        self.theta = theta
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.use_weights = use_weights
        self.preprocess_MSA = preprocess_MSA
        self.threshold_sequence_frac_gaps = threshold_sequence_frac_gaps
        self.threshold_focus_cols_frac_gaps = threshold_focus_cols_frac_gaps
        self.remove_sequences_with_indeterminate_AA_in_focus_cols = remove_sequences_with_indeterminate_AA_in_focus_cols

        self.gen_alignment()
        
    def gen_alignment(self, verbose=False):
        """ Read training alignment and store basics in class instance """
        self.aa_dict = {}
        for i,aa in enumerate(self.alphabet):
            self.aa_dict[aa] = i

        self.seq_name_to_sequence = defaultdict(str)
        name = ""
        with open(self.MSA_location, "r") as msa_data:
            for i, line in enumerate(msa_data):
                line = line.rstrip()
                if line.startswith(">"):
                    name = line
                    if i==0:
                        self.focus_seq_name = name
                else:
                    self.seq_name_to_sequence[name] += line

        
        ## MSA pre-processing to remove inadequate columns and sequences
        if self.preprocess_MSA:
            msa_df = pd.DataFrame.from_dict(self.seq_name_to_sequence, orient='index', columns=['sequence'])
            # Data clean up
            msa_df.sequence = msa_df.sequence.apply(lambda x: x.replace(".","-")).apply(lambda x: ''.join([aa.upper() for aa in x]))
            # Remove columns that would be gaps in the wild type
            non_gap_wt_cols = [aa!='-' for aa in msa_df.sequence[self.focus_seq_name]]
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa for aa,non_gap_ind in zip(x, non_gap_wt_cols) if non_gap_ind]))
            assert 0.0 <= self.threshold_sequence_frac_gaps <= 1.0,"Invalid fragment filtering parameter"
            assert 0.0 <= self.threshold_focus_cols_frac_gaps <= 1.0,"Invalid focus position filtering parameter"
            msa_array = np.array([list(seq) for seq in msa_df.sequence])
            gaps_array = np.array(list(map(lambda seq: [aa=='-' for aa in seq], msa_array)))
            # Identify fragments with too many gaps
            seq_gaps_frac = gaps_array.mean(axis=1)
            seq_below_threshold = seq_gaps_frac <= self.threshold_sequence_frac_gaps
            if verbose: print("Proportion of sequences dropped due to fraction of gaps: "+str(round(float(1 - seq_below_threshold.sum()/seq_below_threshold.shape)*100,2))+"%")
            # Identify focus columns
            columns_gaps_frac = gaps_array[seq_below_threshold].mean(axis=0)
            index_cols_below_threshold = columns_gaps_frac <= self.threshold_focus_cols_frac_gaps
            if verbose: print("Proportion of non-focus columns removed: "+str(round(float(1 - index_cols_below_threshold.sum()/index_cols_below_threshold.shape)*100,2))+"%")
            # Lower case non focus cols and filter fragment sequences
            msa_df['sequence'] = msa_df['sequence'].apply(lambda x: ''.join([aa.upper() if upper_case_ind else aa.lower() for aa, upper_case_ind in zip(x, index_cols_below_threshold)]))
            msa_df = msa_df[seq_below_threshold]
            # Overwrite seq_name_to_sequence with clean version
            self.seq_name_to_sequence = defaultdict(str)
            for seq_idx in range(len(msa_df['sequence'])):
                self.seq_name_to_sequence[msa_df.index[seq_idx]] = msa_df.sequence[seq_idx]

        self.focus_seq = self.seq_name_to_sequence[self.focus_seq_name]
        self.focus_cols = [ix for ix, s in enumerate(self.focus_seq) if s == s.upper() and s!='-'] 
        self.focus_seq_trimmed = [self.focus_seq[ix] for ix in self.focus_cols]
        self.seq_len = len(self.focus_cols)
        self.alphabet_size = len(self.alphabet)

        # Connect local sequence index with uniprot index (index shift inferred from 1st row of MSA)
        try:
            focus_loc = self.focus_seq_name.split("/")[-1]
            start,stop = focus_loc.split("-")
            self.focus_start_loc = int(start)
            self.focus_stop_loc = int(stop)
        except:
            start,stop = 1,len(self.focus_seq)
            self.focus_start_loc = int(start)
            self.focus_stop_loc = int(stop)
        self.uniprot_focus_col_to_wt_aa_dict \
            = {idx_col+int(start):self.focus_seq[idx_col] for idx_col in self.focus_cols} 
        self.uniprot_focus_col_to_focus_idx \
            = {idx_col+int(start):idx_col for idx_col in self.focus_cols} 

        # Move all letters to CAPS; keeps focus columns only
        self.raw_seq_name_to_sequence = self.seq_name_to_sequence.copy()
        for seq_name,sequence in self.seq_name_to_sequence.items():
            sequence = sequence.replace(".","-")
            self.seq_name_to_sequence[seq_name] = [sequence[ix].upper() for ix in self.focus_cols]

        # Remove sequences that have indeterminate AA (e.g., B, J, X, Z) in the focus columns
        if self.remove_sequences_with_indeterminate_AA_in_focus_cols:
            alphabet_set = set(list(self.alphabet))
            seq_names_to_remove = []
            for seq_name,sequence in self.seq_name_to_sequence.items():
                for letter in sequence:
                    if letter not in alphabet_set and letter != "-":
                        seq_names_to_remove.append(seq_name)
                        continue
            seq_names_to_remove = list(set(seq_names_to_remove))
            for seq_name in seq_names_to_remove:
                del self.seq_name_to_sequence[seq_name]

        # Encode the sequences
        self.one_hot_encoding = np.zeros((len(self.seq_name_to_sequence.keys()),len(self.focus_cols),len(self.alphabet)))
        if verbose: print("One-hot encoded sequences shape:" + str(self.one_hot_encoding.shape))
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            sequence = self.seq_name_to_sequence[seq_name]
            for j,letter in enumerate(sequence):
                if letter in self.aa_dict: 
                    k = self.aa_dict[letter]
                    self.one_hot_encoding[i,j,k] = 1.0

        if self.use_weights:
            try:
                self.weights = np.load(file=self.weights_location)
                if verbose: print("Loaded sequence weights from disk")
            except:
                if verbose: print ("Computing sequence weights")
                list_seq = self.one_hot_encoding
                list_seq = list_seq.reshape((list_seq.shape[0], list_seq.shape[1] * list_seq.shape[2]))
                def compute_weight(seq):
                    number_non_empty_positions = np.dot(seq,seq)
                    if number_non_empty_positions>0:
                        denom = np.dot(list_seq,seq) / np.dot(seq,seq) 
                        denom = np.sum(denom > 1 - self.theta) 
                        return 1/denom
                    else:
                        return 0.0 #return 0 weight if sequence is fully empty
                self.weights = np.array(list(map(compute_weight,list_seq)))
                np.save(file=self.weights_location, arr=self.weights)
        else:
            # If not using weights, use an isotropic weight matrix
            if verbose: print("Not weighting sequence data")
            self.weights = np.ones(self.one_hot_encoding.shape[0])

        self.Neff = np.sum(self.weights)
        self.num_sequences = self.one_hot_encoding.shape[0]
        self.seq_name_to_weight={}
        for i,seq_name in enumerate(self.seq_name_to_sequence.keys()):
            self.seq_name_to_weight[seq_name]=self.weights[i]

        if verbose:
            print ("Neff =",str(self.Neff))
            print ("Data Shape =",self.one_hot_encoding.shape)