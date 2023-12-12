import argparse
import os.path
import pandas as pd 


def main(args):

    import json, time, os, sys, glob
    import shutil
    import warnings
    import numpy as np
    import torch
    from torch import optim
    from torch.utils.data import DataLoader
    from torch.utils.data.dataset import random_split, Subset
    import copy
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    import os.path
    import subprocess
    
    from protein_mpnn_utils import loss_nll, loss_smoothed, gather_edges, gather_nodes, gather_nodes_t, cat_neighbors_nodes, _scores, _S_to_seq, tied_featurize, parse_PDB, parse_fasta
    from protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

    if args.seed:
        seed=args.seed
    else:
        seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)   
    
    hidden_dim = 128
    num_layers = 3 
    
    mapping_df = pd.read_csv(args.DMS_reference_file_path)
    DMS_id = mapping_df.iloc[args.DMS_index]['DMS_id']
    DMS_filename = mapping_df.iloc[args.DMS_index]['DMS_filename']
    output_filename = args.output_scores_folder + os.sep + DMS_id + ".csv"
    pdb_file = args.structure_folder + os.sep + mapping_df.iloc[args.DMS_index]['pdb_file']
    if not os.path.exists(args.output_scores_folder):
        os.makedirs(args.output_scores_folder)

    checkpoint_path = args.checkpoint

    NUM_BATCHES = args.num_seq_per_target//args.batch_size
    BATCH_COPIES = args.batch_size
    temperatures = [float(item) for item in args.sampling_temp.split()]
    omit_AAs_list = args.omit_AAs
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    alphabet_dict = dict(zip(alphabet, range(21)))    
    print_all = args.suppress_print == 0 
    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    if os.path.isfile(args.chain_id_jsonl):
        with open(args.chain_id_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            chain_id_dict = json.loads(json_str)
    else:
        chain_id_dict = None
        if print_all:
            print(40*'-')
            print('chain_id_jsonl is NOT loaded')
        
    if os.path.isfile(args.fixed_positions_jsonl):
        with open(args.fixed_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            fixed_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('fixed_positions_jsonl is NOT loaded')
        fixed_positions_dict = None
    
    
    if os.path.isfile(args.pssm_jsonl):
        with open(args.pssm_jsonl, 'r') as json_file:
            json_list = list(json_file)
        pssm_dict = {}
        for json_str in json_list:
            pssm_dict.update(json.loads(json_str))
    else:
        if print_all:
            print(40*'-')
            print('pssm_jsonl is NOT loaded')
        pssm_dict = None
    
    
    if os.path.isfile(args.omit_AA_jsonl):
        with open(args.omit_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            omit_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('omit_AA_jsonl is NOT loaded')
        omit_AA_dict = None
    
    
    if os.path.isfile(args.bias_AA_jsonl):
        with open(args.bias_AA_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            bias_AA_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('bias_AA_jsonl is NOT loaded')
        bias_AA_dict = None
    
    
    if os.path.isfile(args.tied_positions_jsonl):
        with open(args.tied_positions_jsonl, 'r') as json_file:
            json_list = list(json_file)
        for json_str in json_list:
            tied_positions_dict = json.loads(json_str)
    else:
        if print_all:
            print(40*'-')
            print('tied_positions_jsonl is NOT loaded')
        tied_positions_dict = None

    
    if os.path.isfile(args.bias_by_res_jsonl):
        with open(args.bias_by_res_jsonl, 'r') as json_file:
            json_list = list(json_file)
    
        for json_str in json_list:
            bias_by_res_dict = json.loads(json_str)
        if print_all:
            print('bias by residue dictionary is loaded')
    else:
        if print_all:
            print(40*'-')
            print('bias by residue dictionary is not loaded, or not provided')
        bias_by_res_dict = None
   

    if print_all: 
        print(40*'-')
    bias_AAs_np = np.zeros(len(alphabet))
    if bias_AA_dict:
            for n, AA in enumerate(alphabet):
                    if AA in list(bias_AA_dict.keys()):
                            bias_AAs_np[n] = bias_AA_dict[AA]
    
    pdb_dict_list = parse_PDB(pdb_file, ca_only=False)
    dataset_valid = StructureDatasetPDB(pdb_dict_list, truncate=None, max_length=args.max_length)
    all_chain_list = [item[-1:] for item in list(pdb_dict_list[0]) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    if args.pdb_path_chains:
        designed_chain_list = [str(item) for item in args.pdb_path_chains.split()]
    else:
        designed_chain_list = all_chain_list
    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list]
    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]['name']]= (designed_chain_list, fixed_chain_list)

    checkpoint = torch.load(checkpoint_path, map_location=device) 
    noise_level_print = checkpoint['noise_level']
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=hidden_dim, edge_features=hidden_dim, hidden_dim=hidden_dim, num_encoder_layers=num_layers, num_decoder_layers=num_layers, augment_eps=args.backbone_noise, k_neighbors=checkpoint['num_edges'])
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    if print_all:
        print(40*'-')
        print('Number of edges:', checkpoint['num_edges'])
        print(f'Training noise level: {noise_level_print}A')
 
    # Timing
    start_time = time.time()
    total_residues = 0
    protein_list = []
    total_step = 0
    with torch.no_grad():
        test_sum, test_weights = 0., 0.
        dms_df = pd.read_csv(args.DMS_data_folder + os.sep + DMS_filename)
        # For loop should iterate once for our case (not using multiple structures/proteins)
        if len(dataset_valid) > 1:
            raise NotImplementedError("This script is meant for a single protein in dataset_valid. Has not yet been extended to multiple proteins")
        for ix, protein in enumerate(dataset_valid):
            score_list = []
            global_score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(protein) for i in range(BATCH_COPIES)]
            X, S, mask, lengths, chain_M, chain_encoding_all, chain_list_list, visible_list_list, masked_list_list, masked_chain_length_list_list, chain_M_pos, omit_AA_mask, residue_idx, dihedral_mask, tied_pos_list_of_lists_list, pssm_coef, pssm_bias, pssm_log_odds_all, bias_by_res_all, tied_beta = tied_featurize(batch_clones, device, chain_id_dict, fixed_positions_dict, omit_AA_dict, tied_positions_dict, pssm_dict, bias_by_res_dict, ca_only=False)
            pssm_log_odds_mask = (pssm_log_odds_all > args.pssm_threshold).float()
            name_ = batch_clones[0]['name']
            loop_c = 0 
            fasta_names, fasta_seqs = dms_df["mutant"].tolist(), dms_df["mutated_sequence"].tolist()
            loop_c = len(fasta_seqs)
            global_score_list = []
            global_seq_list = []
            for fc in range(loop_c):
                native_score_list = []
                global_native_score_list = []
                input_seq_length = len(fasta_seqs[fc])
                S_input = torch.tensor([alphabet_dict[AA] for AA in fasta_seqs[fc]], device=device)[None,:].repeat(X.shape[0], 1)
                S[:,:input_seq_length] = S_input #assumes that S and S_input are alphabetically sorted for masked_chains
                for j in range(NUM_BATCHES):
                    randn_1 = torch.randn(chain_M.shape, device=X.device)
                    log_probs = model(X, S, mask, chain_M*chain_M_pos, residue_idx, chain_encoding_all, randn_1)
                    mask_for_loss = mask*chain_M*chain_M_pos
                    scores = _scores(S, log_probs, mask_for_loss)
                    native_score = scores.cpu().data.numpy()
                    native_score_list.append(native_score)
                    global_scores = _scores(S, log_probs, mask)
                    global_native_score = global_scores.cpu().data.numpy()
                    global_native_score_list.append(global_native_score)
                native_score = np.concatenate(native_score_list, 0)
                global_native_score = np.concatenate(global_native_score_list, 0)
                ns_mean = native_score.mean()
                ns_mean_print = np.format_float_positional(np.float32(ns_mean), unique=False, precision=4)
                ns_std = native_score.std()
                ns_std_print = np.format_float_positional(np.float32(ns_std), unique=False, precision=4)

                global_ns_mean = global_native_score.mean()
                global_ns_mean_print = np.format_float_positional(np.float32(global_ns_mean), unique=False, precision=4)
                global_ns_std = global_native_score.std()
                global_ns_std_print = np.format_float_positional(np.float32(global_ns_std), unique=False, precision=4)

                ns_sample_size = native_score.shape[0]
                seq_str = _S_to_seq(S[0,], chain_M[0,])
                global_score_list.append(-1*global_native_score[0])
                global_seq_list.append(seq_str)
                if print_all:
                    if fc == 0:
                        print(f'Score for {name_} from PDB, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size},  global score, mean: {global_ns_mean_print}, std: {global_ns_std_print}, sample size: {ns_sample_size}')
                    else:
                        print(f'Score for {name_}_{fc} from FASTA, mean: {ns_mean_print}, std: {ns_std_print}, sample size: {ns_sample_size},  global score, mean: {global_ns_mean_print}, std: {global_ns_std_print}, sample size: {ns_sample_size}')
            # sanity check
            assert all([global_seq_list[i] == fasta_seqs[i] for i in range(len(global_seq_list))])
            score_df = pd.DataFrame({'mutant': fasta_names, 'mutated_sequence': fasta_seqs, 'pmpnn_ll': global_score_list}) 
            score_df.to_csv(output_filename, index=False)
   
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--DMS_reference_file_path',type=str,help='path to DMS reference file')
    argparser.add_argument('--DMS_data_folder',type=str,help="path to folder containing DMS data")
    argparser.add_argument('--structure_folder',type=str,help='folder containing pdb files for each DMS')
    argparser.add_argument('--DMS_index',type=int,help='index of DMS in DMS reference file')
    argparser.add_argument('--checkpoint',type=str,help='path to model')
    argparser.add_argument("--suppress_print", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--seed", type=int, default=0, help="If set to 0 then a random seed will be picked;")
    argparser.add_argument("--backbone_noise", type=float, default=0.00, help="Standard deviation of Gaussian noise to add to backbone atoms")
    argparser.add_argument("--num_seq_per_target", type=int, default=1, help="Number of sequences to generate per target")
    argparser.add_argument("--batch_size", type=int, default=1, help="Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory")
    argparser.add_argument("--max_length", type=int, default=200000, help="Max sequence length")
    argparser.add_argument("--sampling_temp", type=str, default="0.1", help="A string of temperatures, 0.2 0.25 0.5. Sampling temperature for amino acids. Suggested values 0.1, 0.15, 0.2, 0.25, 0.3. Higher values will lead to more diversity.")    
    argparser.add_argument("--output_scores_folder", type=str, help="Path to a folder to output sequences, e.g. /home/out/")
    argparser.add_argument("--pdb_path", type=str, default='', help="Path to a single PDB to be designed")
    argparser.add_argument("--pdb_path_chains", type=str, default='', help="Define which chains need to be designed for a single PDB ")
    argparser.add_argument("--jsonl_path", type=str, help="Path to a folder with parsed pdb into jsonl")
    argparser.add_argument("--chain_id_jsonl",type=str, default='', help="Path to a dictionary specifying which chains need to be designed and which ones are fixed, if not specied all chains will be designed.")
    argparser.add_argument("--fixed_positions_jsonl", type=str, default='', help="Path to a dictionary with fixed positions")
    argparser.add_argument("--omit_AAs", type=list, default='X', help="Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.")
    argparser.add_argument("--bias_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies AA composion bias if neededi, e.g. {A: -1.1, F: 0.7} would make A less likely and F more likely.")
    argparser.add_argument("--bias_by_res_jsonl", default='', help="Path to dictionary with per position bias.") 
    argparser.add_argument("--omit_AA_jsonl", type=str, default='', help="Path to a dictionary which specifies which amino acids need to be omited from design at specific chain indices")
    argparser.add_argument("--pssm_jsonl", type=str, default='', help="Path to a dictionary with pssm")
    argparser.add_argument("--pssm_multi", type=float, default=0.0, help="A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions")
    argparser.add_argument("--pssm_threshold", type=float, default=0.0, help="A value between -inf + inf to restric per position AAs")
    argparser.add_argument("--pssm_log_odds_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--pssm_bias_flag", type=int, default=0, help="0 for False, 1 for True")
    argparser.add_argument("--tied_positions_jsonl", type=str, default='', help="Path to a dictionary with tied positions")
    
    args = argparser.parse_args()    
    main(args)   
