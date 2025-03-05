import argparse
import pathlib
import os,sys
import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from baselines.esm import esm
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer

from utils.scoring_utils import get_optimal_window, set_mutant_offset, undo_mutant_offset
from utils.data_utils import DMS_file_cleanup
from utils.msa_utils import MSA_processing

def standardization(x):
    """Assumes input is numpy array or pandas series"""
    return (x - x.mean()) / x.std()

def sample_msa(filename: str, nseq: int, sampling_strategy: str, random_seed: int, weight_filename=None, processed_msa=None):
    """Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    print("Sampling sequences from MSA with strategy: "+str(sampling_strategy))
    random.seed(random_seed)
    if sampling_strategy=='first_x_rows':
        msa = [
            (record.description, str(record.seq))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
        ]
    elif sampling_strategy=='random':
        msa = [
            (record.description, str(record.seq)) for record in SeqIO.parse(filename, "fasta")
        ]
        nseq = min(len(msa),nseq)
        msa = random.sample(msa, nseq)
    elif sampling_strategy=='sequence-reweighting':
        # If MSA has already been processed, just use it here
        if processed_msa is None:
            if weight_filename is None:
                print("Need weight filename if using sequence-reweighting sample strategy")
            MSA = MSA_processing(
                MSA_location=filename,
                use_weights=True,
                weights_location=weight_filename
            )
            print("Name of focus_seq: "+str(MSA.focus_seq_name))
        else:
            MSA = processed_msa

        # Make sure we always keep the WT in the subsampled MSA
        msa = [(MSA.focus_seq_name,MSA.raw_seq_name_to_sequence[MSA.focus_seq_name])]

        non_wt_weights = np.array([w for k, w in MSA.seq_name_to_weight.items() if k != MSA.focus_seq_name])
        non_wt_sequences = [(k, s) for k, s in MSA.seq_name_to_sequence.items() if k != MSA.focus_seq_name]
        non_wt_weights = non_wt_weights / non_wt_weights.sum() # Renormalize weights

        # Sample the rest of the MSA according to their weights
        if len(non_wt_sequences) > 0:
            msa.extend(random.choices(non_wt_sequences, weights=non_wt_weights, k=nseq-1))

        print("Check sum weights MSA: "+str(non_wt_weights.sum()))

    msa = [(desc, ''.join(seq) if isinstance(seq, list) else seq) for desc, seq in msa]
    msa = [(desc, seq.upper()) for desc, seq in msa]
    print("First 10 elements of sampled MSA: ")
    print(msa[:10])
    #[('CCDB_ECOLI/1-101', 'MQFKVYTYKRESRYRLFVDVQSDIIDTPGRRMVIPLASARLLSDKVSRELYPVVHIGDESWRMMTTDMASVPVSVIGEEVADLSHRENDIKNAINLMFWGI'), ('UniRef100_A0A6M6E7X5/22-113', '.........--GKNIPCVILQNNKGNASNTTIIVPIIAESNYIKSSPTYVHIRKYNLDEDSIAVCDQIRVIDKKRITKAAKALSEEKKQIEDGI--.....'), ('UniRef100_A0A6M6E7X5/144-229', '.........--EKAFPALVIGRDNK-EKQTLLIAPLLQAKKR.YLLPTQAKIPNGCLSDGKILSLEQMRVIDKSRIVSQNIKFSDTLLEIE-----.....'), ('UniRef100_A0A6M6E7X5/256-332', '........SEQRGLRPCLIIQNDTGNKLSTTIVLPLTSSVPKVDLVVNVIVKQEEFVGRSSIVLCNQIQTIDSSRIKQV-----------------.....'), ('UniRef100_A0A6M6E7X5/406-489', '.........------PAVCIQNSYGNHYSVLIVAPLISSKRT.RLLPTQVKIDFNDSGELMVAALEQVRVIDKRRVVDVLDELPEEKKEILNAYCVS....'), ('UniRef100_UPI000B4B97B8/22-115', '........SEQKGERPAVVVQNDFGNRASTTLIVPLTSNFKT.-EIPTHVNISSKELGVDSIALCEQVRVISKERIIQVRTILSKEVKEIDDALLISF...'), ('UniRef100_UPI000B4B97B8/135-229', '........NEQKGNRPAIVIQNDVGNKYSTLIVAPLQLKKKK.-RLPTHVEIPGNLISKDSIALLEQVRVVDKERVTGVVQNLTEEFKNIEEALLVSF...'), ('UniRef100_UPI0009C11867/73-161', '.......GSEQNGLRPVVIIQNNLGNKYGTLIVAPITSQDKK.-DLPVHSEIYNNSLEKDSTILLEQVTTIDKNKVKEFVGHLTRNEKKLNIALAR.....'), ('UniRef100_F4A1A6/19-111', '........SEQGGVRPVLVVQNDIGNKYSTVIVAAITSQINK.AKLPTHVEISASDYGKDSVILLEQIRTIDKKRLREKIGYLSAETKKVDEALQISF...'), ('UniRef100_A0A150FSC1/22-114', '........SEQGGVRPVLVIQNDIGNKYSTVIVAAITSQINK.AKLPIHIEIKANGLNKDSVVLLEQIRTIDKKRLREKIGHFDEEKEKVDQAIQISL...')]
    return msa


def process_msa(filename: str, weight_filename: str, filter_msa: bool, path_to_hhfilter: str, hhfilter_min_cov=75, hhfilter_max_seq_id=100, hhfilter_min_seq_id=0) -> List[Tuple[str, str]]:
    if filter_msa:
        input_folder = '/'.join(filename.split('/')[:-1])
        msa_name = filename.split('/')[-1].split('.')[0]
        if not os.path.isdir(input_folder+os.sep+'preprocessed'):
            os.mkdir(input_folder+os.sep+'preprocessed')
        if not os.path.isdir(input_folder+os.sep+'hhfiltered'):
            os.mkdir(input_folder+os.sep+'hhfiltered')
        preprocessed_filename = input_folder+os.sep+'preprocessed'+os.sep+msa_name
        os.system('cat '+filename+' | tr  "."  "-" >> '+preprocessed_filename+'.a2m')
        os.system('dd if='+preprocessed_filename+'.a2m of='+preprocessed_filename+'_UC.a2m conv=ucase')
        output_filename = input_folder+os.sep+'hhfiltered'+os.sep+msa_name+'_hhfiltered_cov_'+str(hhfilter_min_cov)+'_maxid_'+str(hhfilter_max_seq_id)+'_minid_'+str(hhfilter_min_seq_id)+'.a2m'
        os.system(path_to_hhfilter+os.sep+'bin/hhfilter -cov '+str(hhfilter_min_cov)+' -id '+str(hhfilter_max_seq_id)+' -qid '+str(hhfilter_min_seq_id)+' -i '+preprocessed_filename+'_UC.a2m -o '+output_filename)
        filename = output_filename

    MSA = MSA_processing(
        MSA_location=filename,
        use_weights=True,
        weights_location=weight_filename
    )
    print("Name of focus_seq: "+str(MSA.focus_seq_name))
    return MSA


def create_parser():
    parser = argparse.ArgumentParser(
        description="Label a deep mutational scan with predictions from an ensemble of ESM-1v models."  # noqa
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="MSA_transformer Vs ESM1v Vs ESM1b",
        default="MSA_transformer",
        nargs="+",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
        nargs="+",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        help="Base sequence to which mutations were applied",
    )
    parser.add_argument(
        "--dms-input",
        type=pathlib.Path,
        help="CSV file containing the deep mutational scan",
    )
    parser.add_argument(
        "--dms_index",
        type=int,
        help="Index of DMS in mapping file",
    )
    parser.add_argument(
        "--dms_mapping",
        type=str,
        help="Location of DMS_mapping",
    )
    parser.add_argument(
        "--mutation-col",
        type=str,
        default="mutant",
        help="column in the deep mutational scan labeling the mutation as 'AiB'"
    )
    parser.add_argument(
        "--dms-output",
        type=pathlib.Path,
        help="Output file containing the deep mutational scan along with predictions",
    )
    parser.add_argument(
        "--offset-idx",
        type=int,
        default=1,
        help="Offset of the mutation positions in `--mutation-col`"
    )
    parser.add_argument(
        "--scoring-strategy",
        type=str,
        default="wt-marginals",
        choices=["wt-marginals", "pseudo-ppl", "masked-marginals"],
        help=""
    )
    parser.add_argument(
        "--msa-path",
        type=pathlib.Path,
        help="path to MSA (required for MSA Transformer)"
    )
    parser.add_argument(
        "--msa-sampling-strategy",
        type=str,
        default='sequence-reweighting',
        help="Strategy to sample sequences from MSA [sequence-reweighting|random|first_x_rows]"
    )
    parser.add_argument(
        "--msa-samples",
        type=int,
        default=400,
        help="number of sequences to randomly sample from the MSA"
    )
    parser.add_argument(
        "--msa-weights-folder",
        type=str,
        default=None,
        help="Folder with weights to sample MSA sequences in 'sequence-reweighting' scheme"
    )
    parser.add_argument(
        '--seeds',
        type=int,
        default=1, 
        help='Random seed used during training',
        nargs="+"
    )
    parser.add_argument(
        '--filter-msa',
        action='store_true',
        help='Whether to use hhfilter to filter input MSA before sampling'
    )
    parser.add_argument(
        '--hhfilter-min-cov',
        type=int,
        default=75, 
        help='minimum coverage with query (%)'
    )
    parser.add_argument(
        '--hhfilter-max-seq-id',
        type=int,
        default=90, 
        help='maximum pairwise identity (%)'
    )
    parser.add_argument(
        '--hhfilter-min-seq-id',
        type=int,
        default=0, 
        help='minimum sequence identity with query (%)'
    )
    parser.add_argument(
        '--path-to-hhfilter',
        type=str,
        default='/n/groups/marks/software/hhsuite/hhsuite-3.3.0', 
        help='Path to hhfilter binaries'
    )
    parser.add_argument(
        '--scoring-window',
        type=str,
        default='optimal', 
        help='Approach to handle long sequences [optimal|overlapping]'
    )
    parser.add_argument(
        '--overwrite-prior-scores',
        action='store_true',
        help='Whether to overwrite prior scores in the dataframe'
    )
    #No ref file provided
    parser.add_argument('--target_seq', default=None, type=str, help='WT sequence mutated in the assay')
    parser.add_argument('--weight_file_name', default=None, type=str, help='Wild type sequence mutated in the assay (to be provided if not using a reference file)')
    parser.add_argument('--MSA_start', default=None, type=int, help='Index of first AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Index of last AA covered by the MSA relative to target_seq coordinates (1-indexing)')
    
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def label_row(row, sequence, token_probs, alphabet, offset_idx):
    score=0
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

        # add 1 for BOS
        score += (token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]).item()
    return score

def get_mutated_sequence(row, wt_sequence, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert wt_sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
    # modify the sequence
    sequence = wt_sequence[:idx] + mt + wt_sequence[(idx + 1) :]
    return sequence
def compute_pppl(sequence, model, alphabet, MSA_data = None, mode = "ESM1v"):
    # encode the sequence
    data = [
        ("protein1", sequence),
    ]
    if mode == "MSA_Transformer":
        data = [data + MSA_data[0]]
    batch_converter = alphabet.get_batch_converter()

    _, _, batch_tokens = batch_converter(data)
    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)  # This might OOM because the MSA batch is large (400 by default)
        if mode == "ESM1v":
            log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
        elif mode == "MSA_Transformer":
            log_probs.append(token_probs[0, 0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)


def main(args):
    if not os.path.exists(args.dms_output): os.mkdir(args.dms_output)
    print("Arguments:", args)

    # Load the deep mutational scan
    mutant_col = args.mutation_col  # Default "mutant"
    if args.dms_index is not None:
        mapping_protein_seq_DMS = pd.read_csv(args.dms_mapping)
        DMS_id = mapping_protein_seq_DMS["DMS_id"][args.dms_index]
        print("Compute scores for DMS: "+str(DMS_id))
        row = mapping_protein_seq_DMS[mapping_protein_seq_DMS["DMS_id"]==DMS_id]
        if len(row) == 0:
            raise ValueError("No mappings found for DMS: "+str(DMS_id))
        elif len(row) > 1:
            raise ValueError("Multiple mappings found for DMS: "+str(DMS_id))
        
        row = row.iloc[0]
        row = row.replace(np.nan, "")  # Makes it more manageable to use in strings

        args.sequence = row["target_seq"].upper()
        args.dms_input = str(args.dms_input)+os.sep+row["DMS_filename"]

        mutant_col = row["DMS_mutant_column"] if "DMS_mutant_column" in mapping_protein_seq_DMS.columns else mutant_col
        args.dms_output=str(args.dms_output)+os.sep+DMS_id+'.csv'
        
        target_seq_start_index = row["start_idx"] if "start_idx" in mapping_protein_seq_DMS.columns and row["start_idx"]!="" else 1
        target_seq_end_index = target_seq_start_index + len(args.sequence) 
        
        if "MSA_transformer" in args.model_type:  # model_type is a list
            # Check MSA_filename exists (might be NaN / empty)
            msa_filename = row["MSA_filename"]
            if msa_filename == "":
                raise ValueError("No MSA found for DMS: "+str(DMS_id))
            
            args.msa_path= str(args.msa_path)+os.sep+msa_filename # msa_path is expected to be the path to the directory where MSAs are located.
            
            msa_start_index = int(row["MSA_start"]) if "MSA_start" in mapping_protein_seq_DMS.columns else 1
            msa_end_index = int(row["MSA_end"]) if "MSA_end" in mapping_protein_seq_DMS.columns else len(args.sequence)
            
            MSA_weight_file_name = args.msa_weights_folder + os.sep + row["weight_file_name"] if ("weight_file_name" in mapping_protein_seq_DMS.columns and args.msa_weights_folder is not None) else None
            if ((target_seq_start_index!=msa_start_index) or (target_seq_end_index!=msa_end_index)):
                args.sequence = args.sequence[msa_start_index-1:msa_end_index]
                target_seq_start_index = msa_start_index
                target_seq_end_index = msa_end_index
        df = pd.read_csv(args.dms_input)
    else:
        
        DMS_id = str(args.dms_input).split(os.sep)[-1].split('.csv')[0]
        args.dms_output=str(args.dms_output)+os.sep+DMS_id+'.csv'
        target_seq_start_index = args.offset_idx
        args.sequence = args.target_seq.upper()
        if (args.MSA_start is None) or (args.MSA_end is None): 
            if args.msa_path: print("MSA start and end not provided -- Assuming the MSA is covering the full WT sequence")
            args.MSA_start = 1
            args.MSA_end = len(args.target_seq)
        msa_start_index = args.MSA_start
        msa_end_index = args.MSA_end
        MSA_weight_file_name = args.msa_weights_folder + os.sep + args.weight_file_name if args.msa_weights_folder is not None else None
        df = pd.read_csv(args.dms_input)
    
    if len(df) == 0:
        raise ValueError("No rows found in the dataframe")
    print(f"df shape: {df.shape}", flush=True)

    # inference for each model
    print("Starting model scoring")
    for model_location in args.model_location:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
        model_location = model_location.split("/")[-1].split(".")[0]
        model.eval()
        if torch.cuda.is_available() and not args.nogpu:
            model = model.cuda()
            print("Transferred model to GPU")
        else:
            print(f"Not using GPU. torch.cuda.is_available(): {torch.cuda.is_available()}, args.nogpu: {args.nogpu}")

        batch_converter = alphabet.get_batch_converter()

        if isinstance(model, MSATransformer):
            args.offset_idx = msa_start_index
            # Process MSA once, then sample from it
            processed_msa = process_msa(filename=args.msa_path, weight_filename=MSA_weight_file_name, filter_msa=args.filter_msa, hhfilter_min_cov=args.hhfilter_min_cov, hhfilter_max_seq_id=args.hhfilter_max_seq_id, hhfilter_min_seq_id=args.hhfilter_min_seq_id, path_to_hhfilter=args.path_to_hhfilter)
            for seed in args.seeds:
                if os.path.exists(args.dms_output):
                    prior_score_df = pd.read_csv(args.dms_output)
                    if f"{model_location}_seed{seed}" in prior_score_df.columns and not args.overwrite_prior_scores:
                        print(f"Skipping seed {seed} as it is already in the dataframe")
                        df = prior_score_df
                        continue
                else:
                    prior_score_df = None 
                data = [sample_msa(sampling_strategy=args.msa_sampling_strategy, filename=args.msa_path, nseq=args.msa_samples, weight_filename=MSA_weight_file_name, processed_msa=processed_msa, random_seed=seed)]
                assert (args.scoring_strategy in ["masked-marginals","pseudo-ppl"]), "Zero-shot scoring strategy not supported with MSA Transformer"

                batch_labels, batch_strs, batch_tokens = batch_converter(data)
                print(f"Batch sizes: {batch_tokens.size()}")

                if args.scoring_strategy == "masked-marginals":
                    all_token_probs = []
                    for i in tqdm(range(batch_tokens.size(2)), desc="Scoring masked-marginals"):
                        batch_tokens_masked = batch_tokens.clone()
                        batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
                        if batch_tokens.size(-1) > 1024:
                            large_batch_tokens_masked=batch_tokens_masked.clone()
                            start, end = get_optimal_window(mutation_position_relative=i, seq_len_wo_special=len(args.sequence)+2, model_window=1024)
                            print("Start index {} - end index {}".format(start,end))
                            batch_tokens_masked = large_batch_tokens_masked[:,:,start:end]
                        else:
                            start=0
                        with torch.no_grad():
                            token_probs = torch.log_softmax(
                                model(batch_tokens_masked.cuda())["logits"], dim=-1
                            )
                        all_token_probs.append(token_probs[:, 0, i-start].detach().cpu())  # vocab size
                    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                    df[f"{model_location}_seed{seed}"] = df.apply(
                        lambda row: label_row(
                            row[mutant_col], args.sequence, token_probs.detach().cpu(), alphabet, args.offset_idx
                        ),
                        axis=1,
                    )
                elif args.scoring_strategy == "pseudo-ppl":
                    tqdm.pandas()
                    if 'mutated_sequence' not in df:
                        df['mutated_sequence'] = df.progress_apply(
                            lambda row: get_mutated_sequence(
                                row[mutant_col], args.sequence, args.offset_idx
                                ),
                            axis=1,
                        )
                    df[f"{model_location}_seed{seed}"] = df.progress_apply(
                        lambda row: compute_pppl(
                            row['mutated_sequence'], model, alphabet, MSA_data = data, mode = "MSA_Transformer"
                            ),
                        axis=1,
                    )
                if os.path.exists(args.dms_output) and not args.overwrite_prior_scores:
                    prior_score_df = pd.read_csv(args.dms_output)
                    assert f"{model_location}_seed{seed}" not in prior_score_df.columns, f"Column {model_location}_seed{seed} already exists in {args.dms_output}"
                    prior_score_df = prior_score_df.merge(df[[f"{model_location}_seed{seed}","mutant"]],on="mutant")
                    prior_score_df.to_csv(args.dms_output, index=False)
                    df = prior_score_df 
                else:
                    df.to_csv(args.dms_output, index=False)
        else:
            args.offset_idx = target_seq_start_index
            data = [
                ("protein1", args.sequence),
            ]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            if args.scoring_strategy == "wt-marginals":
                with torch.no_grad():
                    if batch_tokens.size(1) > 1024 and args.scoring_window=="overlapping": 
                        batch_size, seq_len = batch_tokens.shape #seq_len includes BOS and EOS
                        token_probs = torch.zeros((batch_size,seq_len,len(alphabet))).cuda() # Note: batch_size = 1 (need to keep batch dimension to score with model though)
                        token_weights = torch.zeros((batch_size,seq_len)).cuda()
                        weights = torch.ones(1024).cuda() # 1 for 256≤i<1022-256
                        for i in range(1,257):
                            weights[i] = 1 / (1 + math.exp(-(i-128)/16))
                        for i in range(1022-256,1023):
                            weights[i] = 1 / (1 + math.exp((i-1022+128)/16))
                        start_left_window = 0
                        end_left_window = 1023 #First window is indexed [0-1023]
                        start_right_window = (batch_tokens.size(1) - 1) - 1024 + 1 #Last index is len-1
                        end_right_window = batch_tokens.size(1) - 1
                        while True: 
                            # Left window update
                            left_window_probs = torch.log_softmax(model(batch_tokens[:,start_left_window:end_left_window+1].cuda())["logits"], dim=-1)
                            token_probs[:,start_left_window:end_left_window+1] += left_window_probs * weights.view(-1,1)
                            token_weights[:,start_left_window:end_left_window+1] += weights
                            # Right window update
                            right_window_probs = torch.log_softmax(model(batch_tokens[:,start_right_window:end_right_window+1].cuda())["logits"], dim=-1)
                            token_probs[:,start_right_window:end_right_window+1] += right_window_probs * weights.view(-1,1)
                            token_weights[:,start_right_window:end_right_window+1] += weights
                            if end_left_window > start_right_window:
                                #overlap between windows in that last scoring so we break from the loop
                                break
                            start_left_window+=511
                            end_left_window+=511
                            start_right_window-=511
                            end_right_window-=511
                        #If central overlap not wide engouh, we add one more window at the center
                        final_overlap = end_left_window - start_right_window + 1
                        if final_overlap < 511:
                            start_central_window = int(seq_len / 2) - 512
                            end_central_window = start_central_window + 1023
                            central_window_probs = torch.log_softmax(model(batch_tokens[:,start_central_window:end_central_window+1].cuda())["logits"], dim=-1)
                            token_probs[:,start_central_window:end_central_window+1] += central_window_probs * weights.view(-1,1)
                            token_weights[:,start_central_window:end_central_window+1] += weights
                        #Weight normalization
                        token_probs = token_probs / token_weights.view(-1,1) #Add 1 to broadcast
                    else:                    
                        token_probs = torch.log_softmax(model(batch_tokens.cuda())["logits"], dim=-1)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[mutant_col],
                        args.sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif args.scoring_strategy == "masked-marginals":
                print("Scoring with masked-marginals and model {}".format(model_location))
                all_token_probs = []
                for i in tqdm(range(batch_tokens.size(1)), desc="Scoring masked-marginals"):
                    batch_tokens_masked = batch_tokens.clone()
                    batch_tokens_masked[0, i] = alphabet.mask_idx
                    if batch_tokens.size(1) > 1024 and args.scoring_window=="optimal": 
                        large_batch_tokens_masked=batch_tokens_masked.clone()
                        start, end = get_optimal_window(mutation_position_relative=i, seq_len_wo_special=len(args.sequence)+2, model_window=1024)
                        batch_tokens_masked = large_batch_tokens_masked[:,start:end]
                    elif batch_tokens.size(1) > 1024 and args.scoring_window=="overlapping": 
                        print("Overlapping not yet implemented for masked-marginals")
                        sys.exit(0)
                    else:
                        start=0
                    with torch.no_grad():
                        token_probs = torch.log_softmax(model(batch_tokens_masked.cuda())["logits"], dim=-1)
                    all_token_probs.append(token_probs[:, i-start])  # vocab size
                token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
                df[model_location] = df.apply(
                    lambda row: label_row(
                        row[mutant_col],
                        args.sequence,
                        token_probs,
                        alphabet,
                        args.offset_idx,
                    ),
                    axis=1,
                )
            elif args.scoring_strategy == "pseudo-ppl":
                tqdm.pandas()
                if 'mutated_sequence' not in df:
                    df['mutated_sequence'] = df.progress_apply(
                        lambda row: get_mutated_sequence(
                            row[mutant_col], args.sequence, args.offset_idx
                            ),
                        axis=1,
                    )
                df[model_location] = df.progress_apply(
                    lambda row: compute_pppl(
                        row['mutated_sequence'], model, alphabet
                        ),
                    axis=1,
                )
    # Compute ensemble score Ensemble_ESM1v, standardizes each model score and then averages them
    # note this assumes that all the input model checkpoints are ESM1v
    if "ESM1v" in args.model_type:
        df["Ensemble_ESM1v"] = 0.0
        for model_location in args.model_location:
            model_location = model_location.split("/")[-1].split(".")[0]
            df["Ensemble_ESM1v"] += df[model_location]
        df["Ensemble_ESM1v"] /= len(args.model_location)
    elif "MSA_transformer" in args.model_type:
        df[f"{model_location}_ensemble"] = 0.0
        for seed in args.seeds:
            df[f"{model_location}_ensemble"] += df[f"{model_location}_seed{seed}"]
        df[f"{model_location}_ensemble"] /= len(args.seeds)
    df.to_csv(args.dms_output,index=False)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
