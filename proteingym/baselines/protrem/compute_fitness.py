
import torch
import os
import json
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModelForMaskedLM
from argparse import ArgumentParser

amino_acid_properties = {
    'A': {'hydrophobicity': 1.8,  'charge':  0, 'polarity':  0, 'molecular_weight':  89.09, 'volume':  88.6},
    'R': {'hydrophobicity': -4.5, 'charge': +1, 'polarity':  1, 'molecular_weight': 174.20, 'volume': 173.4},
    'N': {'hydrophobicity': -3.5, 'charge':  0, 'polarity':  1, 'molecular_weight': 132.12, 'volume': 114.1},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity':  1, 'molecular_weight': 133.10, 'volume': 111.1},
    'C': {'hydrophobicity': 2.5,  'charge':  0, 'polarity':  0, 'molecular_weight': 121.15, 'volume': 108.5},
    'Q': {'hydrophobicity': -3.5, 'charge':  0, 'polarity':  1, 'molecular_weight': 146.15, 'volume': 143.8},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity':  1, 'molecular_weight': 147.13, 'volume': 138.4},
    'G': {'hydrophobicity': -0.4, 'charge':  0, 'polarity':  0, 'molecular_weight':  75.07, 'volume':  60.1},
    'H': {'hydrophobicity': -3.2, 'charge':  0, 'polarity':  1, 'molecular_weight': 155.16, 'volume': 153.2},
    'I': {'hydrophobicity': 4.5,  'charge':  0, 'polarity':  0, 'molecular_weight': 131.17, 'volume': 166.7},
    'L': {'hydrophobicity': 3.8,  'charge':  0, 'polarity':  0, 'molecular_weight': 131.17, 'volume': 166.7},
    'K': {'hydrophobicity': -3.9, 'charge': +1, 'polarity':  1, 'molecular_weight': 146.19, 'volume': 168.6},
    'M': {'hydrophobicity': 1.9,  'charge':  0, 'polarity':  0, 'molecular_weight': 149.21, 'volume': 162.9},
    'F': {'hydrophobicity': 2.8,  'charge':  0, 'polarity':  0, 'molecular_weight': 165.19, 'volume': 189.9},
    'P': {'hydrophobicity': -1.6, 'charge':  0, 'polarity':  0, 'molecular_weight': 115.13, 'volume': 112.7},
    'S': {'hydrophobicity': -0.8, 'charge':  0, 'polarity':  1, 'molecular_weight': 105.09, 'volume':  89.0},
    'T': {'hydrophobicity': -0.7, 'charge':  0, 'polarity':  1, 'molecular_weight': 119.12, 'volume': 116.1},
    'W': {'hydrophobicity': -0.9, 'charge':  0, 'polarity':  0, 'molecular_weight': 204.23, 'volume': 227.8},
    'Y': {'hydrophobicity': -1.3, 'charge':  0, 'polarity':  1, 'molecular_weight': 181.19, 'volume': 193.6},
    'V': {'hydrophobicity': 4.2,  'charge':  0, 'polarity':  0, 'molecular_weight': 117.15, 'volume': 140.0},
}
device = "cuda" if torch.cuda.is_available() else "cpu"


def read_multi_fasta(file_path):
    """
    params:
        file_path: path to a fasta file
    return:
        a dictionary of sequences
    """
    sequences = {}
    current_sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences[header] = current_sequence.upper().replace('-', '<pad>').replace('.', '<pad>')
                    current_sequence = ''
                header = line
            else:
                current_sequence += line
        if current_sequence:
            sequences[header] = current_sequence
    return sequences


def read_seq(fasta):
    for record in SeqIO.parse(fasta, "fasta"):
        return str(record.seq)

def count_matrix_from_residue_alignment(tokenizer, alignment_dict):
    alignment_seqs = list(alignment_dict.values())
    try:
        aln_start, aln_end = list(alignment_dict.keys())[0].split('/')[-1].split('-')
    except:
        aln_start, aln_end = 1, len(alignment_seqs[0])
    print(f">>> Alignment start: {aln_start}, end: {aln_end}")
    print(f">>> Start tokenizing {len(alignment_seqs)} residue alignment sequences")
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_ids = tokenized_results["input_ids"][:,1:-1]
    return alignment_ids, int(aln_start)-1, int(aln_end)
    # count distribution of each column, [seq_len, vocab_size]
    count_matrix = torch.zeros(alignment_ids.size(1), tokenizer.vocab_size)
    for i in tqdm(range(alignment_ids.size(1))):
        count_matrix[i] = torch.bincount(alignment_ids[:,i], minlength=tokenizer.vocab_size)
    # calculate coverage of each column and normalize count matrix
    # coverage = (1.0 - (count_matrix == tokenizer.pad_token_id).float().mean(dim=-1)).unsqueeze(-1).to(device)
    
    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
    # count_matrix = count_matrix * coverage
    return count_matrix, int(aln_start)-1, int(aln_end)


def count_matrix_from_structure_alignment(tokenizer, alignment_dict):
    alignment_seqs = list(alignment_dict.values())
    print(f">>> Start tokenizing {len(alignment_seqs)} structure alignment sequences")
    if len(alignment_seqs) == 0:
        return None
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_ids = tokenized_results["input_ids"][:,1:-1]
    return alignment_ids
    # count distribution of each column, [seq_len, vocab_size]
    count_matrix = torch.zeros(alignment_ids.size(1), tokenizer.vocab_size)
    for i in tqdm(range(alignment_ids.size(1))):
        count_matrix[i] = torch.bincount(alignment_ids[:,i], minlength=tokenizer.vocab_size)
    return count_matrix
    count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
    return count_matrix
    


def calculate_property_difference(wild_aa, mutant_aa, weights=None):
    properties = amino_acid_properties[wild_aa].keys()
    if weights is None:
        weights = {prop: 1 for prop in properties}
    differences = []
    for prop in properties:
        wild_value = amino_acid_properties[wild_aa][prop]
        mutant_value = amino_acid_properties[mutant_aa][prop]
        difference = abs(mutant_value - wild_value)
        weighted_diff = weights.get(prop, 1) * difference
        differences.append(weighted_diff)
    return differences

def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor([shift_structure_sequence,], dtype=torch.long)


@torch.no_grad()
def score_protein(model, tokenizer, residue_fasta, structure_fasta, mutant_df, 
                  alpha=0.7, aa_seq_aln_file=None, struc_seq_aln_file=None,
                  sample_size=None, sample_ratio=1.0, sample_times=1):
    sequence = read_seq(residue_fasta)
    structure_sequence = read_seq(structure_fasta)

    structure_sequence = [int(i) for i in structure_sequence.split(",")]
    ss_input_ids = tokenize_structure_sequence(structure_sequence).to(device)
    tokenized_results = tokenizer([sequence], return_tensors="pt")
    input_ids = tokenized_results["input_ids"].to(device)
    attention_mask = tokenized_results["attention_mask"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=ss_input_ids,
        labels=input_ids,
    )

    # loss = outputs.loss.item()
    logits = outputs.logits[0]
    logits = torch.log_softmax(logits[1:-1, :], dim=-1)
    
    
    if aa_seq_aln_file is not None and struc_seq_aln_file is None:
        print(">>> Using residue sequence alignment matrix...")
        alignment_dict = read_multi_fasta(aa_seq_aln_file)
        alignment_matrix, aln_start, aln_end = count_matrix_from_residue_alignment(tokenizer, alignment_dict)
        for sample in range(sample_times):
            if sample_ratio < 1.0:
                print(f">>> Sample {sample+1}/{sample_times} with ratio {sample_ratio}")
                sample_size = int(len(alignment_matrix) * sample_ratio)
                sample_indices = random.sample(range(len(alignment_matrix)), sample_size)
                alignment_matrix_sample = alignment_matrix[sample_indices]
            else:
                alignment_matrix_sample = alignment_matrix
            
            count_matrix = torch.zeros(alignment_matrix_sample.size(1), tokenizer.vocab_size)
            for i in tqdm(range(alignment_matrix_sample.size(1))):
                count_matrix[i] = torch.bincount(alignment_matrix_sample[:,i], minlength=tokenizer.vocab_size)
            
            count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
            count_matrix = torch.log_softmax(count_matrix, dim=-1)
            aln_modify_logits = (1-alpha) * logits[aln_start: aln_end, :] + alpha * count_matrix
            logits = torch.cat([logits[:aln_start], aln_modify_logits, logits[aln_end:]], dim=0)

    if struc_seq_aln_file is not None and aa_seq_aln_file is None:
        print(">>> Using structure sequence alignment matrix...")
        alignment_dict = read_multi_fasta(struc_seq_aln_file)
        alignment_matrix = count_matrix_from_structure_alignment(tokenizer, alignment_dict)
        if alignment_matrix is not None:
            count_matrix = torch.zeros(alignment_matrix.size(1), tokenizer.vocab_size)
            for i in tqdm(range(alignment_matrix.size(1))):
                count_matrix[i] = torch.bincount(alignment_matrix[:,i], minlength=tokenizer.vocab_size)
            count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
            count_matrix = torch.log_softmax(count_matrix, dim=-1)
            logits = (1-alpha) * logits + alpha * count_matrix
            
    if aa_seq_aln_file is not None and struc_seq_aln_file is not None:
        print(">>> Using both residue and structure sequence alignment matrix...")
        plm_logits = logits.clone()
        
        alignment_dict = read_multi_fasta(struc_seq_aln_file)
        structure_alignment_matrix = count_matrix_from_structure_alignment(tokenizer, alignment_dict)
        if structure_alignment_matrix is not None:
            count_matrix = torch.zeros(structure_alignment_matrix.size(1), tokenizer.vocab_size)
            for i in tqdm(range(structure_alignment_matrix.size(1))):
                count_matrix[i] = torch.bincount(structure_alignment_matrix[:,i], minlength=tokenizer.vocab_size)
            
            count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
            count_matrix = torch.log_softmax(count_matrix, dim=-1)
            logits = (1-alpha) * plm_logits + alpha * count_matrix
        
        alignment_dict = read_multi_fasta(aa_seq_aln_file)
        residue_alignment_matrix, aln_start, aln_end = count_matrix_from_residue_alignment(tokenizer, alignment_dict)
        count_matrix = torch.zeros(residue_alignment_matrix.size(1), tokenizer.vocab_size)
        for i in tqdm(range(residue_alignment_matrix.size(1))):
            count_matrix[i] = torch.bincount(residue_alignment_matrix[:,i], minlength=tokenizer.vocab_size)
        
        count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(device)
        count_matrix = torch.log_softmax(count_matrix, dim=-1)
        aln_modify_logits = (1-alpha) * logits[aln_start: aln_end, :] + alpha * count_matrix
        logits = torch.cat([plm_logits[:aln_start], aln_modify_logits, plm_logits[aln_end:]], dim=0)
        
    mutants = mutant_df["mutant"].tolist()
    scores = []
    vocab = tokenizer.get_vocab()
    print(">>> Scoring mutants...")
    for mutant in tqdm(mutants):
        pred_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            assert sequence[idx] == wt, f"Wild type mismatch: {sequence[idx]} != {wt}, idx {idx}"
            score = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            pred_score += score.item()
        scores.append(pred_score)

    return scores
    


def read_names(fasta_dir):
    files = Path(fasta_dir).glob("*.fasta")
    names = [file.stem for file in files]
    return names
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default=["AI4Protein/ProSST-2048"], nargs="+", help="Model name",)
    parser.add_argument("--model_out_name", type=str, default=["ProtREM"], nargs="+", help="Output model name",)
    
    # data directories
    parser.add_argument("--base_dir", type=str, default=None, help="Base directory containing all data",)
    parser.add_argument("--aa_seq_dir", type=str, default=None, help="Directory containing FASTA files of residue sequences",)
    parser.add_argument("--struc_seq_dir", type=str, default=None, help="Directory containing FASTA files of structure sequences",)
    parser.add_argument("--mutant_dir", type=str, default=None, help="Directory containing CSV files with mutants",)
    
    # retrieval and logits mode
    parser.add_argument("--logit_mode", type=str, default="aa_seq_aln", choices=["aa_seq_aln", "struc_seq_aln", "aa_seq_aln+struc_seq_aln", "struc_seq_aln+aa_seq_aln"], help="Mode to retrieve data",)
    parser.add_argument("--alpha", type=float, default=0.8, help="Alpha value for combining logits",)
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use",)
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Ratio of samples to use",)
    parser.add_argument("--sample_times", type=int, default=1, help="Number of times to sample",)
    parser.add_argument("--aa_seq_aln_dir", type=str, default=None, help="Directory containing a2m files of residue alignments",)
    parser.add_argument("--struc_seq_aln_dir", type=str, default=None, help="Directory containing fasta files of foldseek structure alignments",)
    
    # output directory
    parser.add_argument("--output_scores_folder", default=None, help="Directory to save scores")
    args = parser.parse_args()

    print("Scoring proteins...")
    os.makedirs(args.output_scores_folder, exist_ok=True)
    os.makedirs(f"{args.output_scores_folder}/scores", exist_ok=True)
    if args.base_dir:
        if args.aa_seq_dir is None:
            args.aa_seq_dir = f"{args.base_dir}/aa_seq"
        else:
            args.aa_seq_dir = f"{args.base_dir}/{args.aa_seq_dir}"
            
        if args.struc_seq_dir is None:
            args.struc_seq_dir = f"{args.base_dir}/struc_seq"
        else:
            args.struc_seq_dir = f"{args.base_dir}/{args.struc_seq_dir}"
        
        if args.mutant_dir is None:
            args.mutant_dir = f"{args.base_dir}/substitutions"
        else:
            args.mutant_dir = f"{args.base_dir}/{args.mutant_dir}"
        
        if args.aa_seq_aln_dir is None:
            args.aa_seq_aln_dir = f"{args.base_dir}/aa_seq_aln_a2m"
        else:
            args.aa_seq_aln_dir = f"{args.base_dir}/{args.aa_seq_aln_dir}"
            
        if args.struc_seq_aln_dir is None:
            args.struc_seq_aln_dir = f"{args.base_dir}/struc_seq_aln_foldseek"
        else:
            args.struc_seq_aln_dir = f"{args.base_dir}/{args.struc_seq_aln_dir}"
            
            
    protein_names = sorted(read_names(args.aa_seq_dir))
    corrs = []
    print(protein_names)
    print(f">>> total proteins: {len(protein_names)}")
    print("=====================================")
    
    for model_idx, model_name in enumerate(args.model_name):
        print(f">>> Loading model {model_name}...")
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, trust_remote_code=True
        )
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        for idx, protein_name in enumerate(protein_names):
            print(f">>> Scoring {protein_name}, current {idx+1}/{len(protein_names)}...")
            # load data
            residue_fasta = f"{args.aa_seq_dir}/{protein_name}.fasta"
            structure_fasta = f"{args.struc_seq_dir}/{model_name.split('-')[-1]}/{protein_name}.fasta"
            mutant_file = f"{args.mutant_dir}/{protein_name}.csv"
            if args.logit_mode is not None:
                if "aa_seq_aln" in args.logit_mode:
                    if os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.a2m"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.a2m"
                    elif os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.a3m"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.a3m"
                    elif os.path.exists(f"{args.aa_seq_aln_dir}/{protein_name}.fasta"):
                        aa_seq_aln_file = f"{args.aa_seq_aln_dir}/{protein_name}.fasta"
                else:
                    aa_seq_aln_file = None
                
                if "struc_seq_aln" in args.logit_mode:
                    struc_seq_aln_file = f"{args.struc_seq_aln_dir}/{protein_name}.fasta"
                else:
                    struc_seq_aln_file = None
            else:
                aa_seq_aln_file = None
                struc_seq_aln_file = None
                    
            if os.path.exists(f"{args.output_scores_folder}/scores/{protein_name}.csv"):
                mutant_file = f"{args.output_scores_folder}/scores/{protein_name}.csv"
            mutant_df = pd.read_csv(mutant_file)
            
            if args.model_out_name:
                model_out_name = args.model_out_name[model_idx]
            else:
                model_out_name = model_name.split("/")[-1]
                
            if model_out_name not in mutant_df.columns:
                scores = score_protein(
                        model=model,
                        tokenizer=tokenizer,
                        residue_fasta=residue_fasta,
                        structure_fasta=structure_fasta,
                        mutant_df=mutant_df,
                        alpha=args.alpha,
                        aa_seq_aln_file=aa_seq_aln_file,
                        struc_seq_aln_file=struc_seq_aln_file,
                        sample_size=args.sample_size,
                        sample_ratio=args.sample_ratio,
                        sample_times=args.sample_times,
                    )
                mutant_df[model_out_name] = scores
        
            corr = spearmanr(mutant_df["DMS_score"], mutant_df[model_out_name]).correlation
            corrs.append(corr)
            print(f">>> {model_out_name} on {protein_name}: {corr}")
            print("="*50)
            mutant_df.to_csv(f"{args.output_scores_folder}/scores/{protein_name}.csv", index=False)
        
        print(f"====== {model_out_name} average correlation performance: {sum(corrs)/len(corrs)} ======")
        summary_df_path = f"{args.output_scores_folder}/summary_performance.csv"
        if os.path.exists(summary_df_path):
            summary_df = pd.read_csv(summary_df_path)
            summary_df[model_out_name] = corrs
        else:
            summary_df = pd.DataFrame({'protein': protein_names, model_out_name: corrs})
        summary_df.to_csv(f"{args.output_scores_folder}/summary_performance.csv", index=False)