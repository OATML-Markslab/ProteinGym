import os
import argparse
import tqdm 

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss

from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK
from sequence_models.pdb_utils import parse_PDB, process_coords

from proteingym.baselines.carp_mif.carp_mif_utils import load_model_and_alphabet

def label_row(rows, sequence, token_probs, alphabet, offset_idx=1):
    rows = rows.split(":")
    score = 0
    for row in rows:
        wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.index(wt), alphabet.index(mt)

        score_obj = token_probs[0, idx, mt_encoded] - token_probs[0, idx, wt_encoded]
        score += score_obj.item()
    return score / len(rows)

def process_batch_mif(prot,pdb_file,tokenizer,device='cuda:0'):
    coords, wt, _ = parse_PDB(pdb_file)
    coords = {
        'N': coords[:, 0],
        'CA': coords[:, 1],
        'C': coords[:, 2]
    }
    dist, omega, theta, phi = process_coords(coords)
    batch = [[prot, torch.tensor(dist, dtype=torch.float),
            torch.tensor(omega, dtype=torch.float),
            torch.tensor(theta, dtype=torch.float), torch.tensor(phi, dtype=torch.float)]]
    input_ids, nodes, edges, connections, edge_mask = tokenizer(batch)
    input_ids = input_ids.to(device)
    nodes = nodes.to(device)
    edges = edges.to(device)
    connections = connections.to(device)
    edge_mask = edge_mask.to(device)
    return input_ids,nodes,edges,connections,edge_mask

def calc_fitness(model, DMS_data, tokenizer, device='cuda:0', model_context_len=1024, mode="masked_marginals", alphabet=PROTEIN_ALPHABET, mutation_col='mutant', target_seq=None, pdb_file=None, model_name=None, offset_idx=1):
    if mode=="pseudo_likelihood":
        prots=np.array(DMS_data['mutated_sequence'])
        loss_fn = CrossEntropyLoss()
        log_proba_list = []
        with torch.no_grad():
            for prot in tqdm.tqdm(prots):
                loss_fn = CrossEntropyLoss()
                if 'carp' in model_name:
                    input_ids = tokenizer([[prot]])[0].to(device)
                    logits = model(input_ids, logits=True)['logits']
                elif 'mif' in model_name:
                    input_ids,nodes,edges,connections,edge_mask = process_batch_mif(prot,pdb_file,tokenizer,device)
                    logits = model(input_ids, nodes, edges, connections, edge_mask, result='logits')
                log_proba = - loss_fn(target=input_ids.view(-1), input=logits.view(-1,logits.size(-1))).detach().cpu().numpy()
                log_proba_list += [log_proba]
    elif mode=="masked_marginals":
        all_token_probs = []
        if 'carp' in model_name:
            input_ids = tokenizer([[target_seq]])[0].to(device)
        elif 'mif' in model_name:
            input_ids,nodes,edges,connections,edge_mask = process_batch_mif(target_seq,pdb_file,tokenizer,device)
        for i in tqdm.tqdm(range(input_ids.size(1))):
            input_ids_masked = input_ids.clone()
            input_ids_masked[0, i] = PROTEIN_ALPHABET.index(MASK)
            with torch.no_grad():
                if 'carp' in model_name:
                    logits = model(input_ids_masked.cuda(), logits=True)["logits"]
                elif 'mif' in model_name:
                    logits = model(input_ids, nodes, edges, connections, edge_mask, result='logits')
                token_probs = torch.log_softmax(logits, dim=-1)
            all_token_probs.append(token_probs[:, i]) 
        token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
        log_proba_list = DMS_data.apply(
            lambda row: label_row(
                row[mutation_col],
                target_seq,
                token_probs,
                PROTEIN_ALPHABET,
                offset_idx
            ),
            axis=1,
        )
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
            print("Issue with mutant: "+str(mutation))
        relative_position = position - start_idx
        assert (from_AA==focus_seq[relative_position]), "Invalid from_AA or mutant position: "+str(mutation)+" from_AA: "+str(from_AA) + " relative pos: "+str(relative_position) + " focus_seq: "+str(focus_seq)
        assert (to_AA in AA_vocab) , "Mutant to_AA is invalid: "+str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)

def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with Tranception.
    """
    parser = argparse.ArgumentParser(description='Tranception scoring')
    parser.add_argument('--model_name', default=None, type=str, help='Name of or path to [CARP,MIF,Evodiff] model')
    parser.add_argument('--model_path', default=None, type=str, help='Location where model parameters should be stored')
    parser.add_argument('--DMS_reference_file_path', default='/home/pn73/Tranception/proteingym/ProteinGym_reference_file_substitutions.csv', type=str, help='Path to reference file')
    parser.add_argument('--DMS_data_folder', default='/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/Tranception_open_source/DMS_files/ProteinGym_substitutions', type=str, help='Path of DMS folder')
    parser.add_argument('--structure_data_folder', default='', type=str, help='Path of structure folder')
    parser.add_argument('--DMS_index', type=int, help='Index of proteins to compute scores for')
    parser.add_argument('--output_scores_folder', default=None, type=str, help='Name of folder to write model scores to')
    parser.add_argument('--indel_mode', action='store_true', help='Whether to score sequences with insertions and deletions')
    parser.add_argument('--fitness_computation_mode', default="masked_marginals", type=str, help='Fitness computtation mode [masked_marginals|pseudo_likelihood]')
    parser.add_argument('--performance_file', default='CARP_performance.csv', type=str, help='Name of folder to write model scores to')
    args = parser.parse_args()

    model, tokenizer = load_model_and_alphabet(args.model_name, args.model_path)
    model.cuda()

    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id=list_DMS[args.DMS_index]
    if not os.path.exists(args.output_scores_folder): os.mkdir(args.output_scores_folder)
    args.output_scores_folder = args.output_scores_folder + os.sep + args.model_name
    if not os.path.exists(args.output_scores_folder): os.mkdir(args.output_scores_folder)
    scoring_filename = args.output_scores_folder+os.sep+DMS_id+'.csv'
    print("Computing scores for: {} with model: {}".format(DMS_id, args.model_name))

    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
    target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
    
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
    DMS_data['mutated_sequence'] = DMS_data['mutant'].apply(lambda x: get_mutated_sequence(target_seq, x)) if not args.indel_mode else DMS_data['mutant']

    if 'mif' in args.model_name:
        pdb_filenames = mapping_protein_seq_DMS["pdb_file"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].split('|') #if sequence is large (eg., BRCA2_HUMAN) the structure is split in several chunks
        pdb_ranges = mapping_protein_seq_DMS["pdb_range"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].split('|')
        model_scores=[]
        for pdb_index, pdb_filename in enumerate(pdb_filenames):
            pdb_file = args.structure_data_folder + os.sep + pdb_filename
            pdb_range = [int(x) for x in pdb_ranges[pdb_index].split("-")]
            target_seq_split = target_seq[pdb_range[0]-1:pdb_range[1]] #pdb_range is 1-indexed
            DMS_data["mutated_position"] = DMS_data['mutant'].apply(lambda x: int(x[1:-1]))
            filtered_DMS_data = DMS_data[(DMS_data["mutated_position"] >= pdb_range[0]) & (DMS_data["mutated_position"] <= pdb_range[1])]
            model_scores.append(calc_fitness(model=model, DMS_data=filtered_DMS_data, tokenizer=tokenizer, mode=args.fitness_computation_mode, target_seq=target_seq_split, pdb_file=pdb_file, model_name=args.model_name, offset_idx=pdb_range[0]))
        model_scores = np.concatenate(model_scores)
    else:
        model_scores = calc_fitness(model=model, DMS_data=DMS_data, tokenizer=tokenizer, mode=args.fitness_computation_mode, target_seq=target_seq, pdb_file=None, model_name=args.model_name)

    DMS_data[args.model_name+'_score']=model_scores
    DMS_data[['mutant',args.model_name+'_score','DMS_score']].to_csv(scoring_filename, index=False)
    spearman, _ = spearmanr(DMS_data[args.model_name+'_score'], DMS_data['DMS_score'])

    if not os.path.exists(args.performance_file) or os.stat(args.performance_file).st_size==0:
        with open(args.performance_file,"w") as performance_file:
            performance_file.write("DMS_id,spearman\n")    
    with open(args.performance_file, "a") as performance_file:
        performance_file.write(",".join([DMS_id,str(spearman)])+"\n")

if __name__ == '__main__':
    main()