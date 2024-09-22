import argparse
import json
import warnings
import torch
import os
import sys
import yaml
import numpy as np
import pandas as pd
from torch import nn
from torch_geometric.loader import DataLoader
from numpy import nan
from typing import *
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import logging
from src.models import PLM_model, GNN_model
from kermut.data import build_mutant_dataset
from src.utils.utils import param_num

# set path
current_dir = os.getcwd()
sys.path.append(current_dir)
# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

amino_acids_type = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

def label_row(rows, sequence, token_probs, offset_idx=1):
    s = []
    sep = ";"
    if ":" in rows:
        sep = ":"
    for row in rows.split(sep):
        if row.lower() == "wt":
            s.append(0)
            continue
        try:
            wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
        except:
            print(f"row: {row}, sequence: {sequence}")
            raise ValueError
        assert sequence[idx] == wt, f"The {row}, {sequence[idx]}"
        wt_encoded, mt_encoded = amino_acids_type.index(wt), amino_acids_type.index(mt)
        score = token_probs[idx, mt_encoded] - token_probs[idx, wt_encoded]
        score = score.item()
        s.append(score)
        
    return sum(s)


def predict(args, plm_model, gnn_model, loader):
    gnn_model.eval()
    softmax = nn.Softmax()
    result_dict = {"name": [], "count": [], args.score_name: []}

    with torch.no_grad():
        for data in loader:
            protein_name = data.protein_name[0]
            graph_data = plm_model(data)
            out, _ = gnn_model(graph_data)
            seq = "".join([amino_acids_type[i] for i in data.y])
            out = torch.log(softmax(out[:, :20]) + 1e-9)
            
            # check the mutant file
            mutant_file_tsv = os.path.join(args.mutant_dataset_dir, "DATASET", protein_name, f"{protein_name}.tsv")
            mutant_file_csv = os.path.join(args.mutant_dataset_dir, "DATASET", protein_name, f"{protein_name}.csv")
            if os.path.exists(mutant_file_tsv):
                mutant_df = pd.read_table(mutant_file_tsv)
            elif os.path.exists(mutant_file_csv):
                mutant_df = pd.read_csv(mutant_file_csv)
            else:
                raise ValueError(f"Invalid file: {mutant_file_tsv} or {mutant_file_csv}")
            
            # check the offset
            if protein_name == "A0A140D2T1_ZIKV_Sourisseau_2019":
                offset = 291
            else:
                offset = 1
            
            # label the mutant
            mutant_df[args.score_name] = mutant_df[args.mutant_pos_col].apply(
                lambda x: label_row(x, seq, out.cpu().numpy(), offset)
            )
            result_file = os.path.join(args.output_scores_folder, protein_name + ".csv")
            if not os.path.exists(result_file):
                mutant_df.to_csv(result_file, index=False)
                
            result = pd.read_csv(result_file)
            result[args.score_name] = mutant_df[args.score_name]
            result.to_csv(result_file, index=False)
            
            # save the spearmanr score
            result_dict['count'].append(len(result))
            result_dict['name'].append(protein_name)
            spearmanr_score = spearmanr(result[args.mutant_score_col], result[args.score_name]).correlation
            if spearmanr_score is nan:
                spearmanr_score = 0
            result_dict[args.score_name].append(spearmanr_score)
            
            print(f">>> {protein_name}: {spearmanr_score}; mutant_num: {len(result)}")
    
    if args.score_info is not None:
        if os.path.exists(args.score_info):
            total_result = pd.read_csv(args.score_info)
            total_result[args.score_name] = result_dict[args.score_name]
            total_result.to_csv(args.score_info, index=False)
        else:
            pd.DataFrame(result_dict).to_csv(args.score_info, index=False)
            
    print(f">>> {args.score_name} average spearmanr: {np.mean(result_dict[args.score_name])}\n")

def ensemble(args):
    print("----------------- Ensemble -----------------")
    result_files = os.listdir(args.output_scores_folder)
    sp_scores = []
    for file in tqdm(result_files):
        result_file = os.path.join(args.output_scores_folder, file)
        result_df = pd.read_csv(result_file)
        models_pred = [result_df[col].to_list() for col in result_df.columns if col.startswith("ProtSSN")]
        ensemble_pred = np.mean(models_pred, axis=0)
        result_df["ProtSSN_ensemble"] = ensemble_pred
        result_df.to_csv(result_file, index=False)
        sp_score = spearmanr(result_df[args.mutant_score_col], result_df["ProtSSN_ensemble"]).correlation
        sp_scores.append(sp_score)
    print(">>> Ensemble spearmanr: ", np.mean(sp_scores))

def prepare(args, dataset_name, k, h):
    # for build dataset
    args.mutant_name = f"{dataset_name}_k{k}"
    mutant_dataset = build_mutant_dataset(args)
    protein_names = mutant_dataset.protein_names
    print(f">>> Protein names: {protein_names}")
    mutant_loader = DataLoader(mutant_dataset, batch_size=1, shuffle=False)
    print(f">>> Number of proteins: {len(mutant_dataset)}")
    gnn_model = GNN_model(args)
    print(f">>> k{k}_h{h} {param_num(gnn_model)}")
    gnn_model_path = os.path.join(args.gnn_model_dir, f"protssn_k{k}_h{h}.pt")
    gnn_model.load_state_dict(torch.load(gnn_model_path))
    return args, mutant_loader, gnn_model

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnn", type=str, default="egnn", help="gat, gcn, or egnn")
    parser.add_argument("--gnn_config", type=str, default="src/config/egnn.yaml", help="gnn config")
    parser.add_argument("--gnn_model_dir", type=str, default="model/", help="test model name")
    parser.add_argument("--gnn_model_name", type=str, default=None, nargs="+", help="test model name")
    
    parser.add_argument("--plm", type=str, default="facebook/esm2_t33_650M_UR50D", help="esm param number")
    parser.add_argument("--use_ensemble", action="store_true", help="use ensemble model")
    
    # dataset 
    parser.add_argument("--mutant_dataset_dir", type=str, default="data/evaluation", help="mutation dataset")
    parser.add_argument("--mutant_name", type=str, default=None, help="name of mutation dataset")
    parser.add_argument("--mutant_pos_col", type=str, default="mutant", help="mutation column name")
    parser.add_argument("--mutant_score_col", type=str, default="DMS_score", help="the model output score column name")
    
    parser.add_argument("--score_info", type=str, default=None, help="the model output spearmanr score file")
    parser.add_argument("--output_scores_folder", type=str, default="result/", help="the result output path")
    parser.add_argument("--repo_path", type=str, default=None, help="Path to ProteinGym repo")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_parser()    
    args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
    if args.repo_path is None: args.repo_path = os.path.dirname(os.path.dirname(os.getcwd()))
    plm_model = PLM_model(args)
    args.plm_hidden_size = plm_model.model.config.hidden_size
    dataset_name = args.mutant_dataset_dir.split("/")[-1]
    os.makedirs(args.output_scores_folder, exist_ok=True)
    
    for gnn in args.gnn_model_name:
        k, h = gnn.split("_")
        k, h = int(k[1:]), int(h[1:])
        print(f"--------------- ProtSSN k{k}_h{h} ---------------")
        assert k in [10, 20, 30], f"Invalid k: {k}"
        assert h in [512, 768, 1280], f"Invalid h: {h}"
        args.gnn_config["hidden_channels"] = h
        args.c_alpha_max_neighbors = k
        args.score_name = f"ProtSSN_k{k}_h{h}"
        args, mutant_loader, gnn_model = prepare(args, dataset_name, k, h)
        predict(args=args, plm_model=plm_model, gnn_model=gnn_model, loader=mutant_loader)
    
    if args.use_ensemble:
        ensemble(args)
