import os
import sys
import csv
import pprint
import random
import pickle

import numpy as np

from tqdm import tqdm

import torch
from torch.nn import functional as F

#import torch_geometric.data     # Pre-load to bypass the hack from torchdrug
#from torchvision import datasets
from torchdrug import core, utils, data, metrics
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
from s3f import dataset, task, model, gvp


METRICS = ["spearmanr", "pearsonr", "mae", "rmse"]


def evaluate(pred, target):
    pred = pred.float()
    target = target.float()
    metric = {}
    for _metric in METRICS:
        if _metric == "mae":
            score = F.l1_loss(pred, target, reduction="mean")
        elif _metric == "rmse":
            score = F.mse_loss(pred, target, reduction="mean").sqrt()
        elif _metric == "spearmanr":
            score = metrics.spearmanr(pred, target)
        elif _metric == "pearsonr":
            score = metrics.pearsonr(pred, target)
        else:
            raise ValueError("Unknown metric `%s`" % _metric)
        
        metric[_metric] = score

    return metric


def graph_concat(graphs):
    if len(graphs) == 1:
        return graphs[0]
    graph = graphs[0].pack(graphs)
    if isinstance(graph, data.Protein):
        # residue graph
        _graph = data.Protein(edge_list=graph.edge_list, atom_type=graph.atom_type, bond_type=graph.bond_type, 
                                residue_type=graph.residue_type, atom_name=graph.atom_name, atom2residue=graph.atom2residue, 
                                residue_feature=graph.residue_feature, b_factor=graph.b_factor, bond_feature=None,
                                node_position=graph.node_position, num_node=graph.num_atom, num_residue=graph.num_residue,
        )
    else:
        # surface graph
        _graph = data.Graph(edge_list=graph.edge_list, num_node=graph.num_node, num_relation=1)
        with _graph.node():
            _graph.node_position = graph.node_position
            _graph.node_feature = graph.node_feature
            _graph.normals = graph.normals
    return _graph


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
    

def predict(cfg, task, dataset):
    dataloader = data.DataLoader(dataset, cfg.batch_size, shuffle=False, num_workers=0)
    device = torch.device(cfg.gpus[0])
    task = task.cuda(device)
    task.eval()
    seq_prob = []
    for batch in tqdm(dataloader):
        batch = utils.cuda(batch, device=device)
        with torch.no_grad():
            prob, sizes = task.inference(batch)
        cum_sizes = sizes.cumsum(dim=0)
        for i in range(len(sizes)):
            seq_prob.append(prob[cum_sizes[i]-sizes[i]:cum_sizes[i]])
    return seq_prob


def get_prob(seq_prob, mutations, offsets):
    i = 0
    preds = []
    targets = []
    last_sites = None
    for j, item in tqdm(enumerate(mutations)):
        sites, muts, target = item
        if j > 0 and sites != last_sites:
            i += 1

        node_index = torch.tensor(sites, dtype=torch.long)
        offset = offsets[i]
        node_index = node_index - offset
        mt_target = [data.Protein.residue_symbol2id.get(mut[-1], -1) for mut in muts]
        wt_target = [data.Protein.residue_symbol2id.get(mut[0], -1) for mut in muts]
        log_prob = torch.log_softmax(seq_prob[i], dim=-1)
        mt_log_prob = log_prob[node_index, mt_target]
        wt_log_prob = log_prob[node_index, wt_target]
        log_prob = mt_log_prob - wt_log_prob
        score = log_prob.sum(dim=0)

        preds.append(score)
        targets.append(target)
        last_sites = sites

    pred = torch.stack(preds)
    target = torch.tensor(targets).cuda()
    return pred, target
        

def load_dataset(csv_file, protein, task_instance):
    with open(csv_file, "r") as fin:
        reader = csv.reader(fin)
        fields = next(reader)
        mutations = []
        targets = []
        for i, values in enumerate(reader):
            for field, value in zip(fields, values):
                if field == "mutant":
                    mutations.append(value.split(":"))
                elif field == "DMS_score":
                    value = utils.literal_eval(value)
                    targets.append(value)
    
    def mutation_site(x):
        return [int(y[1:-1])-1 for y in x]

    mutations = [(tuple(mutation_site(mut)), mut, tar) for mut, tar in zip(mutations, targets)]
    mutations = sorted(mutations)
    sequences = []
    offsets = []
    for i, mut in enumerate(mutations):
        if i > 0 and mut[0] == mutations[i-1][0]:
            continue
        masked_seq = protein.clone()
        _mutation_site = mut[0]
        node_index = torch.tensor(_mutation_site, dtype=torch.long)

        # truncate long sequences and those only with substructures
        if os.path.basename(csv_file) == "POLG_HCVJF_Qi_2014.csv":
            start, end = 1981, 2225
        elif os.path.basename(csv_file) == "A0A140D2T1_ZIKV_Sourisseau_2019.csv":
            start, end = 290, 794
        elif os.path.basename(csv_file) == "B2L11_HUMAN_Dutta_2010_binding-Mcl-1.csv":
            start, end = 119, 197       # keep high plddt part
        elif masked_seq.num_residue > 1022:
            seq_len = masked_seq.num_residue
            start, end = get_optimal_window(mutation_position_relative=mut[0][0], seq_len_wo_special=seq_len, model_window=1022)
        else:
            start, end = 0, masked_seq.num_residue
        node_index = node_index - start
        residue_mask = torch.zeros((masked_seq.num_residue, ), dtype=torch.bool)
        residue_mask[start:end] = 1
        masked_seq = masked_seq.subresidue(residue_mask)
        with masked_seq.graph():
            masked_seq.start = torch.as_tensor(start)
            masked_seq.end = torch.as_tensor(end)
        offsets.append(start)
        
        mask_id = task_instance.model.sequence_model.alphabet.get_idx("<mask>")
        with masked_seq.residue():
            masked_seq.residue_feature[node_index] = 0
            masked_seq.residue_type[node_index] = mask_id
        sequences.append(masked_seq)

    return sequences, mutations, offsets