import torch
import random
import gc
import torch.nn as nn
from torch_geometric.data import Batch, Dataset
from transformers import AutoTokenizer, EsmModel
from typing import *
from src.module.egnn.network import EGNN
from src.module.gcn.network import GCN
from src.module.gat.network import GAT

class PLM_model(nn.Module):
    possible_amino_acids = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
        'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
    one_letter = {
        'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q',
        'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',
        'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',
        'GLY':'G', 'PRO':'P', 'CYS':'C'
        }
    
    def __init__(self, args):
        super().__init__()
        # load global config
        self.args = args
        
        # esm on the first cuda
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.plm)
        self.model = EsmModel.from_pretrained(self.args.plm).cuda()
        
        
    def forward(self, batch):
        with torch.no_grad():
            if not isinstance(batch, List):
                batch = [batch]
            # get the target sequence
            one_hot_seqs = [list(elem.x[:,:20].argmax(1)) for elem in batch]
            muted_res_seqs = ["".join([self.one_letter[self.possible_amino_acids[idx]] for idx in seq_idx]) for seq_idx in one_hot_seqs]
            one_hot_truth_seqs = [elem.y for elem in batch]
            truth_res_seqs = ["".join([self.one_letter[self.possible_amino_acids[idx]] for idx in seq_idx]) for seq_idx in one_hot_truth_seqs]
            
            if not hasattr(self.args, "noise_type"):
                input_seqs = truth_res_seqs
            elif self.args.noise_type == 'mask':
                input_seqs = self._mask_input_sequence(truth_res_seqs)
            elif self.args.noise_type == 'mut':
                input_seqs = muted_res_seqs
            else:
                raise ValueError(f"No implement of {self.args.noise_type}")

            batch_graph = self._nlp_inference(input_seqs, batch)
        return batch_graph
        
    @torch.no_grad()
    def _mask_input_sequence(self, truth_res_seqs):
        input_seqs = []
        self.mask_ratio = self.args.noise_ratio
        for truth_seq in truth_res_seqs:
            masked_seq = ""
            for truth_token in truth_seq:
                pattern = torch.multinomial(torch.tensor([1 - self.args.noise_ratio, 
                                                          self.mask_ratio*0.8, 
                                                          self.mask_ratio*0.1, 
                                                          self.mask_ratio*0.1]), 
                                            num_samples=1,
                                            replacement=True)
                # 80% of the time, we replace masked input tokens with mask_token ([MASK])
                if pattern == 1:
                    masked_seq += '<mask>'
                # 10% of the time, we replace masked input tokens with random word
                elif pattern == 2:
                    masked_seq += random.sample(list(self.one_letter.values()), 1)[0]
                # The rest of the time (10% of the time) we keep the masked input tokens unchanged
                else:
                    masked_seq += truth_token
            input_seqs.append(masked_seq)
        return input_seqs
    
    
    @torch.no_grad()
    def _nlp_inference(self, input_seqs, batch):    
        inputs = self.tokenizer(input_seqs, return_tensors="pt", padding=True).to("cuda:0")
        batch_lens = (inputs["attention_mask"] == 1).sum(1) - 2
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        for idx, (hidden_state, seq_len) in enumerate(zip(last_hidden_states, batch_lens)):
            batch[idx].esm_rep = hidden_state[1: 1+seq_len]
            del batch[idx].seq
                
        # move to the GNN devices
        batch = [elem.cuda() for elem in batch]
        batch_graph = Batch.from_data_list(batch)
        gc.collect()
        torch.cuda.empty_cache()
        return batch_graph



class GNN_model(nn.Module):    
    def __init__(self, args):
        super().__init__()
        # load graph network config which usually not change
        self.gnn_config = args.gnn_config
        # load global config
        self.args = args
        
        # calculate input dim according to the input feature
        self.out_dim = 20
        self.input_dim = self.args.plm_hidden_size
        
        # gnn on the rest cudas
        if "egnn" == self.args.gnn:
            self.GNN_model = EGNN(self.gnn_config, self.args, self.input_dim, self.out_dim)
        elif "gcn" == self.args.gnn:
            self.GNN_model = GCN(self.gnn_config, self.input_dim, self.out_dim)
        elif "gat" == self.args.gnn:
            self.GNN_model = GAT(self.gnn_config,self.input_dim, self.out_dim)
        else:
            raise KeyError(f"No implement of {self.opt['gnn']}")
        self.GNN_model = self.GNN_model.cuda()

    def forward(self, batch_graph):
        gnn_out = self.GNN_model(batch_graph)
        return gnn_out
    
