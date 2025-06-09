
import os, sys, time, re, random, pickle, copy, gzip, io, configparser, math, shutil, pathlib, tempfile, hashlib, argparse, json, inspect, urllib, collections, subprocess, requests, platform, multiprocessing
from typing import Dict, Union, Optional, Tuple
import multiprocessing as mp
from tqdm.auto import trange, tqdm
import numpy as np
import pandas as pd
from Bio import SeqIO
from datetime import datetime
from typing import Optional, Union, List
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from scipy.special import softmax
from scipy.spatial.distance import squareform, pdist, cdist
from os.path import exists, join, getsize, isfile, isdir, abspath, basename, realpath, dirname
from os import listdir
from tabulate import tabulate
from argparse import Namespace
from modelgenerator.structure_tokenizer.models import EquiformerEncoderLightning, ESMFoldDecoderLightning
from modelgenerator.structure_tokenizer.datasets.protein_dataset import ProteinDataset

SCRIPT_PATH = str(pathlib.Path(__file__).parent.resolve())

class AIDO_Structure_Tokenizer:
    def __init__(self, codebook_path=None, device='cuda'):
        
        if codebook_path is None:
            codebook_path = join(SCRIPT_PATH, "codebook.pt")
        
        self.codebook = torch.load(codebook_path, 'cpu', weights_only=True) # [512, 384]
        
        self.encoder = EquiformerEncoderLightning(pretrained_model_name_or_path="genbio-ai/AIDO.StructureEncoder").eval().to(device)
    
    def to(self, device):
        self.encoder = self.encoder.to(device) if self.encoder is not None else None
        return self
    
    def encode(self, aatype, atom_positions, atom_mask, get_embedding=False):
        """
        aatype: [L]
        atom_positions: [L, 37, 3]
        atom_mask: [L, 37]
        """
        assert self.encoder is not None, "Encoder is not loaded"
        
        assert aatype.ndim == 1
        assert atom_positions.ndim == 3
        assert atom_mask.ndim == 2
        device = next(iter(self.encoder.parameters())).device
        
        residue_index = torch.arange(1, aatype.shape[0]+1)
        aatype = torch.from_numpy(aatype) if isinstance(aatype, np.ndarray) else aatype
        
        if isinstance(atom_positions, np.ndarray):
            atom_positions = atom_positions.copy()
            atom_positions[ atom_mask == 0 ] = np.nan
            atom_mask = torch.from_numpy(atom_mask)
            atom_positions = torch.from_numpy(atom_positions).float()
        elif isinstance(atom_positions, torch.Tensor):
            atom_positions = atom_positions.clone()
            atom_positions[ atom_mask == 0 ] = torch.nan
            atom_positions = atom_positions.float()
        else:
            raise RuntimeError(f"Expect atom_positions be np.ndarray or torch.Tensor, but got {type(atom_positions)}")
        
        batch = [{
            'id':            'xxx',
            'entity_id':     'xxx',
            'chain_id':      'xxx',
            'resolution':     torch.tensor(3.0),
            'aatype':         aatype,
            'atom_positions': atom_positions,
            'atom_mask':      atom_mask,
            'residue_index':  residue_index
        }]
        batch[0] = { k:(v.to(device) if torch.is_tensor(v) else v) for k,v in batch[0].items() }
        batch = ProteinDataset.collate_fn(batch)
        
        with torch.no_grad():
            tokens = self.encoder.predict_step(batch, batch_idx=0, dataloader_idx=0)['xxx_xxx_xxx']['struct_tokens'].cpu()
        
        if get_embedding:
            emb = F.embedding(tokens, self.codebook)
            return (emb, tokens)
        else:
            return tokens

def load_fasta(seqFn, rem_tVersion=False, load_annotation=False, full_line_as_id=False):
    """
    seqFn               -- Fasta file or input handle (with readline implementation)
    rem_tVersion        -- Remove version information. ENST000000022311.2 => ENST000000022311
    load_annotation     -- Load sequence annotation
    full_line_as_id     -- Use the full head line (starts with >) as sequence ID. Can not be specified simutanouly with load_annotation

    Return:
        {tid1: seq1, ...} if load_annotation==False
        {tid1: seq1, ...},{tid1: annot1, ...} if load_annotation==True
    """
    if load_annotation and full_line_as_id:
        raise RuntimeError("Error: load_annotation and full_line_as_id can not be specified simutanouly")
    if rem_tVersion and full_line_as_id:
        raise RuntimeError("Error: rem_tVersion and full_line_as_id can not be specified simutanouly")

    fasta = {}
    annotation = {}
    cur_tid = ''
    cur_seq = ''
    
    if isinstance(seqFn, str):
        IN = open(seqFn)
    elif hasattr(seqFn, 'readline'):
        IN = seqFn
    else:
        raise RuntimeError(f"Expected seqFn: {type(seqFn)}")
    for line in IN:
        if line[0] == '>':
            if cur_seq != '':
                fasta[cur_tid] = re.sub(r"\s", "", cur_seq)
                cur_seq = ''
            data = line[1:-1].split(None, 1)
            cur_tid = line[1:-1] if full_line_as_id else data[0]
            annotation[cur_tid] = data[1] if len(data)==2 else ""
            if rem_tVersion and '.' in cur_tid: 
                cur_tid = ".".join(cur_tid.split(".")[:-1])
        elif cur_tid != '':
            cur_seq += line.rstrip()
    
    if isinstance(seqFn, str):
        IN.close()

    if cur_seq != '':
        fasta[cur_tid] = re.sub(r"\s", "", cur_seq)
    
    if load_annotation:
        return fasta, annotation
    else:
        return fasta

def load_msa_txt(file_or_stream, load_id=False, load_annot=False, sort=False):
    """
    Read msa txt file
    
    Parmeters
    --------------
    file_or_stream: file or stream to read (with read method)
    load_id: read identity and return
    
    Return
    --------------
    msa: list of msa sequences, the first sequence in msa is the query sequence
    id_arr: Identity of msa sequences
    annotations: Annotations of msa sequences
    """
    msa = []
    id_arr = []
    annotations = []
    
    if hasattr(file_or_stream, 'read'):
        lines = file_or_stream.read().strip().split('\n')
    elif file_or_stream.endswith('.gz'):
        with gzip.open(file_or_stream) as IN:
            lines = IN.read().decode().strip().split('\n')
    else:
        with open(file_or_stream) as IN:
            lines = IN.read().strip().split('\n')
    
    for idx,line in enumerate(lines):
        data = line.strip().split()
        if idx == 0:
            assert len(data) == 1, f"Expect 1 element for the 1st line, but got {data} in {file_or_stream}"
            q_seq = data[0]
        else:
            if len(data) >= 2:
                id_arr.append( float(data[1]) )
            else:
                assert len(q_seq) == len(data[0])
                id_ = round(np.mean([ r1==r2 for r1,r2 in zip(q_seq, data[0]) ]), 3)
                id_arr.append(id_)
            msa.append( data[0] )
            if len(data) >= 3:
                annot = " ".join(data[2:])
                annotations.append( annot )
            else:
                annotations.append(None)
    
    id_arr = np.array(id_arr, dtype=np.float64)
    if sort:
        id_order = np.argsort(id_arr)[::-1]
        msa      = [ msa[i] for i in id_order ]
        id_arr   = id_arr[id_order]
        annotations = [ annotations[i] for i in id_order ]
    msa = [q_seq] + msa
    
    outputs = [ msa ]
    if load_id:
        outputs.append( id_arr )
    if load_annot:
        outputs.append( annotations )
    if len(outputs) == 1:
        return outputs[0]
    return outputs

def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, num_tokens: int = None, mode: str = "max", seed: int = None) -> List[Tuple[str, str]]:
    """
    Greedy select msa sequences according to hamming distance. 
    Two modes:
    - by num_seqs: select #seqs from all MSA sequences
    - by num_tokens: select a MSA sequences subset contains #tokens (excluded gap) from all MSA sequences
    """
    msa = msa.copy()
    if seed is not None:
        random.Random(seed).shuffle(msa)
    
    assert mode in ("max", "min")
    assert (num_seqs is None and num_tokens is not None) or (num_seqs is not None and num_tokens is None)
    if num_seqs is not None and len(msa) <= num_seqs:
        return msa
    if num_tokens is not None and sum([ len(s)-s.count('-') for s in msa ]) <= num_tokens:
        return msa
    
    array = np.array([list(seq) for seq in msa], dtype=np.bytes_).view(np.uint8)
    
    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    selected_msa = []
    for _ in range(len(msa)-1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
        selected_msa.append(msa[index])
        if num_seqs is not None and len(indices) >= num_seqs:
            break
        if num_tokens is not None and sum([ len(s)-s.count('-') for s in selected_msa ]) >= num_tokens:
            break
    indices = sorted(indices)
    return [msa[idx] for idx in indices]

def tokenize(q_seq, msa, tokenizer, max_context=12800):
    """
    Tokenizes the input sequence and optionally additional sequences for multiple sequence alignment (MSA).
    
    Args:
        q_seq (str): The query sequence to be tokenized.
        msa (list or None): A list of sequences for multiple sequence alignment. If None, no MSA sequences are added.
        tokenizer (object): The tokenizer object used to encode the sequences.
        max_context (int, optional): The maximum number of tokens to consider in the context. Defaults to 12800.
    
    Returns:
        tuple: A tuple containing:
            - tokens (np.ndarray): The tokenized sequences.
            - pos_encoding (np.ndarray): The positional encoding for the tokens.
    """
    len_seq = len(q_seq)
    tokens = tokenizer.encode(q_seq, add_eos=False).tolist()
    num_seq = 1
    
    for msa_seq in msa:
        assert len(msa_seq) == len_seq, f"len(msa_seq)={len(msa_seq)}, len_seq={len_seq}"
        tokens.extend(tokenizer.encode(msa_seq, add_eos=False).tolist())
        num_seq += 1
    
    pos_encoding = np.stack([ np.tile(np.arange(len_seq), num_seq), np.repeat(np.arange(num_seq), len_seq) ])
    
    tokens = np.array(tokens)
    tok_mask = (tokens != tokenizer.TokenToId('-'))
    tokens, pos_encoding = tokens[tok_mask][:max_context], pos_encoding[..., tok_mask][..., :max_context]
    return tokens, pos_encoding

@torch.no_grad()
def get_logits_table_sliding(q_seq, prot, msa, dms_df, model, tokenizer, str_tokenizer, start, sliding_window=768, sliding_step=768, mask_str=False, verbose=False, disable_tqdm=True):
    """
    xxxx
    """
    # model_type = get_model_type(model)
    # assert model_type == 'emb_model_step1'
    assert len(q_seq) == prot.aatype.shape[0], f"len(q_seq)={len(q_seq)}, prot.aatype.shape[0]={prot.aatype.shape[0]}"
    assert q_seq == msa[0]
    
    pd_scores = []
    all_poses = set()
    for mutant in dms_df['mutant'].tolist():
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - start, sub_mutant[-1]
            assert q_seq[idx] == wt
            all_poses.add(idx)
    
    all_poses   = sorted(list(all_poses))
    vocab_size  = model.language_model.embedding.word_embeddings.weight.shape[0] if hasattr(model, 'language_model') else model.config.padded_vocab_size
    logit_table = np.zeros([len(all_poses), vocab_size])
    count_table = np.zeros([len(all_poses)], dtype=np.int64)
    
    is_last_step = False
    for f_start in range(0, len(q_seq), sliding_step):
        if is_last_step:
            break
        if f_start + sliding_window > len(q_seq) and len(q_seq) > sliding_window:
            f_start = len(q_seq) - sliding_window
            is_last_step = True
        
        f_end = min(f_start + sliding_window, len(q_seq))
        
        f_q_seq = q_seq[f_start:f_end]
        
        # f_msa = rag_utils.greedy_select(list(set([ seq[f_start:f_end] for seq in msa[1:] ])), num_seqs=None, num_tokens=12800, seed=0)
        f_msa = greedy_select([ seq[f_start:f_end] for seq in msa[1:] ], num_seqs=None, num_tokens=12800, seed=0)
        f_msa.sort(key=lambda x: x.count('-'))
        
        str_embs, str_toks = str_tokenizer.encode(prot.aatype[f_start:f_end], prot.atom_positions[f_start:f_end], prot.atom_mask[f_start:f_end], get_embedding=True)
        str_embs, str_toks = str_embs.cuda().bfloat16(), str_toks.cuda()
        if mask_str:
            str_embs[:] = 0
            str_toks[:] = 0
        
        tokens, pos_encoding = tokenize(f_q_seq, f_msa, tokenizer, max_context=12800)
        tokens       = torch.from_numpy(tokens).cuda()
        pos_encoding = torch.from_numpy(pos_encoding).cuda()
        
        if verbose:
            tqdm.write(f"{f_start}-{f_end}, SeqL={len(q_seq)}, TokenL={tokens.shape[0]}")
        
        for i,pos in tqdm(enumerate(all_poses), total=len(all_poses), leave=False, dynamic_ncols=True, disable=disable_tqdm):
            if f_start <= pos < f_end:
                masked_tokens = tokens.clone()
                masked_tokens[pos_encoding[0]==pos-f_start] = tokenizer.TokenToId('tMASK')
                lm_output = model.transformer(
                    input_ids=masked_tokens[None],
                    position_ids=pos_encoding[None],
                    full_attention_mask=None,
                    inputs_str_embeds=str_embs[None]
                )
                logits = model.transformer.output_layer(lm_output['last_hidden_state'])[:, 0]  # [seq_len, batch_size, 128]
                logits = logits[:len(q_seq)].squeeze().cpu().float()
                logit_table[i] += logits[pos-f_start].numpy()
                count_table[i] += 1
    
    if np.any(count_table == 0):
        breakpoint()
    logit_table = logit_table / count_table[:, None]
    return (all_poses, logit_table)

def get_scores_from_table(q_seq, logits_table, all_poses, dms_df, tokenizer, start, temp_mt=1.0, temp_wt=1.5):
    """
    Calculate predicted scores for protein mutants based on a given table and compare them with ground truth scores.
    
    Args:
        q_seq (str): The query sequence of the protein.
        table (np.ndarray): A 2D array containing scores for each position and amino acid.
        all_poses (list): A list of positions in the query sequence.
        DMS_id (str): The identifier for the DMS (Deep Mutational Scanning) dataset.
        tokenizer (Tokenizer): A tokenizer object to convert amino acids to indices.
        start (int): The starting index for the positions in the query sequence.
    
    Returns:
        tuple: A tuple containing:
            - pd_scores (list): A list of predicted scores for each mutant.
            - gt_scores (pd.Series): A series of ground truth scores from the DMS dataset.
    """
    table_mt = np.log(softmax(logits_table / temp_mt, axis=-1))
    table_wt = np.log(softmax(logits_table / temp_wt, axis=-1))
    gt_scores = dms_df['DMS_score']
    pd_scores = []
    vocab = tokenizer.get_vocab()
    all_data = []
    for i, mutant in enumerate(dms_df['mutant'].tolist()):
        mutant_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - start, sub_mutant[-1]
            assert wt == q_seq[idx]
            new_idx = all_poses.index(idx)
            assert new_idx >= 0
            pred = table_mt[new_idx, vocab[mt]] - table_wt[new_idx, vocab[wt]]
            mutant_score += pred.item()
        pd_scores.append(mutant_score)
        all_data.append([ mutant, mutant_score, gt_scores[i] ])
    result_df = pd.DataFrame(all_data, columns=['Mutation', 'Pred_Score', 'GT_Score']).round(5)
    return result_df

