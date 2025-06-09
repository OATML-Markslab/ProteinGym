#!/usr/bin/env python

import os, sys, time, re, random, pickle, copy, gzip, io, yaml, logging, configparser, math, shutil, pathlib, tempfile, hashlib, argparse, json, inspect, urllib, collections, subprocess, requests, platform, multiprocessing, importlib, string, code, warnings, concurrent, gc, functools, types, traceback, base64, bz2, ctypes, tarfile, shlex, socket
from queue import PriorityQueue, Queue, deque, LifoQueue
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from os.path import exists, join, getsize, isfile, isdir, abspath, basename, realpath, dirname
from os import listdir
from typing import Dict, Union, Optional, List, Tuple, Mapping
from functools import partial
from datetime import datetime
import numpy  as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm, trange
from tabulate import tabulate
from argparse import Namespace, ArgumentParser

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForMaskedLM
import safetensors
import torch
from scipy.stats import spearmanr
from scipy.special import softmax
import seaborn as sns
import matplotlib.pyplot as plt

from utils import misc
from utils import protein

SCRIPT_PATH = str(pathlib.Path(__file__).parent.resolve())

def main(args):

    #############################
    ## Set HF cache location if provided
    #############################
    
    if args.hf_cache_location:
        os.environ['HF_HOME'] = args.hf_cache_location
        os.environ['HUGGINGFACE_HUB_CACHE'] = args.hf_cache_location

    #############################
    ## Create structure tokenizer
    #############################

    str_tokenizer = misc.AIDO_Structure_Tokenizer(device='cuda:0')

    #############################
    ## Define model and tokenizer
    #############################
    model_name = "genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot"
    tokenizer  = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        cache_dir=args.hf_cache_location if args.hf_cache_location else None
                    )
    model      = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        cache_dir=args.hf_cache_location if args.hf_cache_location else None
                    )
    model      = model.bfloat16().eval().to('cuda:0')

    #############################
    ## Run model and get predicted DMS scores
    #############################

    os.makedirs(args.output_path, exist_ok=True)
    for dms_id in args.dms_ids:
        
        ## Read Sequence
        dms2seq, dms2annot = misc.load_fasta(f"{args.input_data_path}/query.fasta", load_annotation=True)
        q_seq = dms2seq[dms_id]
        start, end = [int(x) for x in dms2annot[dms_id].split('-')]

        ## Read MSA
        msa = misc.load_msa_txt(f"{args.input_data_path}/msa_data/{dms_id}.txt.gz")
        assert q_seq == msa[0]

        ## Read PDB
        with open(f"{args.input_data_path}/struc_data/{dms_id}.pdb") as IN:
            text = IN.read()

        prot = protein.from_pdb_string(text, molecular_type='protein')
        assert prot.seq(True) == q_seq
        
        ## Read DMS Table
        dms_df = pd.read_csv(f'{args.input_data_path}/dms_data/{dms_id}.csv')
    
        ## Inference
        all_poses, logits_table = misc.get_logits_table_sliding(q_seq, prot, msa, dms_df, model, tokenizer, str_tokenizer, start, mask_str=args.mask_str, disable_tqdm=False)
        result_df = misc.get_scores_from_table(q_seq, logits_table, all_poses, dms_df, tokenizer, start, temp_mt=1.0, temp_wt=1.5)
        
        assert np.all(dms_df['mutant'] == result_df['Mutation'])
        assert np.allclose(dms_df['DMS_score'] , result_df['GT_Score'], 1e-05, 1e-05)
        dms_df['AIDO.Protein-RAG-16B-zeroshot'] = result_df['Pred_Score']
        result_df = dms_df
        
        result_df.to_csv(join(args.output_path, f"{dms_id}.csv"), index=False)
        r = round(spearmanr(result_df['AIDO.Protein-RAG-16B-zeroshot'], result_df['DMS_score'])[0], 4)
        print(f"{dms_id}: R={r}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dms_ids", type=str, default=None, nargs="+")
    parser.add_argument("--dms_index", type=int, default=None, help="Index of a single DMS_id in full list of assays (all_dms_ids)")
    parser.add_argument("--input_data_path", type=str, default=f"{SCRIPT_PATH}", help="Input data path",)
    parser.add_argument("--output_path", type=str, default=f"{SCRIPT_PATH}/output", help="Output path",)
    parser.add_argument("--hf_cache_location", type=str, default=None, help="Hugging Face cache directory for downloading models and tokenizers")
    parser.add_argument("--mask-str", action='store_true', help="Mask the structure input")
    args = parser.parse_args()
    all_dms_ids = [n[:-7] for n in os.listdir(f'{args.input_data_path}/msa_data') if n.endswith('.txt.gz')]
    
    if args.dms_ids is None: 
        if args.dms_index is not None: # Compute scores for a single DMS id, indexed by dms_index
            args.dms_ids=[all_dms_ids[args.dms_index]]
        else:
            args.dms_ids=all_dms_ids # Compute scores for all assays in ProteinGym
    main(args)
