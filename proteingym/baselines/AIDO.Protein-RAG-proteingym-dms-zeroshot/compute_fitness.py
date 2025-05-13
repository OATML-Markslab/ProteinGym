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
    ## Create structure tokenizer
    #############################

    str_tokenizer = misc.AIDO_Structure_Tokenizer(device='cuda:0')

    #############################
    ## Define model and tokenizer
    #############################

    tokenizer  = AutoTokenizer.from_pretrained("genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot", trust_remote_code=True)
    model      = AutoModelForCausalLM.from_pretrained("genbio-ai/AIDO.Protein-RAG-16B-proteingym-dms-zeroshot", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model      = model.bfloat16().eval().to('cuda:0')

    #############################
    ## Run model and get predicted DMS scores
    #############################

    os.makedirs(args.output_path, exist_ok=True)
    for dms_id in args.dms_ids:
        
        ## Read Sequence
        dms2seq, dms2annot = misc.load_fasta(f"{SCRIPT_PATH}/query.fasta", load_annotation=True)
        q_seq = dms2seq[dms_id]
        start, end = [int(x) for x in dms2annot[dms_id].split('-')]

        ## Read MSA
        msa = misc.load_msa_txt(f"{SCRIPT_PATH}/msa_data/{dms_id}.txt.gz")
        assert q_seq == msa[0]

        ## Read PDB
        with open(f"{SCRIPT_PATH}/struc_data/{dms_id}.pdb") as IN:
            text = IN.read()

        prot = protein.from_pdb_string(text, molecular_type='protein')
        assert prot.seq(True) == q_seq
        
        ## Read DMS Table
        dms_df = pd.read_csv(f'{SCRIPT_PATH}/dms_data/{dms_id}.csv')
    
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
    all_dms_ids = [n[:-7] for n in os.listdir(f'{SCRIPT_PATH}/msa_data') if n.endswith('.txt.gz')]
    parser.add_argument("--dms_ids",     type=str, default=all_dms_ids, nargs="+")
    parser.add_argument("--output_path", type=str, default=f"{SCRIPT_PATH}/output", help="Output path",)
    parser.add_argument("--mask-str", action='store_true', help="Mask the structure input")
    args = parser.parse_args()
    main(args)
