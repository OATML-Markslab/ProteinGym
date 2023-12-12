import os
import argparse
import tqdm 
import json 

from scipy.stats import spearmanr
import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss

from tokenizers import Tokenizer
from models.progen.modeling_progen import ProGenForCausalLM


########################################################################
# model

def create_model(ckpt, fp16):
    if fp16:
        return ProGenForCausalLM.from_pretrained(ckpt, revision='float16', torch_dtype=torch.float16, low_cpu_mem_usage=True)
    else:
        return ProGenForCausalLM.from_pretrained(ckpt)


def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

########################################################################
# fitness

def calc_fitness(model, prots, tokenizer, device='cuda:0', model_context_len=1024, fp16=False, reduction='mean'):
    loss_list = []
    loss_fn = CrossEntropyLoss()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=fp16):
            for prot in tqdm.tqdm(prots):
                loss_val = 0
                
                sequence_chunks=[]
                if len(prot) < model_context_len:
                    sequence_chunks = [prot]
                else:
                    len_target_seq = len(prot)
                    num_windows = 1 + int( len_target_seq / model_context_len)
                    start=0
                    for window_index in range(1, num_windows+1):
                        sequence_chunks.append(prot[start:start+model_context_len])
                        start += model_context_len
                
                for chunk in sequence_chunks:
                    for p in [chunk, chunk[::-1]]:
                        ids = torch.tensor(tokenizer.encode(p).ids).to(device)

                        input_ids = ids[:-1]
                        targets   = ids[1:]
                        
                        logits=model(input_ids).logits

                        # remove terminals
                        bos_token, eos_token = 3, 4
                        if targets[-1] in [bos_token, eos_token]:
                            logits = logits[:-1, ...]
                            targets = targets[:-1]
                        assert (targets == bos_token).sum() == 0
                        assert (targets == eos_token).sum() == 0

                        # remove unused logits
                        first_token, last_token = 5, 29
                        logits = logits[:, first_token:(last_token+1)]
                        targets = targets - first_token

                        assert logits.shape[1] == (last_token - first_token + 1)

                        loss = loss_fn(target=targets.view(-1), input=logits.view(-1,logits.size(-1)))
                        loss_val += - loss.item()
                
                loss_val /= 2.0 #normalizing for mirroring

                if reduction=='mean':
                    loss_val /= len(prot) #average by seq length

                loss_list += [loss_val]
    return np.array(loss_list)

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
    return "1"+"".join(mutated_seq)+"2"

def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with Tranception.
    """

    models_151M = [ 'progen2-small' ]
    models_754M = [ 'progen2-medium', 'progen2-oas', 'progen2-base' ]
    models_2B = [ 'progen2-large', 'progen2-BFD90' ]
    models_6B = [ 'progen2-xlarge' ]
    models = models_151M + models_754M + models_2B + models_6B

    parser = argparse.ArgumentParser(description='Tranception scoring')
    parser.add_argument('--Progen2_model_name_or_path', default="/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/baseline_models/progen2/progen2-small", type=str, help='Name of or path to Progen2 model')
    parser.add_argument('--DMS_reference_file_path', default='/home/pn73/Tranception/proteingym/ProteinGym_reference_file_substitutions.csv', type=str, help='Path of DMS folder')
    parser.add_argument('--DMS_data_folder', default='/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/Tranception_open_source/DMS_files/ProteinGym_substitutions', type=str, help='Path of DMS folder')
    parser.add_argument('--DMS_index', type=int, help='Path of DMS folder')
    parser.add_argument('--output_scores_folder', default=None, type=str, help='Name of folder to write model scores to')
    parser.add_argument('--indel_mode', action='store_true', help='Whether to score sequences with insertions and deletions')
    parser.add_argument('--fp16', action='store_true', help='Whether to score sequences with half precision')
    parser.add_argument('--test', action='store_true', help='Test mode of fitness computation')
    args = parser.parse_args()

    model = create_model(ckpt=args.Progen2_model_name_or_path, fp16=args.fp16).cuda()
    config = json.load(open(args.Progen2_model_name_or_path+os.sep+'config.json',"r"))
    print("Maximum context length: {}".format(config['n_positions']))
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tokenizer_path = os.path.join(dir_path, 'tokenizer.json')
    tokenizer = create_tokenizer_custom(file=tokenizer_path)

    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    DMS_id=list_DMS[args.DMS_index]
    print("Computing scores for: {} with Progen2: {}".format(DMS_id, args.Progen2_model_name_or_path))
    DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
    target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
    
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
    DMS_data['mutated_sequence'] = DMS_data['mutant'].apply(lambda x: get_mutated_sequence(target_seq, x)) if not args.indel_mode else DMS_data['mutant'].map(lambda x: "1"+x+"2")

    if args.test:
        x_uniref90bfd30 = '2GFLPFRGADEGLAAREAATLAARGTAARAYREDSWAVPVPRGLLGDLTARVAALGAASPPPADPLAVTLDLHHVTAEVALTTVLDAATLVHGQTRVLSAEDAAEAATAAAAATEAYLERLQDFVLFMSASVRVWRRGNAAGATGPEWDQWYTVADRDALGSAPTHLAVLGRQADALCHFVLDRVAWGTCGTPLWSGDEDLGNVVATFAGYADRLATAPRDLIM1'
        x_oas = '1EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPWKGLEYVSAISSNGGSTYYANSVKGRFTISRDNSKNTLYLQMGSLRAEDMAVYYCARDESGYSYGWGYYFDYWGQGTLVTVSS2'
        x_bfd90 = '1TAPRSTRASGSEGSRPPGIPAKGRRCLPSRAGSVTPRFRHARQGTATVAKEQGRKLIASNRKARHDYHIEDTFEAGLVLTGTEVKSLRMGRASLIDGYAVFYGEELWLEGVHIPEYLNGNWTNHTPRRRRKLLLNRSELTKLAHKTSESGHTIVPLALYFKDGRAKVEIAVAKGKKAYDKRHALRERQDQREV2'
        model_size = args.Progen2_model_name_or_path.split('/')[-1]
        print("Model: {}".format(model_size))
        checkpoint_x_ll = {
                'progen2-small': (x_uniref90bfd30, -2.4),
                'progen2-medium': (x_uniref90bfd30, -1.9),
                'progen2-base': (x_uniref90bfd30, -1.9),
                'progen2-large': (x_uniref90bfd30, -1.8),
                'progen2-xlarge': (x_uniref90bfd30, -1.0),
        }
        model_scores = calc_fitness(model=model, prots=np.array([checkpoint_x_ll[model_size][0]]), tokenizer=tokenizer, fp16=args.fp16, reduction='sum')
        print(model_scores, checkpoint_x_ll[model_size][1], abs(model_scores - checkpoint_x_ll[model_size][1]))
        assert abs(model_scores - checkpoint_x_ll[model_size][1]) < 0.1
    
    model_scores = calc_fitness(model=model, prots=np.array(DMS_data['mutated_sequence']), model_context_len=int(config['n_positions']), tokenizer=tokenizer, fp16=args.fp16)
    
    DMS_data['Progen2_score']=model_scores
    scoring_filename = args.output_scores_folder+os.sep+DMS_id+'.csv'
    DMS_data[['mutant','Progen2_score','DMS_score']].to_csv(scoring_filename, index=False)

if __name__ == '__main__':
    main()
