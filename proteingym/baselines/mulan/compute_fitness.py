import os
# os.environ["HF_HOME"] = './workspace/data/transformers_cache'
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mulan.dataset import ProteinDataset, data_collate_fn_dynamic
from mulan.model import StructEsmForMaskedLM
from mulan.model_utils import auto_detect_base_tokenizer

import os
import argparse

import numpy as np
import pandas as pd

import torch
from tqdm import tqdm



def predict_mut(model, esm_tokenizer, dataloader, use_foldseek, mut_info: str) -> float:
    """
    Predict the mutational effect of a given mutation
    Args:
        seq: The wild type sequence

        mut_info: The mutation information in the format of "A123B", where A is the original amino acid, 123 is the
                  position and B is the mutated amino acid. If multiple mutations are provided, they should be
                  separated by colon, e.g. "A123B:C124D".

    Returns:
        The predicted mutational effect
    """
    foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

    with torch.no_grad():
        for batch in dataloader:
            batch['input_ids'], batch['struct_inputs'] = mask_mutated_positions(
                batch['input_ids'], 
                batch['struct_inputs'], 
                esm_tokenizer, 
                mut_info,
            )
            struct_inputs = [struct_input.to(model.device) for struct_input in batch['struct_inputs']]
            logits = model(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                struct_inputs=struct_inputs,
                output_hidden_states=False,
            )["logits"]["scores"]

            probs = logits.softmax(dim=-1)
        
            score = 0
            for single in mut_info.split(":"):
                ori_aa, pos, mut_aa = single[0], int(single[1:-1]), single[-1]
                if use_foldseek:
                    ori_st = esm_tokenizer.get_vocab()[ori_aa + foldseek_struc_vocab[0]]
                    mut_st = esm_tokenizer.get_vocab()[mut_aa + foldseek_struc_vocab[0]]
                    
                    ori_prob = probs[0, pos, ori_st: ori_st + len(foldseek_struc_vocab)].sum()
                    mut_prob = probs[0, pos, mut_st: mut_st + len(foldseek_struc_vocab)].sum()
                else:
                    ori_st = esm_tokenizer.get_vocab()[ori_aa]
                    mut_st = esm_tokenizer.get_vocab()[mut_aa]
                    
                    ori_prob = probs[0, pos, ori_st]
                    mut_prob = probs[0, pos, mut_st]                
                score += torch.log(mut_prob / ori_prob)

    return score


def calc_fitness(model, dataloader, DMS_data, esm_tokenizer, use_foldseek, mutation_col='mutant', 
                 offset_idx=1):
    log_proba_list = []
    for mut_info in tqdm(DMS_data[mutation_col]):
        mutations = []
        for row in mut_info.split(":"):
            ori, pos, mut = row[0], row[1: -1], row[-1]
            pos = int(pos) - offset_idx + 1
            mutations.append(f"{ori}{pos}{mut}")

        all_mut = ":".join(mutations)
        score = predict_mut(model, esm_tokenizer, dataloader, use_foldseek, all_mut).item()
        log_proba_list.append(score)
    
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
            print("Issue with mutant: " + str(mutation))
        relative_position = position - start_idx
        assert (from_AA == focus_seq[relative_position]), "Invalid from_AA or mutant position: " + str(
            mutation) + " from_AA: " + str(from_AA) + " relative pos: " + str(relative_position) + " focus_seq: " + str(
            focus_seq)
        assert (to_AA in AA_vocab), "Mutant to_AA is invalid: " + str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)


def mask_mutated_positions(inputs, struct_inputs, esm_tokenizer, mut_info):
    positions = np.array([int(single[1:-1]) for single in mut_info.split(":")])

    masked_indices = torch.zeros_like(inputs, dtype=bool, device=inputs.device)
    masked_indices[0, positions] = True

    inputs[masked_indices] = esm_tokenizer.mask_token_id
    struct_inputs[0].masked_fill_(masked_indices.unsqueeze(-1), value=-4.)

    return inputs, struct_inputs


def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with MULAN.
    """
    parser = argparse.ArgumentParser(description='MULAN scoring')
    parser.add_argument('--MULAN_model_name_or_path',
                        default="DFrolova/MULAN-small",
                        type=str, help='Name of or path to MULAN model')
    parser.add_argument('--DMS_reference_file_path',
                        default='reference_files/DMS_substitutions.csv',
                        type=str, help='Path of DMS_substitutions.csv file')
    parser.add_argument('--DMS_data_folder',
                        default='./workspace/data/docking/downstream_tasks/ProteinGym_data/DMS_ProteinGym_substitutions',
                        type=str, help='Path of DMS data folder')
    parser.add_argument('--structure_data_folder', 
                        default='./workspace/data/docking/downstream_tasks/ProteinGym_data/ProteinGym_AF2_structures/', 
                        type=str, help='Path of structure folder')
    parser.add_argument('--DMS_index', type=int, help='DMS index')
    parser.add_argument('--use_foldseek', action="store_true", help='Whether the model uses foldseek sequences')
    parser.add_argument('--foldseek_path', 
                        default='./bin/foldseek', 
                        type=str, help='Path of foldseek binary file')
    parser.add_argument('--output_scores_folder', 
                        default='./workspace/data/docking/downstream_tasks/ProteinGym_results/sub_results/MULAN/MULAN_small/', type=str,
                        help='Name of folder to write model scores to')
    parser.add_argument('--output_dataset_folder', 
                        default='./workspace/data/docking/downstream_tasks/ProteinGym_results/DMS_substitution_dataset/', type=str,
                        help='Name of folder to save preprocessed MULAN dataset to')
    parser.add_argument('--hf_cache_folder',type=str,help="location of HF cache", default="/workspace/data/transformers_cache")
    parser.add_argument('--indel_mode', action='store_true',
                        help='Whether to score sequences with insertions and deletions')
    args = parser.parse_args()
    os.environ["HF_HOME"] = args.hf_cache_folder

    use_foldseek_sequences = args.use_foldseek

    # preprocess the dataset
    dataset = ProteinDataset(
        protein_data_path=args.structure_data_folder, 
        saved_dataset_path=args.output_dataset_folder,
        use_foldseek_sequences=use_foldseek_sequences,
        is_experimental_structure=False,
        extract_foldseek_in_tokenizer=use_foldseek_sequences,
        foldseek_path=args.foldseek_path,
    )

    # load model
    model = StructEsmForMaskedLM.from_pretrained(
        args.MULAN_model_name_or_path,
    #    device_map="auto",
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device', device)
    model.to(device)

    # prepare the dataloader
    esm_tokenizer = auto_detect_base_tokenizer(model.config, use_foldseek_sequences)

    # Initialize dataloader
    def data_collator(x): 
        if use_foldseek_sequences:
            one_letter_aas = esm_tokenizer.all_tokens[5:]
        else: 
            one_letter_aas = dataset.tokenizer.one_letter_aas

        return data_collate_fn_dynamic(x, 
            esm_tokenizer=esm_tokenizer,
            nan_value=np.deg2rad(dataset.tokenizer.nan_fill_value),
            mask_inputs=False,
            all_amino_acids=one_letter_aas,
            use_foldseek_sequences=use_foldseek_sequences)

    # read DMS data    
    mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
    list_DMS = mapping_protein_seq_DMS["DMS_id"]

    # for DMS_index in range(217):
    DMS_index = args.DMS_index
    DMS_id = list_DMS[DMS_index]
    scoring_filename = args.output_scores_folder + os.sep + DMS_id + '.csv'
    if os.path.exists(scoring_filename):
        print("Scores already computed for: {}".format(DMS_id))
        # continue
    
    else:
        print("Computing scores for: {}, {} with MULAN: {}".format(DMS_index, DMS_id, args.MULAN_model_name_or_path))
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
        target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].upper()
        
        DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
        DMS_data['mutated_sequence'] = DMS_data['mutant'].apply(
            lambda x: get_mutated_sequence(target_seq, x)) if not args.indel_mode else DMS_data['mutant']
            
        pdb_filenames = mapping_protein_seq_DMS["pdb_file"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[
            0].split('|')  # if sequence is large (eg., BRCA2_HUMAN) the structure is split in several chunks
        pdb_ranges = mapping_protein_seq_DMS["pdb_range"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].split(
            '|')

        print('pdb_filenames', pdb_filenames)
        print('pdb_ranges', pdb_ranges)
        model_scores = []
        for pdb_index, pdb_filename in enumerate(pdb_filenames):
            dataset_id_index = [i for i, name in enumerate(dataset.protein_names) if name[0] == pdb_filename[:-4]]
            dataset_id_index = dataset_id_index[0]
            id_dataset = Subset(dataset, [dataset_id_index])

            dataloader = DataLoader(id_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)

            pdb_file = args.structure_data_folder + os.sep + pdb_filename
            print(pdb_file)
            pdb_range = [int(x) for x in pdb_ranges[pdb_index].split("-")]

            # fix for this ID because of the shift in pdb sequence and target_seq_split
            if DMS_id == 'POLG_HCVJF_Qi_2014':
                print(pdb_range)
                pdb_range[0] = pdb_range[0] + 1
                print(pdb_range)

            target_seq_split = target_seq[pdb_range[0] - 1:pdb_range[1]]  # pdb_range is 1-indexed
            DMS_data["mutated_position"] = DMS_data['mutant'].apply(lambda x: int(x.split(':')[0][1:-1])) #if multiple mutant, will extract position of first mutant
            filtered_DMS_data = DMS_data[
                (DMS_data["mutated_position"] >= pdb_range[0]) & (DMS_data["mutated_position"] <= pdb_range[1])]
            model_scores.append(calc_fitness(model=model, dataloader=dataloader, DMS_data=filtered_DMS_data,
                                                use_foldseek=use_foldseek_sequences,
                                                esm_tokenizer=esm_tokenizer, offset_idx=pdb_range[0]))
        
        model_scores = np.concatenate(model_scores)
        
        DMS_data['MULAN_score'] = model_scores
        scoring_filename = args.output_scores_folder + os.sep + DMS_id + '.csv'
        DMS_data[['mutant', 'MULAN_score', 'DMS_score']].to_csv(scoring_filename, index=False)


if __name__ == '__main__':
    main()
