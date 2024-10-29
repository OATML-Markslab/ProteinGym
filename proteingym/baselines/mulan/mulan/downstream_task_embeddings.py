import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deli import save_json, save

from mulan.dataset import ProteinDataset, data_collate_fn_dynamic


def flatten_list(dataset_list):
    flattened_list = []
    for protein_list in dataset_list:
        flattened_list += list(protein_list)
    return flattened_list


def get_downstream_datasets(saved_dataset_path, dataset_names, 
                            batch_limit, use_foldseek_sequences, is_experimental_structure):
    min_protein_length = 1
    max_protein_length = -1
    split_ids_file = None
    protein_data_path = None
    use_sorted_batching = True
    predict_contacts = 'none'

    print('saved_dataset_path', saved_dataset_path)
    print('Chunks:', dataset_names)
    all_datasets = {}
    for chunk_id in dataset_names:
        dataset = ProteinDataset(
            protein_data_path=protein_data_path, 
            saved_dataset_path=saved_dataset_path,
            split_ids_file=split_ids_file,
            split=chunk_id, 
            min_protein_length=min_protein_length, 
            max_protein_length=max_protein_length,
            use_sorted_batching=use_sorted_batching,
            batch_limit=batch_limit,
            predict_contacts=predict_contacts,
            use_foldseek_sequences=use_foldseek_sequences,
            is_experimental_structure=is_experimental_structure,
            )
        all_datasets[chunk_id] = dataset
    return all_datasets


def get_embeddings(dataset, model, esm_tokenizer,
                   protein_level, required_positions, 
                   mask_angle_inputs_with_plddt, use_foldseek_sequences):    
    mean_embeddings = []
    protein_embeddings = []
    all_names = []

    def data_collator(x): 
        if use_foldseek_sequences:
            one_letter_aas = esm_tokenizer.all_tokens[5:]
        else: 
            one_letter_aas = dataset.tokenizer.one_letter_aas

        return data_collate_fn_dynamic(x, esm_tokenizer=esm_tokenizer,
                          nan_value=np.deg2rad(dataset.tokenizer.nan_fill_value),
                          predict_contacts='none',
                          max_prot_len=100000000000000, # big number equals to no protein cropping
                          mask_inputs=False,
                          mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
                          all_amino_acids=one_letter_aas,
                          use_foldseek_sequences=use_foldseek_sequences)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=data_collator)
    device = model.device
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            struct_inputs = [struct_inp.to(device) for struct_inp in batch['struct_inputs']]
            res = model(
                input_ids=batch['input_ids'].to(device), 
                attention_mask=batch['attention_mask'].to(device),
                struct_inputs=struct_inputs,
                output_hidden_states=True
            )['hidden_states'][-1]

            get_protein_length = lambda seq: int(len(seq) / 2) if use_foldseek_sequences else len(seq)
            lengths = [get_protein_length(seq) for seq in dataset.sequences[i]]

            if protein_level:
                for p_ind in range(len(lengths)):
                    mean_embeddings.append(res[p_ind, 1:lengths[p_ind] + 1].mean(dim=0).cpu())

            else:
                for p_ind in range(len(lengths)):
                    if required_positions is not None:
                        cur_positions = required_positions[i][p_ind] - 1
                    else:
                        cur_positions = range(lengths[p_ind])
                    mean_embeddings.extend(res[p_ind, 1:lengths[p_ind] + 1][cur_positions].cpu())
                
                for p_ind in range(len(lengths)):
                    all_names.append([f'{dataset.protein_names[i][p_ind]}_{pos}' for pos in range(lengths[p_ind])])

    if protein_level:          
        mean_embeddings = torch.stack(mean_embeddings).numpy()
        ret_data = (mean_embeddings, protein_embeddings)
    else:
        mean_embeddings = torch.stack(mean_embeddings).numpy()
        ret_data = (mean_embeddings, all_names)

    return ret_data


def evaluate_downstream_task(model, esm_tokenizer, saved_dataset_path, 
                             downstream_dataset_path,
                             batch_limit, mask_angle_inputs_with_plddt,
                             protein_level, use_foldseek_sequences,
                             is_experimental_structure=False,
                             dataset_names=['valid', 'test', 'train']):

    all_datasets = get_downstream_datasets(
        saved_dataset_path=saved_dataset_path, 
        dataset_names=dataset_names,
        batch_limit=batch_limit,
        use_foldseek_sequences=use_foldseek_sequences,
        is_experimental_structure=is_experimental_structure,
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    print('downstream_dataset_path', downstream_dataset_path)
    os.makedirs(downstream_dataset_path, exist_ok=True)
    
    for dataset_name in all_datasets.keys():
        ret_embeddings = get_embeddings(
            dataset=all_datasets[dataset_name],
            model=model,
            esm_tokenizer=esm_tokenizer,
            protein_level=protein_level,
            use_foldseek_sequences=use_foldseek_sequences,
            required_positions=None,
            mask_angle_inputs_with_plddt=mask_angle_inputs_with_plddt,
        )

        if protein_level:
            mean_embeddings, _ = ret_embeddings
            names = []
            for batch_names in all_datasets[dataset_name].protein_names:
                names += batch_names
        else:
            mean_embeddings, all_names = ret_embeddings
            names = flatten_list(all_names)
            
        save_json(names, 
                os.path.join(downstream_dataset_path, f'{dataset_name}_names.json'))
        print(len(names), type(mean_embeddings), mean_embeddings.shape)
        save(mean_embeddings, os.path.join(downstream_dataset_path, 
                                        f'{dataset_name}_avg_embeddings.npy.gz'), compression=1)