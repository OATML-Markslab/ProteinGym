from tqdm.auto import tqdm
import random

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset
from deli import load_json

from mulan.tokenizer import Tokenizer


class ProteinDataset(Dataset):
    def __init__(self, protein_data_path, 
                 saved_dataset_path=None,
                 split_ids_file=None, 
                 split='test', 
                 min_protein_length=1, 
                 max_protein_length=100000000000, # dummy big number for unlimited length 
                 use_sorted_batching=False, 
                 # aka batch size (number of tokens per batch), 
                 # can be reduced up to the maximum protein length used
                 batch_limit=5000, 
                 predict_contacts='none',
                 use_foldseek_sequences=False,
                 extract_foldseek_in_tokenizer=True,
                 is_experimental_structure=False,
                 foldseek_path=None,
                 ):

        self.tokenizer = Tokenizer(use_foldseek_sequences=use_foldseek_sequences)
        self.nan_value = np.deg2rad(self.tokenizer.nan_fill_value)
        self.pad_value = self.tokenizer.pad_value
        self.use_foldseek_sequences = use_foldseek_sequences

        self.split = split

        self.predict_contacts = predict_contacts

        self.sequences, self.angles, self.protein_names = self.tokenizer.load_dataset(saved_dataset_path, 
                                                                                      split)
        print('INIT ANGLES', self.angles)
        if self.sequences is None:
            if split_ids_file is not None:
                protein_ids = load_json(split_ids_file)[split]
            else:
                protein_ids = None
            self.tokenizer.save_dataset(protein_data_path, saved_dataset_path, split,
                                        is_experimental_structure=is_experimental_structure, 
                                        extract_foldseek=extract_foldseek_in_tokenizer, 
                                        protein_ids=protein_ids,
                                        foldseek_path=foldseek_path)
            self.sequences, self.angles, self.protein_names = self.tokenizer.load_dataset(saved_dataset_path, 
                                                                                          split)
        
        self.get_protein_length = lambda seq: int(len(seq) / 2) if self.use_foldseek_sequences else len(seq)

        lengths = [self.get_protein_length(seq) for seq in self.sequences]
        cumsum = np.cumsum(lengths)
        self.angles = [self.angles[start:end] for start, end in tqdm(zip(np.insert(cumsum, 0, 0)[:-1], cumsum))]

        print('Protein lengths before:', min(lengths), max(lengths))
        if max_protein_length == -1:
            max_protein_length = max(lengths)

        self.protein_names = [name for name, seq in zip(self.protein_names, self.sequences) 
                              if self.get_protein_length(seq) >= min_protein_length 
                              and self.get_protein_length(seq) <= max_protein_length]
        self.angles = [angle for angle, seq in zip(self.angles, self.sequences) 
                              if self.get_protein_length(seq) >= min_protein_length 
                              and self.get_protein_length(seq) <= max_protein_length]
        self.sequences = [seq for seq in self.sequences 
                         if self.get_protein_length(seq) >= min_protein_length 
                         and self.get_protein_length(seq) <= max_protein_length]
        
        lengths = [self.get_protein_length(seq) for seq in self.sequences]
        print('Protein lengths after:', min(lengths), max(lengths))
        print('Check for lengths', len(self.sequences[0]), lengths[0], self.sequences[0])

        print('self.angles', len(self.angles), self.angles[0].shape)
        self.use_sorted_batching = use_sorted_batching
        print('use_sorted_batching', use_sorted_batching)

        self.form_batches(batch_limit)
        self.sampling_indices = list(range(len(self.sequences)))

        print('self.angles', len(self.angles), self.angles[0].shape)
        print('self.plddts', len(self.plddts), self.plddts[0].shape) 
        print(self.plddts[0])
        print()

    def get_sorted_batches(self, lengths, batch_limit):
        batch_indices = []
        cur_batch = []
        if self.use_sorted_batching:
            for i, cur_len in enumerate(lengths[self.sorted_indices]):
                if (len(cur_batch) + 1) * cur_len <= batch_limit:
                    cur_batch.append(i)
                else:
                    batch_indices.append(cur_batch)
                    if cur_len <= batch_limit:
                        cur_batch = [i]
                    else:
                        print('Protein too long', i, cur_len, cur_len ** 2)
                        break

            if len(cur_batch) * cur_len <= batch_limit:
                print('append the last batch')
                batch_indices.append(cur_batch)
        else:
            batch_indices = [[i] for i, cur_len in enumerate(lengths[self.sorted_indices])]
        return batch_indices

    def init_sorted_indices(self):
        # save the order of sorted lengths for sorted batching
        lengths = np.array([self.get_protein_length(seq) for seq in self.sequences])
        self.sorted_indices = np.argsort(lengths)
        return lengths
    
    def rearrange_data(self, data, batch_indices):
        return [[data[self.sorted_indices[i]] for i in batch] for batch in batch_indices]

    def form_batches(self, batch_limit):

        lengths = self.init_sorted_indices()
        print(len(self.sequences), len(lengths), len(self.sorted_indices))

        batch_indices = self.get_sorted_batches(lengths, batch_limit)

        self.sequences = self.rearrange_data(self.sequences, batch_indices)
        self.protein_names = self.rearrange_data(self.protein_names, batch_indices)
        self.angles = self.rearrange_data(self.angles, batch_indices)

        batched_angles = []
        self.plddts = []
        self.coords = []
        for batch_angles in tqdm(self.angles, desc='Preproc angles'):
            tensor_batch_angles = np.ones((len(batch_angles), batch_angles[-1].shape[0] + 2,
                                           batch_angles[-1].shape[-1]), 
                                           dtype=np.float32) * self.pad_value
            for i, seq_angles in enumerate(batch_angles):
                tensor_batch_angles[i, 1:len(seq_angles)+1] = seq_angles
                
            if tensor_batch_angles.shape[-1] == 11:
                batched_angles.append(tensor_batch_angles[:, :, 4:])
            else:
                batched_angles.append(tensor_batch_angles)

            self.plddts.append(tensor_batch_angles[:, 1:-1, 0]) 
            self.coords.append(tensor_batch_angles[:, 1:-1, 1:4]) 

        self.angles = batched_angles

    def __len__(self):
        return len(self.sampling_indices)

    def __getitem__(self, ind):
        i = self.sampling_indices[ind]

        output = (self.sequences[i], torch.tensor(self.angles[i]), self.protein_names[i])
        output += (torch.tensor(self.plddts[i]), )
        if self.predict_contacts != 'none':
            output += (torch.tensor(self.coords[i]), )
        else:
            output += ([], )
        return output


def mask_inputs_(inputs, labels, special_tokens_mask, struct_inputs, 
                esm_tokenizer, mlm_probability, struct_labels, 
                all_amino_acid_ids, use_foldseek_sequences):

    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(
        labels.shape, 0.8)).bool() & masked_indices
    
    if use_foldseek_sequences:
        to_replace = []
        for token_id in inputs[indices_replaced]:
            token = esm_tokenizer.id_to_token(int(token_id))
            token = "#" + token[-1]
            to_replace.append(esm_tokenizer.token_to_id(token))
        inputs[indices_replaced] = torch.tensor(to_replace) #esm_tokenizer.mask_token_id
    else:
        inputs[indices_replaced] = esm_tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(
        labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_word_indices = torch.randint(
        len(all_amino_acid_ids), labels.shape, dtype=torch.long)
    random_words = all_amino_acid_ids[random_word_indices]

    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    # mask the same angles and letters
    angle_masked_indices = masked_indices.clone()
    angle_indices_replaced = indices_replaced.clone()
    angle_indices_random = indices_random.clone()
        
    ok_angles_mask = (~special_tokens_mask & ~angle_masked_indices).bool()
    random_indices = random.sample(range(ok_angles_mask.sum()), angle_indices_random.sum())
    random_indices = torch.tensor(random_indices)
    replaced_angles = struct_inputs[ok_angles_mask][random_indices].clone()

    struct_inputs.masked_fill_(angle_indices_replaced.unsqueeze(-1), value=-4.)
    struct_inputs[angle_indices_random] = replaced_angles

    return inputs, labels, struct_inputs, struct_labels


def data_collate_fn_dynamic(protein_parts_tuple, esm_tokenizer, nan_value, 
                         mask_inputs, 
                         all_amino_acids,
                         use_foldseek_sequences,
                         mask_angle_inputs_with_plddt=True, 
                         max_prot_len=100000000000000,
                         predict_contacts='none',
                         mlm_probability=0.15):
    
    get_protein_length = lambda seq: int(len(seq) / 2) if use_foldseek_sequences else len(seq)

    protein_parts_tuple = protein_parts_tuple[0]
    sequences = protein_parts_tuple[0]

    struct_inputs = protein_parts_tuple[1]
    plddts = protein_parts_tuple[3]
    batch_coords = protein_parts_tuple[4]

    if use_foldseek_sequences:
        crop_multiplier = 2
    else:
        crop_multiplier = 1

    # crop long proteins
    if get_protein_length(sequences[-1]) > max_prot_len:
        start_indices = [np.random.randint(get_protein_length(seq) - max_prot_len + 1)
                         if get_protein_length(seq) > max_prot_len else 0 for seq in sequences]
        sequences = [seq[start_ind * crop_multiplier: (start_ind + max_prot_len) * crop_multiplier] 
                     for seq, start_ind in zip(sequences, start_indices)]

        start_struct_padding = struct_inputs[:, :1].clone()
        end_struct_padding = struct_inputs[:, -1:].clone()
        struct_inputs = torch.stack([struct_inputs[i, 1 + start_indices[i]:1 + start_indices[i] + max_prot_len] 
                                     for i in range(len(sequences))])
        struct_inputs = torch.cat([start_struct_padding, struct_inputs, end_struct_padding], dim=1)
        plddts = torch.stack([plddts[i, start_indices[i]:start_indices[i] + max_prot_len] 
                                     for i in range(len(sequences))])
        if predict_contacts != 'none':
            batch_coords = torch.stack([batch_coords[i, start_indices[i]:start_indices[i] + max_prot_len] 
                                        for i in range(len(sequences))])

    encoded = esm_tokenizer.batch_encode_plus(sequences, return_special_tokens_mask=True, 
                                              padding=True, return_tensors='pt')

    if not use_foldseek_sequences:
        unk_token_id = esm_tokenizer.token_to_id('X')
        # fill X tokens as special tokens
        encoded['special_tokens_mask'][encoded['input_ids'] == unk_token_id] = 1

    distance_matrices = []
    if predict_contacts != 'none' and not predict_contacts.startswith('eval'):
        distance_matrices = np.linalg.norm(batch_coords[:, None] - batch_coords[:, :, None], axis=-1)

        bad_plddt_mask = plddts <= 70.

        if predict_contacts == 'contact':
            bin_value = 8.
            distance_matrices = (distance_matrices < bin_value).astype(np.float32)
        elif predict_contacts == 'bin_distance':
            boundaries = np.array([5., 8., 16., 32.])
            distance_matrices = np.digitize(distance_matrices, boundaries)

        # mask bad plddts regions with -1
        distance_matrices[bad_plddt_mask] = -1
        distance_matrices = distance_matrices.transpose((0, 2, 1))
        distance_matrices[bad_plddt_mask] = -1
    
    ignore_index = -100.
    pad_value = 4.

    struct_labels = struct_inputs[:, :, :7].clone()
    struct_labels = (struct_labels + torch.pi) / (2 * torch.pi)
    struct_labels.masked_fill_(struct_inputs[:, :, :7] == nan_value, value=ignore_index)
    struct_labels.masked_fill_(struct_inputs[:, :, :7] == pad_value, value=ignore_index)

    aa_mapping_edges = {esm_tokenizer.token_to_id(token): i for i, token in enumerate(all_amino_acids)}
    all_amino_acid_ids = torch.tensor([uid for uid in aa_mapping_edges.keys() 
                                        if uid != esm_tokenizer.token_to_id('X')])
    aa_mapping_edges[esm_tokenizer.mask_token_id] = len(aa_mapping_edges)

    inputs = encoded['input_ids']
    labels = None
    if mask_inputs:
        inputs, labels, struct_inputs, struct_labels = mask_inputs_(
            encoded['input_ids'], encoded['input_ids'].clone(), encoded['special_tokens_mask'], 
            struct_inputs, esm_tokenizer, mlm_probability, struct_labels, 
            all_amino_acid_ids, use_foldseek_sequences)
        
    # pad plddts
    padded_plddts = torch.ones_like(inputs) * pad_value
    padded_plddts[:, 1:-1] = plddts

    if mask_angle_inputs_with_plddt:
        bad_plddt_mask = padded_plddts <= 70.
        struct_inputs[bad_plddt_mask] = pad_value

    struct_inputs = (struct_inputs, padded_plddts)

    labels = (labels, torch.tensor(distance_matrices), struct_labels)
    ret_dict = {'input_ids': inputs, 
                'labels': labels,
                'attention_mask': encoded['attention_mask'], 
                'struct_inputs': struct_inputs,
                }
            
    return ret_dict
