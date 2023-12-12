import os
import argparse
import json
import pandas as pd

import torch

from transformers import PreTrainedTokenizerFast
import tranception
from tranception import config, model_pytorch

dir_path = os.path.dirname(os.path.abspath(__file__))

def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with Tranception.
    """
    parser = argparse.ArgumentParser(description='Tranception scoring')
    parser.add_argument('--checkpoint', type=str, help='Path of Tranception model checkpoint')
    parser.add_argument('--model_framework', default='pytorch', type=str, help='Underlying framework [pytorch|JAX]')
    parser.add_argument('--batch_size_inference', default=20, type=int, help='Batch size for inference')

    #We may pass in all required information about the DMS via the provided reference files, or specify all relevant fields manually
    parser.add_argument('--DMS_reference_file_path', default=None, type=str, help='Path to reference file with list of DMS to score')
    parser.add_argument('--DMS_index', default=0, type=int, help='Index of DMS assay in reference file')
    #Fields to be passed manually if reference file is not used
    parser.add_argument('--target_seq', default=None, type=str, help='Full wild type sequence that is mutated in the DMS asssay')
    parser.add_argument('--DMS_file_name', default=None, type=str, help='Name of DMS assay file')
    parser.add_argument('--MSA_filename', default=None, type=str, help='Name of MSA (eg., a2m) file constructed on the wild type sequence')
    parser.add_argument('--MSA_weight_file_name', default=None, type=str, help='Weight of sequences in the MSA (optional)')
    parser.add_argument('--MSA_start', default=None, type=int, help='Sequence position that the MSA starts at (1-indexing)')
    parser.add_argument('--MSA_end', default=None, type=int, help='Sequence position that the MSA ends at (1-indexing)')

    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
    parser.add_argument('--output_scores_folder', default='./', type=str, help='Name of folder to write model scores to')
    parser.add_argument('--deactivate_scoring_mirror', action='store_true', help='Whether to deactivate sequence scoring from both directions (Left->Right and Right->Left)')
    parser.add_argument('--indel_mode', action='store_true', help='Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions')
    parser.add_argument('--scoring_window', default="optimal", type=str, help='Sequence window selection mode (when sequence length longer than model context size)')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of workers for model scoring data loader')
    parser.add_argument('--inference_time_retrieval', action='store_true', help='Whether to perform inference-time retrieval')
    parser.add_argument('--retrieval_inference_weight', default=0.6, type=float, help='Coefficient (alpha) used when aggregating autoregressive transformer and retrieval')
    parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')
    parser.add_argument('--MSA_weights_folder', default=None, type=str, help='Path to MSA weights for neighborhood scoring')
    parser.add_argument('--clustal_omega_location', default=None, type=str, help='Path to Clustal Omega (only needed with scoring indels with retrieval)')
    args = parser.parse_args()
    
    model_name = args.checkpoint.split("/")[-1]
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=dir_path+os.sep+"tranception/utils/tokenizers/Basic_tokenizer",
                                                unk_token="[UNK]",
                                                sep_token="[SEP]",
                                                pad_token="[PAD]",
                                                cls_token="[CLS]",
                                                mask_token="[MASK]"
                                            )

    if args.DMS_reference_file_path:
        mapping_protein_seq_DMS = pd.read_csv(args.DMS_reference_file_path)
        list_DMS = mapping_protein_seq_DMS["DMS_id"]
        DMS_id=list_DMS[args.DMS_index]
        print("Compute scores for DMS: "+str(DMS_id))
        target_seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0].upper()
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]
        if args.inference_time_retrieval:
            MSA_data_file = args.MSA_folder + os.sep + mapping_protein_seq_DMS["MSA_filename"][args.DMS_index] if args.MSA_folder is not None else None
            MSA_weight_file_name = args.MSA_weights_folder + os.sep + mapping_protein_seq_DMS["weight_file_name"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0] if args.MSA_weights_folder else None
            MSA_start = int(mapping_protein_seq_DMS["MSA_start"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) - 1 # MSA_start typically based on 1-indexing
            MSA_end = int(mapping_protein_seq_DMS["MSA_end"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0])
    else:
        target_seq=args.target_seq
        DMS_file_name=args.DMS_file_name
        DMS_id = DMS_file_name.split(".")[0]
        if args.inference_time_retrieval:
            MSA_data_file = args.MSA_folder + os.sep + args.MSA_filename if args.MSA_folder is not None else None
            MSA_weight_file_name = args.MSA_weights_folder + os.sep + args.MSA_weight_file_name if args.MSA_weights_folder is not None else None
            MSA_start = args.MSA_start - 1 # MSA_start based on 1-indexing
            MSA_end = args.MSA_end

    print(f"TMP Lood at score_tranception_proteingym.py: MSA_start={MSA_start}, MSA_end={MSA_end}")
    config = json.load(open(args.checkpoint+os.sep+'config.json'))
    config = tranception.config.TranceptionConfig(**config)
    config.attention_mode="tranception"
    config.position_embedding="grouped_alibi"
    config.tokenizer = tokenizer
    config.scoring_window = args.scoring_window

    if args.inference_time_retrieval:
        config.retrieval_aggregation_mode = "aggregate_indel" if args.indel_mode else "aggregate_substitution"
        config.MSA_filename=MSA_data_file
        config.full_protein_length=len(target_seq)
        config.MSA_weight_file_name=MSA_weight_file_name
        config.retrieval_inference_weight=args.retrieval_inference_weight
        config.MSA_start = MSA_start
        config.MSA_end = MSA_end
        if args.indel_mode:
            config.clustal_omega_location = args.clustal_omega_location
    else:
        config.retrieval_aggregation_mode = None
        
    if args.model_framework=="pytorch":
        model = tranception.model_pytorch.TranceptionLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.checkpoint,config=config)
        if torch.cuda.is_available():
            model.cuda()
    model.eval()
    
    if not os.path.isdir(args.output_scores_folder):
        os.mkdir(args.output_scores_folder)
    retrieval_type = '_retrieval_' + str(args.retrieval_inference_weight) if args.inference_time_retrieval else '_no_retrieval'
    mutation_type = '_indels' if args.indel_mode else '_substitutions'
    mirror_type = '_no_mirror' if args.deactivate_scoring_mirror else ''
    scoring_filename = args.output_scores_folder + os.sep + DMS_id + ".csv"

    
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
    print("TMP Lood starting scoring")
    all_scores = model.score_mutants(
                                    DMS_data=DMS_data, 
                                    target_seq=target_seq, 
                                    scoring_mirror=not args.deactivate_scoring_mirror, 
                                    batch_size_inference=args.batch_size_inference,  
                                    num_workers=args.num_workers, 
                                    indel_mode=args.indel_mode
                                    )
    all_scores.to_csv(scoring_filename, index=False)

if __name__ == '__main__':
    main()