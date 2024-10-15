import os
import argparse
import json
import pandas as pd

import torch

from transformers import PreTrainedTokenizerFast
import trancepteve
from trancepteve import config, model_pytorch
from trancepteve.utils import dms_utils

dir_path = os.path.dirname(os.path.abspath(__file__))

def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with TranceptEVE.
    """
    parser = argparse.ArgumentParser(description='TranceptEVE scoring')
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
    parser.add_argument('--UniprotID', default=None, type=str, help='Uniprot ID of protein (EVE retrieval only)')
    parser.add_argument('--MSA_threshold_sequence_frac_gaps', default=None, type=float, help='MSA processing: pct fragments threshold')
    parser.add_argument('--MSA_threshold_focus_cols_frac_gaps', default=None, type=float, help='MSA processing: pct col filled threshold')

    parser.add_argument('--DMS_data_folder', type=str, help='Path to folder that contains all DMS assay datasets')
    parser.add_argument('--output_scores_folder', default='./', type=str, help='Name of folder to write model scores to')
    parser.add_argument('--deactivate_scoring_mirror', action='store_true', help='Whether to deactivate sequence scoring from both directions (Left->Right and Right->Left)')
    parser.add_argument('--indel_mode', action='store_true', help='Flag to be used when scoring insertions and deletions. Otherwise assumes substitutions')
    parser.add_argument('--scoring_window', default="optimal", type=str, help='Sequence window selection mode (when sequence length longer than model context size)')
    
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for model scoring data loader')
    parser.add_argument('--inference_time_retrieval_type', default=None, type=str, help='Type of inference time retrieval [None,Tranception,TranceptEVE]')
    parser.add_argument('--retrieval_weights_manual', action='store_true', help='Whether to manually select the MSA/EVE aggregation weights')
    parser.add_argument('--retrieval_inference_MSA_weight', default=0.5, type=float, help='Coefficient (alpha) used when aggregating autoregressive transformer and MSA retrieval')
    parser.add_argument('--retrieval_inference_EVE_weight', default=0.5, type=float, help='Coefficient (beta) used when aggregating autoregressive transformer and EVE retrieval')
    
    parser.add_argument('--MSA_folder', default='.', type=str, help='Path to MSA for neighborhood scoring')
    parser.add_argument('--MSA_weights_folder', default=None, type=str, help='Path to MSA weights for neighborhood scoring')
    parser.add_argument('--clustal_omega_location', default=None, type=str, help='Path to Clustal Omega (only needed with scoring indels with retrieval)')

    parser.add_argument('--EVE_model_folder', type=str, help='Path to folder containing the EVE model(s)')
    parser.add_argument('--EVE_seeds', nargs='*', help='Seeds of the EVE model(s) to be leveraged')
    parser.add_argument('--EVE_num_samples_log_proba', default=10, type=int, help='Number of samples to compute the EVE log proba')
    parser.add_argument('--EVE_model_parameters_location', default=None, type=str, help='Path to EVE model parameters')
    parser.add_argument('--MSA_recalibrate_probas', action='store_true', help='Whether to normalize EVE & MSA log probas (matching temp. of Transformer)')
    parser.add_argument('--EVE_recalibrate_probas', action='store_true', help='Whether to normalize EVE & MSA log probas (matching temp. of Transformer)')
    parser.add_argument('--clinvar_scoring', action='store_true', help='Tweaks when scoring ClinVar input file')
    args = parser.parse_args()
    print(args)
    model_name = args.checkpoint.split("/")[-1]
    
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=dir_path+os.sep+"trancepteve/utils/tokenizers/Basic_tokenizer",
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
        UniProt_ID = mapping_protein_seq_DMS["UniProt_ID"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0] if "UniProt_ID" in mapping_protein_seq_DMS else "No ID"
        if args.inference_time_retrieval_type is not None:
            MSA_data_file = args.MSA_folder + os.sep + mapping_protein_seq_DMS["MSA_filename"][args.DMS_index] if args.MSA_folder is not None else None
            MSA_weight_file_name = args.MSA_weights_folder + os.sep + mapping_protein_seq_DMS["weight_file_name"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0] if args.MSA_weights_folder else None
            MSA_start = int(mapping_protein_seq_DMS["MSA_start"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) - 1 # MSA_start typically based on 1-indexing
            MSA_end = int(mapping_protein_seq_DMS["MSA_end"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0])
            MSA_threshold_sequence_frac_gaps = float(mapping_protein_seq_DMS["MSA_threshold_sequence_frac_gaps"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) if "MSA_threshold_sequence_frac_gaps" in mapping_protein_seq_DMS else 0.5
            if args.clinvar_scoring:
                MSA_threshold_focus_cols_frac_gaps = float(mapping_protein_seq_DMS["MSA_threshold_focus_cols_frac_gaps"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) if "MSA_threshold_focus_cols_frac_gaps" in mapping_protein_seq_DMS else 1.0
            else:
                MSA_threshold_focus_cols_frac_gaps = float(mapping_protein_seq_DMS["MSA_threshold_focus_cols_frac_gaps"][mapping_protein_seq_DMS["DMS_id"]==DMS_id].values[0]) if "MSA_threshold_focus_cols_frac_gaps" in mapping_protein_seq_DMS else 1.0
            print("Sequence (fragment) gap threshold: "+str(MSA_threshold_sequence_frac_gaps))
            print("Focus column gap threshold: "+str(MSA_threshold_focus_cols_frac_gaps))
    else:
        target_seq=args.target_seq
        DMS_file_name=args.DMS_file_name
        DMS_id = DMS_file_name.split(".")[0]
        UniprotID = args.UniprotID
        if args.inference_time_retrieval_type is not None:
            MSA_data_file = args.MSA_folder + os.sep + args.MSA_filename if args.MSA_folder is not None else None
            MSA_weight_file_name = args.MSA_weights_folder + os.sep + args.MSA_weight_file_name if args.MSA_weights_folder is not None else None
            MSA_start = args.MSA_start - 1 # MSA_start based on 1-indexing
            MSA_end = args.MSA_end
            MSA_threshold_sequence_frac_gaps=args.MSA_threshold_sequence_frac_gaps
            MSA_threshold_focus_cols_frac_gaps=args.MSA_threshold_focus_cols_frac_gaps
            
    config = json.load(open(args.checkpoint+os.sep+'config.json'))
    config = trancepteve.config.TranceptEVEConfig(**config)
    config.attention_mode="tranception"
    config.position_embedding="grouped_alibi"
    config.tokenizer = tokenizer
    config.full_target_seq = target_seq
    config.scoring_window = args.scoring_window

    if args.inference_time_retrieval_type is not None:
        config.inference_time_retrieval_type = args.inference_time_retrieval_type
        config.retrieval_aggregation_mode = "aggregate_indel" if args.indel_mode else "aggregate_substitution"
        config.MSA_filename = MSA_data_file
        config.MSA_weight_file_name = MSA_weight_file_name
        config.MSA_start = MSA_start
        config.MSA_end = MSA_end
        config.MSA_threshold_sequence_frac_gaps = MSA_threshold_sequence_frac_gaps
        config.MSA_threshold_focus_cols_frac_gaps = MSA_threshold_focus_cols_frac_gaps
        config.retrieval_weights_manual = args.retrieval_weights_manual
        config.retrieval_inference_MSA_weight = args.retrieval_inference_MSA_weight
        config.retrieval_inference_EVE_weight = args.retrieval_inference_EVE_weight

        if "TranceptEVE" in args.inference_time_retrieval_type:
            EVE_model_paths = []
            EVE_seeds = args.EVE_seeds 
            num_seeds = len(EVE_seeds)
            print("Number of distinct EVE models to be leveraged: {}".format(num_seeds))
            for seed in EVE_seeds:
                print(f"{args.EVE_model_folder}/{os.path.basename(MSA_data_file.split('.a2m')[0])}_seed_{seed}")
                if os.path.exists(f"{args.EVE_model_folder}/{os.path.basename(MSA_data_file.split('.a2m')[0])}_seed_{seed}"):
                    EVE_model_name = f"{os.path.basename(MSA_data_file.split('.a2m')[0])}_seed_{seed}"
                elif os.path.exists(f"{args.EVE_model_folder}/{UniProt_ID}_seed_{seed}"):
                    EVE_model_name = f"{UniProt_ID}_seed_{seed}"
                else:
                    print(f"No EVE Model available for {MSA_data_file} with random seed {seed} in {args.EVE_model_folder}. Exiting")
                    exit(1)
                    
                EVE_model_paths.append(args.EVE_model_folder + os.sep + EVE_model_name)
            config.EVE_model_paths = EVE_model_paths
            config.EVE_num_samples_log_proba = args.EVE_num_samples_log_proba
            config.EVE_model_parameters_location = args.EVE_model_parameters_location
            config.MSA_recalibrate_probas = args.MSA_recalibrate_probas
            config.EVE_recalibrate_probas = args.EVE_recalibrate_probas
        else:
            num_seeds=0
        if args.indel_mode:
            config.clustal_omega_location = args.clustal_omega_location
    else:
        config.inference_time_retrieval_type = None
        config.retrieval_aggregation_mode = None
        
    if args.model_framework=="pytorch":
        model = trancepteve.model_pytorch.TrancepteveLMHeadModel.from_pretrained(pretrained_model_name_or_path=args.checkpoint,config=config)
        if torch.cuda.is_available():
            model.cuda()
    model.eval()
    
    if not os.path.isdir(args.output_scores_folder):
        os.mkdir(args.output_scores_folder)
    mutation_type = '_indels' if args.indel_mode else '_substitutions'
    mirror_type = '_no_mirror' if args.deactivate_scoring_mirror else ''
    normalization = '_norm-EVE' if args.EVE_recalibrate_probas else ''
    normalization = normalization + '_norm-MSA' if args.MSA_recalibrate_probas else normalization
    retrieval_weights = '_MSA-' + str(args.retrieval_inference_MSA_weight) +'_EVE-'+ str(args.retrieval_inference_EVE_weight) if args.retrieval_weights_manual else ''
    retrieval_type = ('_retrieval_' + args.inference_time_retrieval_type + retrieval_weights + '_' + str(num_seeds) + '-EVE-models' + normalization) if args.inference_time_retrieval_type is not None else '_no_retrieval'
    scoring_filename = args.output_scores_folder
    if not os.path.isdir(scoring_filename):
        os.mkdir(scoring_filename)
    scoring_filename += os.sep + DMS_id + '.csv'
    
    DMS_data = pd.read_csv(args.DMS_data_folder + os.sep + DMS_file_name, low_memory=False)
    with torch.no_grad():
        all_scores = model.score_mutants(
                                    DMS_data=DMS_data, 
                                    target_seq=target_seq, 
                                    scoring_mirror=not args.deactivate_scoring_mirror, 
                                    batch_size_inference=args.batch_size_inference,  
                                    num_workers=args.num_workers, 
                                    indel_mode=args.indel_mode
                                    )
    if len(all_scores) > 0 and args.clinvar_scoring: all_scores = pd.merge(all_scores,DMS_data,how="left",on="mutant")
    all_scores.to_csv(scoring_filename, index=False)
    
    if args.clinvar_scoring:
        experiment_name = args.output_scores_folder.split('/')[-1]
        print(experiment_name)
        
        with open("ClinVar_scoring_Tranception_20221130",'a+') as filew:
            if os.stat("ClinVar_scoring_Tranception_20221130").st_size == 0:
                header = "DMS_id,num_mutants_scored,num_mutants_scored_no_na,processed_MSA_depth,retrieval_inference_MSA_weight,retrieval_inference_EVE_weight\n"
                filew.write(header)
            all_scores_no_na = all_scores.dropna()
            stat = ",".join([str(x) for x in [DMS_id,len(all_scores),len(all_scores_no_na),model.MSA_processed_depth,model.EVE_processed_depth,model.retrieval_inference_MSA_weight,model.retrieval_inference_EVE_weight]])
            filew.write(stat+"\n")
    else:
        with open("TranceptEVE_aggregation_coefficients_log",'a+') as filew:
            if os.stat("TranceptEVE_aggregation_coefficients_log").st_size == 0:
                header = "DMS_id,num_mutants_scored,num_mutants_scored_no_na,processed_MSA_depth,retrieval_inference_MSA_weight,retrieval_inference_EVE_weight\n"
                filew.write(header)
            all_scores_no_na = all_scores.dropna()
            stat = ",".join([str(x) for x in [DMS_id,len(all_scores),len(all_scores_no_na),model.MSA_processed_depth,model.EVE_processed_depth,model.retrieval_inference_MSA_weight,model.retrieval_inference_EVE_weight]])
            filew.write(stat+"\n")

if __name__ == '__main__':
    main()