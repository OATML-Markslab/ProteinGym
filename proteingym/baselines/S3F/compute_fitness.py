import os
import sys
import pprint
import random
import pickle

import numpy as np
import pandas as pd

import torch

from torchdrug import core, utils, data
from torchdrug.utils import comm, pretty

sys.path.append(os.path.dirname(__file__))
import sys
print(sys.path)

import util
from s3f import dataset
from s3f.model import FusionNetwork
from script.evaluate import evaluate, graph_concat, predict, get_prob, load_dataset
from proteingym.utils.scoring_utils import standardize

METRICS = ["spearmanr", "pearsonr", "mae", "rmse"]

def compute_fitness():
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    cfg.summary.csv_file = args.ProteinGym_reference_file 
    working_dir = util.create_working_directory(cfg)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Output dir: %s" % working_dir)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    summary = core.Configurable.load_config_dict(cfg.summary)
    num_assay = len(summary.ids)
    num_mutant = [int(summary.assay_dict[id]["DMS_number_single_mutants"]) for id in summary.ids]
    total_mutant = sum(num_mutant)
    if comm.get_rank() == 0:
        logger.warning("# total assays: %d, # total mutations: %d" % (num_assay, total_mutant))

    # pick the subset of DMS assays to be evaluated
    id_list = cfg.get("id_list", summary.ids)
    model_name = os.path.splitext(os.path.basename(args.config))[0]
    if args.DMS_index: #We only score a single DMS, referenced by its index 
        DMS_id = id_list[args.DMS_index]
        print(f"Scoring {DMS_id} with model {model_name}")
        id_list = [DMS_id]
    if "exclude_id_list" in cfg:
        exclude_id_list = set(cfg.exclude_id_list)
        id_list = [id for id in id_list if id not in exclude_id_list]

    if comm.get_rank() == 0:
        num_mutant = [int(summary.assay_dict[id]["DMS_number_single_mutants"]) for id in id_list]
        logger.warning("# assays: %d, # mutations: %d" % (len(id_list), sum(num_mutant)))

    task = core.Configurable.load_config_dict(cfg.task)
    task.preprocess(None, None, None)

    with open("results.csv", "w") as f:
        f.write("DMS_id,UniProt_ID,seq_len,DMS_number_single_mutants,%s\n" % (",".join(METRICS)))

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))['model']
        task.load_state_dict(model_dict)
    
    assay_result = {}
    for i, id in enumerate(id_list):
        DMS_filename = summary.assay_dict[id]["DMS_filename"]
        output_location = os.path.join(args.output_scores_folder,DMS_filename)
        if comm.get_rank() == 0:
            if os.path.exists(output_location):
                print(f"Assay {DMS_filename} already computed. Skipping.")
                continue
            os.makedirs(os.path.dirname(output_location), exist_ok=True)
            logger.warning(pretty.separator)
            logger.warning("Start evaluation on DMS assay %s" % id)
        
        # wild-type sequence
        sequence = summary.assay_dict[id]["target_seq"]
        protein = data.Protein.from_sequence(sequence, atom_feature=None, bond_feature=None)
        protein.view = "residue"

        # wild-type structure
        wild_types = []
        for _pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
            pdb_file = os.path.join(os.path.expanduser(cfg.structure_path), _pdb_file)
            wild_type = dataset.bio_load_pdb(pdb_file)[0]
            ca_index = wild_type.atom_name == wild_type.atom_name2id["CA"]
            wild_type = wild_type.subgraph(ca_index)
            wild_types.append(wild_type)
        wild_type = graph_concat(wild_types)
        wild_type.view = "residue"

        # pdb range
        pdb_range = summary.assay_dict[id]["pdb_range"].split("-")
        start, end = int(pdb_range[0])-1, int(pdb_range[-1])
        if id=="POLG_HCVJF_Qi_2014": start, end = 1981, 2225 
        with wild_type.graph():
            wild_type.start = torch.as_tensor(start)
            wild_type.end = torch.as_tensor(end)

        # surface graph
        if cfg.get("surface_path"):
            surf_graphs = []
            res2surfs = []
            for pdb_file in summary.assay_dict[id]["pdb_file"].split("|"):
                surf_graph_file = pdb_file.split(".")[0] + ".pkl"
                surf_graph_file = os.path.join(os.path.expanduser(cfg.surface_path), surf_graph_file)
                with open(surf_graph_file, "rb") as fin:
                    surf_dict = pickle.load(fin)
                surf_graph = dataset.load_surface(surf_dict)
                surf_graphs.append(surf_graph)
                res2surfs.append(torch.as_tensor(surf_dict["res2surf"]))
            surf_graph = graph_concat(surf_graphs)
            res2surf = torch.cat(res2surfs, dim=0)
            with wild_type.residue():
                wild_type.res2surf = res2surf
        else:
            surf_graph = None

        # mutants
        csv_file = os.path.join(summary.path, DMS_filename)
        masked_sequences, mutations, offsets = load_dataset(csv_file, protein, task)
        if comm.get_rank() == 0:
            logger.warning("Number of masked sequences: %d" % len(masked_sequences))
            logger.warning("Number of mutations: %d" % len(mutations))

        _dataset = dataset.MutantDataset(masked_sequences, wild_type, surf_graph=surf_graph)
        seq_prob = predict(cfg, task, _dataset)
        pred, target = get_prob(seq_prob, mutations, offsets)
        
        if args.MSA_retrieval_location:  # If we specify a path to EVE predictions we also compute S2F-MSA / S3F-MSA
            EVE_prediction_filename = os.path.join(args.MSA_retrieval_location, DMS_filename)
            EVE_scores = pd.read_csv(EVE_prediction_filename)
            
            # Create a dictionary for fast lookup of mutations
            mutation_dict = {':'.join(mut[1]): i for i, mut in enumerate(mutations)}
            
            # Create a dictionary of EVE scores for fast lookup
            eve_dict = {row['mutant']: row['EVE_ensemble'] for _, row in EVE_scores.iterrows()}
            
            # Find common mutations and their indices in one pass
            common_indices = []
            common_eve_scores = []
            
            for mutant, eve_score in eve_dict.items():
                if mutant in mutation_dict:
                    common_indices.append(mutation_dict[mutant])
                    common_eve_scores.append(eve_score)
            
            # Only operate on common mutations
            if len(common_indices) < len(mutations):
                logger.warning(f"Warning: Only {len(common_indices)} out of {len(mutations)} mutations were matched with EVE scores")
            
            # Get tensors for the filtered mutations in one go
            idx_tensor = torch.tensor(common_indices, device=pred.device)
            filtered_pred = pred[idx_tensor]
            filtered_target = target[idx_tensor]
            eve_scores_tensor = torch.tensor(common_eve_scores, device=pred.device)
            
            # Standardize scores
            pred_mean = filtered_pred.mean()
            pred_std = filtered_pred.std()
            eve_mean = eve_scores_tensor.mean()
            eve_std = eve_scores_tensor.std()
            
            standardized_pred = (filtered_pred - pred_mean) / pred_std
            standardized_eve = (eve_scores_tensor - eve_mean) / eve_std
            
            # Combine scores
            pred_MSA = (standardized_pred + standardized_eve) / 2.0
            
            # Update mutations list to only include common mutations
            filtered_mutations = [mutations[i] for i in common_indices]
            
            # Update the values
            pred = filtered_pred
            target = filtered_target
            mutations = filtered_mutations
            
            result_MSA = evaluate(pred_MSA, target)
        else:
            pred_MSA = None

        result = evaluate(pred, target)
        if args.MSA_retrieval_location: result_MSA = evaluate(pred_MSA, target)

        if comm.get_rank() == 0:
            with open(output_location, "w") as f:
                scoring_file_header = f",mutant,mutated_sequence,DMS_score,{model_name}_score"
                if args.MSA_retrieval_location: 
                    scoring_file_header += f",{model_name}_MSA_score"
                f.write(scoring_file_header+"\n")
                pred_np = pred.cpu().numpy()
                target_np = target.cpu().numpy()
                for i in range(len(mutations)):
                    line_to_write = f",{':'.join(mutations[i][1])},,{target_np[i]:.6f},{pred_np[i]:.6f}"
                    if args.MSA_retrieval_location:
                        line_to_write += f",{pred_MSA[i].cpu().numpy():.6f}"
                    f.write(line_to_write+"\n")
            logger.warning(pretty.separator)
            logger.warning("Test results")
            logger.warning(pretty.line)
            logger.warning(pprint.pformat(result))
            
            util.write_aggregate_performance(args.output_scores_folder, id, summary, result, METRICS, result_filename=f"results_{model_name}.csv")
            util.write_aggregate_performance(args.output_scores_folder, id, summary, result_MSA, METRICS, result_filename=f"results_{model_name}_MSA.csv")
            
        assay_result[id] = result

    if comm.get_rank() == 0:
        logger.warning(pretty.separator)
        logger.warning("Average results on all assays")
        logger.warning(pretty.line)
        logger.warning(pprint.pformat(utils.mean(utils.stack(list(assay_result.values())))))

if __name__ == "__main__":
    compute_fitness()