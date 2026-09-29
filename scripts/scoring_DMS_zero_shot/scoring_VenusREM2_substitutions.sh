source ../zero_shot_config.sh
source activate prosst

# VenusREM2 = vrh calibration on the official ProSST ensemble
# (K = 20, 128, 512, 1024, 2048, 4096), then a per-protein z-mean.
# Checkpoints: https://huggingface.co/AI4Protein
#
# Expected layout under DMS_folder (see proteingym/baselines/venusrem2/README.md):
#   aa_seq/ or residue_sequence/
#   aa_seq_aln_a2m/          ProteinGym DMS MSAs (a2m)
#   substitutions/           ProteinGym substitution CSVs
# Structure tokens (struc_seq/{K}/) are read if present. Missing tokens are
# built from the PDB directory below with VenusREM's PdbQuantizer.
# That PDB directory is also used for RSA / pLDDT.
export DMS_folder="Path to unzipped sequence, structure-token, MSA, and mutant folder"
export DMS_output_score_folder="${DMS_output_score_folder_subs}/VenusREM2"

python ../../proteingym/baselines/venusrem2/compute_fitness.py \
    --base_dir ${DMS_folder} \
    --pdb_dir ${DMS_structure_folder} \
    --output_scores_folder ${DMS_output_score_folder}
