import pandas as pd
import numpy as np
from trancepteve.utils import scoring_utils

def DMS_file_cleanup(DMS_filename, target_seq, start_idx=1, end_idx=None, DMS_mutant_column='mutant', DMS_phenotype_name='score', DMS_directionality=1, AA_vocab = "ACDEFGHIKLMNPQRSTVWY"):
    """
    Function to process the raw substitution DMS assay data (eg., removing invalid mutants, aggregate silent mutations).
    """
    DMS_data = pd.read_csv(DMS_filename, low_memory=False)
    end_idx = start_idx + len(target_seq) - 1 if end_idx is None else end_idx
    DMS_data['mutant'] = DMS_data[DMS_mutant_column]
    
    DMS_data=DMS_data[DMS_data['mutant'].notnull()].copy()
    DMS_data=DMS_data[DMS_data['mutant'].apply(lambda x: all([len(y)>=3 for y in x.split(":")]))].copy() #Mutant triplets should have at least 3 or more characters
    DMS_data=DMS_data[DMS_data['mutant'].apply(lambda x: all([(y[0] in AA_vocab) and (y[1:-1].isnumeric()) and (y[-1] in AA_vocab) for y in x.split(":")]))].copy()
    DMS_data=DMS_data[DMS_data['mutant'].apply(lambda x: all([int(y[1:-1])-start_idx >=0 and int(y[1:-1]) <= end_idx for y in x.split(":")]))].copy()
    DMS_data=DMS_data[DMS_data['mutant'].apply(lambda x: all([y[0]==target_seq[int(y[1:-1])-start_idx] for y in x.split(":")]))].copy()
    
    DMS_data[DMS_phenotype_name]=pd.to_numeric(DMS_data[DMS_phenotype_name],errors='coerce')
    DMS_data=DMS_data[np.isfinite(DMS_data[DMS_phenotype_name])]
    DMS_data.dropna(subset = [DMS_phenotype_name], inplace=True)
    DMS_data['DMS_score'] = DMS_data[DMS_phenotype_name] * DMS_directionality
    DMS_data=DMS_data[['mutant','DMS_score']]
    DMS_data=DMS_data.groupby('mutant').mean().reset_index()

    DMS_data['mutated_sequence'] = DMS_data['mutant'].apply(lambda x: scoring_utils.get_mutated_sequence(target_seq, x))
    DMS_data=DMS_data[['mutant','mutated_sequence','DMS_score']]
    
    return DMS_data

