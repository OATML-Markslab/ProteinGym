import pandas as pd 
import os

individual_files = '/n/groups/marks/projects/marks_lab_and_oatml/ProteinGym/model_scores/zero_shot_substitutions/EVE' #'/n/groups/marks/projects/marks_lab_and_oatml/protein_transformer/model_scores/EVE_all_mutants/20220807_EVE_models_all_mutants'
folder_name = '/n/groups/marks/projects/marks_lab_and_oatml/ProteinGym/model_scores/zero_shot_substitutions/EVE/merged'

#mapping = "/home/pn73/protein_transformer/utils/mapping_files/ProteinGym_reference_file_substitutions_20220227.csv"
mapping='/home/pn73/ProteinGym/ProteinGym_reference_file_substitutions.csv'
mapping = pd.read_csv(mapping)

#exclude=[
#'A4D664_9INFA_Soh_CCL141_2019',
#'CAPSD_AAV2S_Sinai_substitutions_2021',
#'ENV_HV1B9_DuenasDecamp_2016',
#'I6TAH8_I68A0_Doud_2015',
#'NCAP_I34A1_Doud_2015',
#'NRAM_I33A0_Jiang_standard_2016',
#'P53_HUMAN_Kotler_2018',
#'PA_I34A1_Wu_2015',
#'R1AB_SARS2_Flynn_growth_2022',
#'SPG1_STRSG_Olson_2014',
#'SPIKE_SARS2_Starr_bind_2020',
#'SPIKE_SARS2_Starr_expr_2020'
#]

for DMS_id in mapping["DMS_id"][87:100]:
    print("DMS id : {}".format(DMS_id))
    #if DMS_id in exclude:
    #    print("Exclude : {}".format(DMS_id))
    #    continue
    
    evol_seed = {}
    for i in [1000,2000,3000,4000,5000]:
        #evol_seed[i] = pd.read_csv(individual_files+os.sep+"seed_"+str(i)+os.sep+DMS_id+'_20000_samples_Aug7_seed_'+str(i)+'.csv')
        evol_seed[i] = pd.read_csv(individual_files+os.sep+DMS_id+'_20000_samples_Aug7_seed_'+str(i)+'.csv')
        evol_seed[i] = evol_seed[i].groupby(['mutant']).mean().reset_index()

    evol_ensemble = evol_seed[1000]
    for i in [2000,3000,4000,5000]:
        evol_ensemble = pd.merge(evol_ensemble,evol_seed[i], on='mutant', how='left', suffixes=("_seed_"+str(i-1000),"_seed_"+str(i)))
    evol_ensemble['evol_indices_seed_5000'] = evol_ensemble['evol_indices']
    
    evol_ensemble['evol_indices_ensemble']=0
    for i in [1000,2000,3000,4000,5000]:
        evol_ensemble['evol_indices_ensemble'] += evol_ensemble["evol_indices_seed_"+str(i)] / 5.0

    final_list = ['mutant']+['evol_indices_seed_'+str(i) for i in [1000,2000,3000,4000,5000]]+['evol_indices_ensemble']
    evol_ensemble = evol_ensemble[final_list]

    evol_ensemble.to_csv(folder_name+os.sep+DMS_id+'.csv', index=False)