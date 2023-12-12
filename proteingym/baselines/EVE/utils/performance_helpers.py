from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def compute_stats(input_array):
    return {
        'mean':input_array.mean(),
        'std':input_array.std(),
        'min':input_array.min(),
        'max':input_array.max(),
        'P1':np.percentile(input_array,1),
        'P5':np.percentile(input_array,5),
        'P10':np.percentile(input_array,10),
        'P25':np.percentile(input_array,25),
        'P33':np.percentile(input_array,33),
        'P40':np.percentile(input_array,40),
        'P45':np.percentile(input_array,45),
        'P50':np.median(input_array),
        'P55':np.percentile(input_array,55),
        'P60':np.percentile(input_array,60),
        'P66':np.percentile(input_array,66),
        'P75':np.percentile(input_array,75),
        'P90':np.percentile(input_array,90),
        'P95':np.percentile(input_array,95),
        'P99':np.percentile(input_array,99)
}

def compute_accuracy_with_uncertain(class_pred, labels):
    temp_df = pd.DataFrame({'class_pred': class_pred.copy(),'labels': labels.copy()})
    initial_num_obs = len(temp_df['labels'])
    temp_df=temp_df[temp_df['class_pred'] != 'Uncertain']
    filtered_num_obs = len(temp_df['labels'])
    temp_df['class_pred_bin'] = temp_df['class_pred'].map(lambda x: 1 if x == 'Pathogenic' else 0)
    correct_classification = (temp_df['class_pred_bin'] == temp_df['labels']).astype(int)
    accuracy = round(correct_classification.mean()*100,1)
    pct_mutations_kept = round(filtered_num_obs/float(initial_num_obs)*100,1)
    return accuracy, pct_mutations_kept

def compute_AUC_overall_with_uncertain(scores, class_pred, labels):
    temp_df = pd.DataFrame({'class_pred': class_pred.copy(),'labels': labels.copy(), 'scores': scores.copy()})
    temp_df=temp_df[temp_df['class_pred'] != 'Uncertain']
    AUC = roc_auc_score(y_true=temp_df['labels'], y_score=temp_df['scores'])
    return round(AUC*100,1)

def compute_avg_protein_level_AUC_with_uncertain(scores, class_pred, labels, protein_ID):
    temp_df = pd.DataFrame({'class_pred': class_pred.copy(),'labels': labels.copy(), 'scores': scores.copy(), 'protein_ID': protein_ID.copy()})
    temp_df=temp_df[temp_df['class_pred'] != 'Uncertain']
    def compute_auc_group(group):
        protein_scores = group['scores']
        protein_labels = group['labels']
        try: 
            result = roc_auc_score(y_true=protein_labels, y_score=protein_scores)
        except: 
            result = np.nan
        return result
    protein_level_AUC = temp_df.groupby('protein_ID').apply(compute_auc_group)
    avg_AUC = protein_level_AUC.mean(skipna=True)
    return round(avg_AUC*100,1)

def compute_pathogenic_rate_with_uncertain(class_pred, labels):
    temp_df = pd.DataFrame({'class_pred': class_pred.copy(),'labels': labels.copy()})
    temp_df=temp_df[temp_df['class_pred'] != 'Uncertain']
    rate = len(temp_df[temp_df['class_pred'] == 'Pathogenic']) / float(len(temp_df))
    return round(rate*100,1)

def compute_uncertainty_deciles(score_dataframe, score_name="EVE_scores", uncertainty_name='uncertainty', suffix=''):
    uncertainty_deciles_name='uncertainty_deciles'+suffix
    score_dataframe[uncertainty_deciles_name] = pd.qcut(score_dataframe[uncertainty_name], q=10, labels=range(1,11)).astype(int)
    uncertainty_cutoffs_deciles={}
    scores_at_uncertainty_deciles_cuttoffs_UB_lower_part={}
    scores_at_uncertainty_deciles_cuttoffs_LB_upper_part={}
    for decile in range(1,11):
        uncertainty_cutoffs_deciles[str(decile)]= np.max(score_dataframe[uncertainty_name][score_dataframe[uncertainty_deciles_name] == decile])
        scores_at_uncertainty_deciles_cuttoffs_UB_lower_part[str(decile)]= np.max(score_dataframe[score_name][(score_dataframe[uncertainty_deciles_name] == decile) & (score_dataframe[score_name] < 0.5)])
        scores_at_uncertainty_deciles_cuttoffs_LB_upper_part[str(decile)]= np.min(score_dataframe[score_name][(score_dataframe[uncertainty_deciles_name] == decile) & (score_dataframe[score_name] > 0.5)])
    return uncertainty_cutoffs_deciles, scores_at_uncertainty_deciles_cuttoffs_UB_lower_part, scores_at_uncertainty_deciles_cuttoffs_LB_upper_part

def compute_uncertainty_quartiles(score_dataframe, score_name="EVE_scores", uncertainty_name='uncertainty', suffix=''):
    uncertainty_deciles_name='uncertainty_quartiles'+suffix
    score_dataframe[uncertainty_deciles_name] = pd.qcut(score_dataframe[uncertainty_name], q=4, labels=range(1,5)).astype(int)
    uncertainty_cutoffs_quartiles={}
    scores_at_uncertainty_quartiles_cuttoffs_UB_lower_part={}
    scores_at_uncertainty_quartiles_cuttoffs_LB_upper_part={}
    for quartile in range(1,5):
        uncertainty_cutoffs_quartiles[str(quartile)]= np.max(score_dataframe[uncertainty_name][score_dataframe[uncertainty_deciles_name] == quartile])
        scores_at_uncertainty_quartiles_cuttoffs_UB_lower_part[str(quartile)]= np.max(score_dataframe[score_name][(score_dataframe[uncertainty_deciles_name] == quartile) & (score_dataframe[score_name] < 0.5)])
        scores_at_uncertainty_quartiles_cuttoffs_LB_upper_part[str(quartile)]= np.min(score_dataframe[score_name][(score_dataframe[uncertainty_deciles_name] == quartile) & (score_dataframe[score_name] > 0.5)])
    return uncertainty_cutoffs_quartiles, scores_at_uncertainty_quartiles_cuttoffs_UB_lower_part, scores_at_uncertainty_quartiles_cuttoffs_LB_upper_part
    
def compute_performance_by_uncertainty_decile(score_dataframe, metric="Accuracy", verbose=False, score_name="EVE_scores", uncertainty_name="uncertainty", label_name='ClinVar_labels', protein_name='protein_name', class_100pct_retained_name='EVE_classes_100_pct_retained', suffix=''):
    uncertainty_cutoffs_deciles, scores_at_uncertainty_deciles_cuttoffs_UB_lower_part, scores_at_uncertainty_deciles_cuttoffs_LB_upper_part = compute_uncertainty_deciles(score_dataframe, score_name, uncertainty_name, suffix)
    performance_by_uncertainty_deciles = {}
    pathogenic_rate_by_uncertainty_deciles = {}
    for decile in range(1,11):
        classification_name = 'class_pred_removing_'+str((10-decile)*10)+"_pct_most_uncertain"+suffix
        score_dataframe[classification_name] = score_dataframe[class_100pct_retained_name]
        score_dataframe.loc[score_dataframe['uncertainty_deciles'+suffix] > decile, classification_name] = 'Uncertain'
        if metric=="Accuracy":
            performance_decile = compute_accuracy_with_uncertain(score_dataframe[classification_name], score_dataframe[label_name])[0]
        elif metric =="Avg_AUC":
            performance_decile = compute_avg_protein_level_AUC_with_uncertain(scores=score_dataframe[score_name], class_pred=score_dataframe[classification_name], labels=score_dataframe[label_name], protein_ID=score_dataframe[protein_name])
        performance_by_uncertainty_deciles[decile] = performance_decile
        pathogenic_rate_by_uncertainty_deciles[decile] = compute_pathogenic_rate_with_uncertain(class_pred=score_dataframe[classification_name], labels=score_dataframe[label_name])
        if verbose:
            print(str(metric)+" when dropping the "+str((10-decile)*10)+"% of cases with highest uncertainty:\t"+str(performance_by_uncertainty_deciles[decile])+"% \t with pathogenic rate of "+str(pathogenic_rate_by_uncertainty_deciles[decile])+"%\n")
            print("Uncertainty decile #"+str(decile)+" cutoff: "+str(uncertainty_cutoffs_deciles[str(decile)])+"\n")
            print("Score upper bound for lower part in uncertainty decile: "+str(scores_at_uncertainty_deciles_cuttoffs_UB_lower_part[str(decile)])+"\n")
            print("Score lower bound for higher part in uncertainty decile: "+str(scores_at_uncertainty_deciles_cuttoffs_LB_upper_part[str(decile)])+"\n")
    return performance_by_uncertainty_deciles, pathogenic_rate_by_uncertainty_deciles

def compute_performance_by_uncertainty_quartile(score_dataframe, metric="Accuracy", verbose=False, score_name="EVE_scores", uncertainty_name="uncertainty", label_name='ClinVar_labels', protein_name='protein_name', class_100pct_retained_name='EVE_classes_100_pct_retained', suffix=''):
    uncertainty_cutoffs_quartiles, scores_at_uncertainty_quartiles_cuttoffs_UB_lower_part, scores_at_uncertainty_quartiles_cuttoffs_LB_upper_part = compute_uncertainty_quartiles(score_dataframe, score_name, uncertainty_name, suffix)
    performance_by_uncertainty_quartiles = {}
    pathogenic_rate_by_uncertainty_quartiles = {}
    for quartile in range(1,5):
        classification_name = 'class_pred_removing_'+str((4-quartile)*25)+"_pct_most_uncertain"+suffix
        score_dataframe[classification_name] = score_dataframe[class_100pct_retained_name] 
        score_dataframe.loc[score_dataframe['uncertainty_quartiles'+suffix] > quartile, classification_name] = 'Uncertain'
        if metric=="Accuracy":
            performance_quartile = compute_accuracy_with_uncertain(score_dataframe[classification_name], score_dataframe[label_name])[0]
        elif metric =="Avg_AUC":
            performance_quartile = compute_avg_protein_level_AUC_with_uncertain(scores=score_dataframe[score_name], class_pred=score_dataframe[classification_name], labels=score_dataframe[label_name], protein_ID=score_dataframe[protein_name])
        performance_by_uncertainty_quartiles[quartile] = performance_quartile
        pathogenic_rate_by_uncertainty_quartiles[quartile] = compute_pathogenic_rate_with_uncertain(class_pred=score_dataframe[classification_name], labels=score_dataframe[label_name])
        if verbose:
            print(str(metric)+" when dropping the "+str((4-quartile)*25)+"% of cases with highest uncertainty:\t"+str(performance_by_uncertainty_quartiles[quartile])+"% \t with pathogenic rate of "+str(pathogenic_rate_by_uncertainty_quartiles[quartile])+"%\n")
            print("Uncertainty quartile #"+str(quartile)+" cutoff: "+str(uncertainty_cutoffs_quartiles[str(quartile)])+"\n")
            print("Score upper bound for lower part in uncertainty quartile: "+str(scores_at_uncertainty_quartiles_cuttoffs_UB_lower_part[str(quartile)])+"\n")
            print("Score lower bound for higher part in uncertainty quartile: "+str(scores_at_uncertainty_quartiles_cuttoffs_LB_upper_part[str(quartile)])+"\n")
    return performance_by_uncertainty_quartiles, pathogenic_rate_by_uncertainty_quartiles
    
def predictive_entropy_binary_classifier(class1_scores, eps=1e-8):
    class1_scores = pd.Series(class1_scores).map(lambda x: x - eps if x==1.0 else x + eps if x==0 else x)
    class0_scores = 1 - class1_scores
    return - np.array((np.log(class1_scores) * class1_scores + np.log(class0_scores) * class0_scores))

def compute_weighted_score_two_GMMs(X_pred, main_model, protein_model, cluster_index_main, cluster_index_protein, protein_weight):
    return protein_model.predict_proba(X_pred)[:,cluster_index_protein] * protein_weight + (main_model.predict_proba(X_pred)[:,cluster_index_main]) * (1 - protein_weight)

def compute_weighted_class_two_GMMs(X_pred, main_model, protein_model, cluster_index_main, cluster_index_protein, protein_weight):
    """By construct, 1 is always index of pathogenic, 0 always that of benign"""
    proba_pathogenic = protein_model.predict_proba(X_pred)[:,cluster_index_protein] * protein_weight + (main_model.predict_proba(X_pred)[:,cluster_index_main]) * (1 - protein_weight)
    return (proba_pathogenic > 0.5).astype(int)
