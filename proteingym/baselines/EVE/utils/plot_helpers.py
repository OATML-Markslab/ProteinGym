import os
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_histograms(all_evol_indices, dict_models, dict_pathogenic_cluster_index, protein_GMM_weight, plot_location, output_eve_scores_filename_suffix, protein_list):
    x = np.linspace(-10, 20, 2000)
    logprob = dict_models['main'].score_samples(x.reshape(-1,1))
    pdf = np.exp(logprob)
    component_share = dict_models['main'].predict_proba(x.reshape(-1, 1))
    pdf_pathogenic = component_share[:,dict_pathogenic_cluster_index['main']] * pdf 
    pdf_benign = component_share[:,1 - dict_pathogenic_cluster_index['main']] * pdf 
    plt.plot(x,pdf, '--k', color='black')
    plt.plot(x,pdf_pathogenic, '--k', color = 'xkcd:red',linewidth=4)
    plt.plot(x,pdf_benign, '--k', color = 'xkcd:sky blue',linewidth=4)
    plt.hist(all_evol_indices['evol_indices'], color = 'xkcd:grey', bins = 80, histtype='stepfilled', alpha=0.4, density=True)
    plt.xlabel("Evolutionary index", fontsize=13)
    plt.ylabel("% of variants", fontsize=13)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(plot_location+os.sep+'histogram_random_samples_'+str(output_eve_scores_filename_suffix)+"_all.png", dpi=800, bbox_inches='tight')
    plt.clf()
    if protein_GMM_weight > 0.0:
        for protein in tqdm.tqdm(protein_list,"Plot protein histograms"):
            x = np.linspace(-10, 20, 2000)
            logprob = dict_models[protein].score_samples(x.reshape(-1,1))
            pdf = np.exp(logprob)
            component_share = dict_models[protein].predict_proba(x.reshape(-1, 1))
            pdf_pathogenic = component_share[:,dict_pathogenic_cluster_index[protein]] * pdf
            pdf_benign = component_share[:, 1 - dict_pathogenic_cluster_index[protein]] * pdf
            plt.plot(x,pdf, '--k', color='black')
            plt.plot(x,pdf_pathogenic, '--k', color = 'xkcd:red',linewidth=4)
            plt.plot(x,pdf_benign, '--k', color = 'xkcd:sky blue',linewidth=4)
            plt.hist(all_evol_indices['evol_indices'][all_evol_indices['protein_name']==protein], color = 'xkcd:grey', bins = 80, histtype='stepfilled', alpha=0.4, density=True)
            plt.xlabel("Evolutionary index", fontsize=13)
            plt.ylabel("% of variants", fontsize=13)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.savefig(plot_location+os.sep+'histogram_random_samples_'+str(output_eve_scores_filename_suffix)+"_"+str(protein)+".png", dpi=800, bbox_inches='tight')
            plt.clf()

def plot_scores_vs_labels(score_df, plot_location, output_eve_scores_filename_suffix, mutation_name='mutations', score_name="EVE_scores", label_name='labels'):
    score_df_local = score_df.copy()
    score_df_local = score_df_local[score_df_local[mutation_name] !='w-1t'] #Remove wild type sequence
    score_df_local['mutation_position'] = score_df[mutation_name].map(lambda x: int(x[1:-1]))
    labels = score_df_local[label_name]
    pathogenic = plt.scatter(x=score_df_local['mutation_position'][labels==1], y=score_df_local[score_name][labels==1], color='xkcd:red')
    benign = plt.scatter(x=score_df_local['mutation_position'][labels==0], y=score_df_local[score_name][labels==0], color='xkcd:sky blue')
    plt.legend([pathogenic,benign],['pathogenic','benign'])
    plt.savefig(plot_location+os.sep+'scores_vs_labels_plots_'+str(output_eve_scores_filename_suffix)+".png", dpi=400, bbox_inches='tight')
    plt.clf()