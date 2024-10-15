import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from utils.metric_utils import spearman, topk_mean, hit_rate, aucroc


def get_stratified_metrics(df, model_name, max_n_mut, metric_fn):
    strat_metrics = np.zeros(max_n_mut+1)
    for i in range(0, max_n_mut+1):
        # 0th position is overall aggregated metric
        tmp = df
        if i > 0:
            tmp = df[df.n_mut == i]
        strat_metrics[i] = metric_fn(tmp[model_name], tmp.log_fitness)
    return strat_metrics


def plot_stratified_metrics(ax, df, models, max_n_mut, metric_fn, vmin, vmax):
    strat_matrix = np.zeros((len(models), 1+max_n_mut))
    xticklabels=['All'] + list(range(1, max_n_mut+1))
    for i, m in enumerate(models):
        strat_matrix[i] = get_stratified_metrics(df, m, max_n_mut, metric_fn)    
    sns.heatmap(strat_matrix, yticklabels=models, xticklabels=xticklabels,
                vmin=vmin, vmax=vmax, ax=ax, cmap='viridis')
    ax.set_xlabel('# Mutations')
    ax.vlines([1], *ax.get_ylim(), colors='black')
    

def plot_auc_and_corr(df, models, functional_threshold, wt_log_fitness,
        max_n_mut=5, vmin=[None, None, None], vmax=[None, None, None], topk=96):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

    ax = axes[0]
    fn = partial(aucroc, y_cutoff=functional_threshold)
    plot_stratified_metrics(ax, df, models, max_n_mut, fn, vmin[0], vmax[0])
    ax.set_title(f'Functional vs Non-Functional AUC-ROC')

    ax = axes[1]
    fn = partial(aucroc, y_cutoff=wt_log_fitness)
    plot_stratified_metrics(ax, df[df.log_fitness >= functional_threshold],
            models, max_n_mut, fn, vmin[1], vmax[1])
    ax.set_title(f'Functional, <WT vs >=WT AUC-ROC')
    
    ax = axes[2]
    plot_stratified_metrics(ax, df[df.log_fitness >= functional_threshold],
            models, max_n_mut, spearman, vmin[2], vmax[2])
    ax.set_title('Rank Correlation (Functional)')

    fig.suptitle('Model performance, stratified by # mutations')
    plt.subplots_adjust(wspace=0.1, top=0.85)
    plt.show()