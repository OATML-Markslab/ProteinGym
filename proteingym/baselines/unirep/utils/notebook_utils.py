import os
from Bio import SeqIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import MultipleLocator, FixedLocator
import seaborn as sns

from utils.io_utils import load, read_fasta
from utils.data_utils import seq2effect

# EVMutation imports
from couplings_model import CouplingsModel


def add_unirep_model(df, model_names, path, name):
    # The inference for UniRep has sorted sequences.
    df = df.sort_values('seq')
    seqlen = len(df.seq.values[0])
    df[name] = -seqlen * load(path)
    model_names.append(name)
    return df, model_names


def add_ev_model(df, model_names, path, name, dataset, include_indep=False):
    wt = read_fasta(os.path.join('../data', dataset, 'wt.fasta'))[0]

    couplings_model = CouplingsModel(path)
    df[f'{name}'] = seq2effect(df.seq.values, couplings_model)
    model_names.append(name)
    
    if include_indep:
        indep_model = couplings_model.to_independent_model()
        df[f'{name}_indep'] = seq2effect(df.seq.values, wt, indep_model)
        model_names.append(f'{name}_indep')
    return df, model_names


def add_hmm_model(df, model_names, path, name, dataset):
    df = df.sort_values('seq')
    records = SeqIO.parse(os.path.join('../data', dataset, 'seqs.fasta'),
            'fasta')
    ids = []
    seqs = []
    for rec in records:
        seqs.append(str(rec.seq))
        ids.append(str(rec.id))
    id2seq = pd.Series(index=ids, data=seqs, name='seq')
    hmm_ll = pd.read_csv(path)[['target', 'score_full']]
    hmm_ll = hmm_ll.join(id2seq, on='target', how='left')
    hmm_ll = hmm_ll.drop_duplicates(subset='seq')
    df[name] = hmm_ll.sort_values('seq')['score_full'].values
    model_names.append(name)
    return df, model_names


metric_display_name = {
    'ndcg': 'NDCG',
    'topk_mean': 'Top 96 mean',
    'spearman': 'Spearman correlation',
}


def retrieve_metric(df, metric_name, n_mut=None, predictor=None):
    tmp = df
    if predictor is not None:
        if isinstance(predictor, str):
            predictor = [predictor]
        tmp = tmp.loc[tmp.predictor.apply(lambda x: x in predictor)]
    if n_mut is not None:
        metric_name = f'{metric_name}_{n_mut}mut'
    tmp = tmp[['predictor', 'n_train', metric_name]]
    return tmp.rename(columns={metric_name:'val'})


def metric_lineplot(df, predictors, metric, predictor_names, dataset_name,
    max_n_mut, savename='figure', legend=None, mutcounts=None, **kwargs):
    fig, axes = plt.subplots(1, max_n_mut+1,
                             figsize=((max_n_mut+1)*3, 4),
                             sharex=True, sharey=True)
    ax = axes[0]
    nmut_to_title = {
        1: 'Single mutants',
        2: 'Double mutants',
        3: 'Triple mutants',
        4: 'Quadruple mutants',
    }
    nmut_to_title.update({i: f'{i}th-order Mutants' for i in range(5, 11)})
    tmp = retrieve_metric(df, metric, n_mut=None, predictor=predictors)
    sns.lineplot(data=tmp, x='n_train', y='val',
                 hue='predictor', style='predictor', ax=ax,
                 hue_order=predictors, style_order=predictors, **kwargs)
    #ax.hlines(df[df.predictor == 'mutation'].mean().spearman, 48, 240, color='dimgrey')
    ax.set_title(f'mutants of all orders')
    ax.set_ylabel(metric_display_name[metric])
    ax.set_xlabel('Training data size')
    for n_mut in range(1, max_n_mut+1):
        ax = axes[n_mut]
        tmp = retrieve_metric(df, metric, n_mut=n_mut, predictor=predictors)
        sns.lineplot(data=tmp, x='n_train', y='val',
                     hue='predictor', style='predictor', ax=ax,
                     hue_order=predictors, style_order=predictors, **kwargs)
        ax.set_title(nmut_to_title[n_mut])
        ax.set_ylabel(metric_display_name[metric])
        ax.set_xlabel('Training data size')
    if mutcounts is not None:
        for i in range(max_n_mut+1):
            axes[i].annotate(f'Data size: {int(mutcounts[i])}',
                xy=(0.29, 0.03), xycoords='axes fraction', 
                fontsize=9)

    if legend is not None:
        handles, labels = legend['handles'], legend['labels']
        lgd = fig.legend(handles, labels, bbox_to_anchor=legend['loc'],
                loc='upper left', ncol=1, fontsize=11, frameon=False)
    
    ax.xaxis.set_minor_locator(MultipleLocator(24))
    ax.xaxis.set_major_locator(MultipleLocator(48))
    ax.xaxis.set_major_formatter('{x:.0f}')

    for ax in axes:
        ax.get_legend().remove()
        
    pad = 8
    ax = axes[0]
    annot = ax.annotate(dataset_name, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90, fontsize=14)
    ax.annotate('Test on:', xy=(-0.05, 0.492), xycoords=ax.title, textcoords='offset points',
                size='large', ha='right', va='center', rotation=0, fontsize=13)

    plt.subplots_adjust(top=0.80, wspace=0.1)
    if legend is not None:
        plt.savefig('../figs/' + savename + '.png', format='png', dpi=600,
                bbox_extra_artists=(annot,lgd,), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('../figs/' + savename + '.png', format='png', dpi=600,
                bbox_inches='tight', pad_inches=0)
    plt.show()