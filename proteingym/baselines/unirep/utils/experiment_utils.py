import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import svm

from utils.metric_utils import get_spearman_fractions, wt_improvement_metric, topk_median


def test_regression_multiseeds(X, y, n_train, n_seeds, y_wt, y_cutoff=None,
        mutation_counts=None, mutation_count_cutoff=None):
    spm = np.zeros((n_seeds, len(SPEARMAN_FRACTIONS)))
    r2 = np.zeros(n_seeds)
    wt_imprv = np.zeros(n_seeds)
    topk_med = np.zeros(n_seeds)
    best_alpha = np.zeros(n_seeds)
    for i in range(n_seeds):
        spm[i], r2[i], wt_imprv[i], topk_med[i], best_alpha[i] = test_regression(
                X, y, n_train, y_wt, y_cutoff, i, mutation_counts,
                mutation_count_cutoff)
    df = pd.DataFrame({
        'R2 score': r2,
        'Improvement over WT': wt_imprv,
        'Top K median': topk_med,
        'N train': n_train,
        'Best alpha': best_alpha,
    })
    for i, f in enumerate(SPEARMAN_FRACTIONS):
        df[f'Spearman correlation at {f:.1f}'] = spm[:, i]
    df['Spearman correlation'] = spm[:, -1]
    return df


def test_regression(X, y, n_train, y_wt, y_cutoff=None, seed=0,
        mutation_counts=None, mutation_count_cutoff=None):
    if y_cutoff is not None:
        is_valid = (y >= y_cutoff)
        X, y = X[is_valid], y[is_valid]
        if mutation_counts is not None:
            mutation_counts = mutation_counts[is_valid]
    X_tr, X_eval, X_test, y_tr, y_eval, y_test = train_eval_test_split(
            X, y, seed, n_train, mutation_counts, mutation_count_cutoff)
    # Model selection.
    best_alpha = None
    best_spm = -999.9
    for alpha in [0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_eval)
        spm = spearmanr(y_pred, y_eval).correlation
        if spm > best_spm:
            best_alpha = alpha
            best_spm = spm
    model = Ridge(alpha=best_alpha)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    spm = get_spearman_fractions(y_pred, y_test)
    r2 = model.score(X_test, y_test)
    wt_imprv = wt_improvement_metric(y_pred, y_test, y_wt)
    topk_med = topk_median(y_pred, y_test)
    return spm, r2, wt_imprv, topk_med, best_alpha


def test_classification_multiseeds(X, y, n_train, n_seeds, y_cutoff=None,
        mutation_counts=None, mutation_count_cutoff=None):
    acc = np.zeros(n_seeds)
    best_C = np.zeros(n_seeds)
    for i in range(n_seeds):
        acc[i], best_C[i] = test_classification(
                X, y, n_train, y_cutoff, i, mutation_counts,
                mutation_count_cutoff)
    return acc, best_C


def test_classification(X, y, n_train, y_cutoff, seed=0,
        mutation_counts=None, mutation_count_cutoff=None):
    y = (y > y_cutoff).astype(int)
    X_tr, X_eval, X_test, y_tr, y_eval, y_test = train_eval_test_split(
            X, y, seed, n_train, mutation_counts, mutation_count_cutoff)
    while len(np.unique(y_tr)) < 2:
        X_tr, X_eval, X_test, y_tr, y_eval, y_test = train_eval_test_split(
                X, y, seed+np.random.randint(10000), n_train, mutation_counts,
                mutation_count_cutoff)
    best_C = None
    best_acc = -999.9
    for C in [0.01, 0.1, 0.5, 1.0, 2.0]:
        model = svm.LinearSVC(C=C)
        model.fit(X_tr, y_tr)
        acc = model.score(X_eval, y_eval)
        if acc > best_acc:
            best_C = C
            best_acc = acc
    model = svm.LinearSVC(C=best_C)
    model.fit(X_tr, y_tr)
    return model.score(X_test, y_test), best_C


def run_regression(feature_reps, y, y_wt, y_cutoff, n_seeds,
        mutation_counts=None, mutation_count_cutoff=None):
    results = pd.DataFrame()
    for feature_rep, X in feature_reps.items():
        print('Staring runs for', feature_rep)
        for n_train in [8, 24, 96, 192, 480, 960, 9600]:
            if n_train >= 0.8 * X.shape[0]:
                continue
            print('n_train:', n_train)
            df = test_regression_multiseeds(X, y, n_train, n_seeds, y_wt,
                    y_cutoff, mutation_counts, mutation_count_cutoff)
            df['Feature rep'] = feature_rep
            results = pd.concat([results, df], axis=0)
    return results


def run_classification(feature_reps, y, y_cutoff, n_seeds,
        mutation_counts=None, mutation_count_cutoff=None):
    results = pd.DataFrame()
    for feature_rep, X in feature_reps.items():
        print('Staring runs for', feature_rep)
        for n_train in [8, 24, 96, 192, 480, 960]:
            if n_train >= 0.8 * X.shape[0]:
                continue
            print('n_train:', n_train)
            acc, best_C = test_classification_multiseeds(
                X, y, n_train, n_seeds, y_cutoff, mutation_counts,
                mutation_count_cutoff)
            df = pd.DataFrame({
                'Accuracy': acc,
                'Best reg coeff': best_C,
                'N train': n_train,
                'Feature rep': feature_rep,
            })
            results = pd.concat([results, df], axis=0)
    return results