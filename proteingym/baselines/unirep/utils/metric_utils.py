import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score, ndcg_score

SPEARMAN_FRACTIONS = np.linspace(0.1, 1.0, 10)


def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true).correlation


def spearman_scoring_fn(sklearn_estimator, X, y):
    return spearman(sklearn_estimator.predict(X), y)


def ndcg(y_pred, y_true):
    y_true_normalized = (y_true - y_true.mean()) / y_true.std()
    return ndcg_score(y_true_normalized.reshape(1, -1), y_pred.reshape(1, -1))


def topk_mean(y_pred, y_true, topk=96):
    return np.mean(y_true[np.argsort(y_pred)[-topk:]])


def r2(y_pred, y_true):
    return r2_score(y_true, y_pred)


def hit_rate(y_pred, y_true, y_ref=0.0, topk=96):
    n_above = np.sum(y_true[np.argsort(y_pred)[-topk:]] > y_ref)
    return float(n_above) / float(topk)


def aucroc(y_pred, y_true, y_cutoff):
    y_true_bin = (y_true >= y_cutoff)
    return roc_auc_score(y_true_bin, y_pred, average='micro')


def get_spearman_fractions(y_pred, y_true):
    results = np.zeros(len(SPEARMAN_FRACTIONS))
    for i, f in enumerate(SPEARMAN_FRACTIONS):
        k = int(f * len(y_true))
        idx = np.argsort(y_true)[-k:]
        results[i] = spearmanr(y_pred[idx], y_true[idx]).correlation
    return results


def wt_improvement_metric(y_pred, y_true, y_wt, topk=96):
    hr = hit_rate(y_pred, y_true, y_wt, topk)
    baseline = float(np.sum(y_true > y_wt)) / len(y_true)
    return hr / baseline


def topk_median(y_pred, y_true, topk=96):
    return np.median(y_true[np.argsort(y_pred)[-topk:]])