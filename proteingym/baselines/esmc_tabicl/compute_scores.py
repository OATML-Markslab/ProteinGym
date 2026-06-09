"""ESMC-TabICL: supervised variant-effect prediction on the ProteinGym substitution benchmark.

For each DMS assay, features per variant are:
  - ESMC-600M second-to-last-layer (hidden_states[-2]) embedding, mean-pooled over residues  (1152)
  - approximate pseudo-log-likelihood: one unmasked forward pass, sum_i log P(s_i | s)          (1)
  - masked wt-marginal mutation-effect score (mask each mutated position in the wild type, read
    the blind prediction logP[mut]-logP[wt]), aggregated as deciles over the mutated positions    (11)
These 1164 features go to a TabICLv2 regressor under ProteinGym's CV protocol (train on 4 folds,
predict the held-out fold; targets standardized per fold). One score file per assay per CV split:
    <output_dir>/<cv_scheme>/<DMS_id>.csv  with columns [mutant, y, y_pred, fold].

Usage:
    python compute_scores.py --dms_reference reference_files/DMS_substitutions.csv \
        --dms_folder <cv_folds_singles_substitutions> --output_dir <scores_out> [--dms_index N]

Both ESMC-600M (biohub/ESMC-600M) and TabICLv2 (tabicl) are open source.
"""
import argparse, os, re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tabicl import TabICLRegressor

MODEL = "biohub/ESMC-600M"
CV_SCHEMES = ["fold_random_5", "fold_modulo_5", "fold_contiguous_5"]
TOKEN_BUDGET = 60000     # tokens/batch for the forward passes
TRAIN_CAP = 10000        # cap training context (a single TabICL fit peaks ~80GB at n=10k)
DECILES = list(range(0, 101, 10))
MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_muts(mut):
    return [(m.group(1), int(m.group(2)), m.group(3))
            for m in (MUT_RE.match(p) for p in str(mut).split(":"))]


def derive_wt(df):
    """Reconstruct the wild-type sequence by reverting any variant's mutation(s)."""
    seq = list(df["mutated_sequence"].iloc[0])
    for wt_aa, pos, mut_aa in parse_muts(df["mutant"].iloc[0]):
        seq[pos - 1] = wt_aa
    return "".join(seq)


def batched(seqs, budget=TOKEN_BUDGET):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    i = 0
    while i < len(order):
        j, Lmax = i, len(seqs[order[i]]) + 2
        while j < len(order):
            L = max(Lmax, len(seqs[order[j]]) + 2)
            if (j - i + 1) * L > budget and j > i:
                break
            Lmax = L; j += 1
        yield order[i:j]
        i = j


@torch.inference_mode()
def featurize(df, tok, model, dev):
    """Return (emb (N,1152), pll (N,), wtdec (N,11)) for all variants in an assay."""
    seqs = df["mutated_sequence"].tolist()
    n, d = len(seqs), model.config.d_model
    cls, eos, pad = tok.cls_token_id, tok.eos_token_id, tok.pad_token_id
    cap = {}
    h = model.esmc.transformer.blocks[-2].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))

    # (1+2) per-variant emb (mean-pooled second-to-last layer) and approximate pseudo-LL
    emb = np.zeros((n, d), np.float32); pll = np.zeros(n, np.float32)
    for idx in batched(seqs):
        enc = tok([seqs[k] for k in idx], return_tensors="pt", padding=True).to(dev)
        ids = enc["input_ids"]
        logits = model(**enc).logits
        keep = enc["attention_mask"].bool() & (ids != cls) & (ids != eos) & (ids != pad)
        mm = keep.unsqueeze(-1).to(cap["h"].dtype)
        pooled = (cap["h"] * mm).sum(1) / mm.sum(1).clamp_min(1.0)
        lp = F.log_softmax(logits, -1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        sp = (lp * keep).sum(1)
        for r, k in enumerate(idx):
            emb[k] = pooled[r].float().cpu().numpy(); pll[k] = float(sp[r])
    h.remove()

    # (3) masked wt-marginal: mask each unique mutated position in the WT once
    wt = derive_wt(df)
    base = tok([wt], return_tensors="pt")["input_ids"][0]
    positions = sorted({p for mut in df["mutant"] for _, p, _ in parse_muts(mut)})
    Lt = base.shape[0]; bs = max(1, TOKEN_BUDGET // Lt)
    logp_at = {}
    for s in range(0, len(positions), bs):
        chunk = positions[s:s + bs]
        bt = base.unsqueeze(0).repeat(len(chunk), 1).clone()
        for r, p in enumerate(chunk):
            bt[r, p] = tok.mask_token_id
        out = model(input_ids=bt.to(dev), attention_mask=torch.ones_like(bt).to(dev)).logits
        for r, p in enumerate(chunk):
            logp_at[p] = F.log_softmax(out[r, p].float(), -1).cpu().numpy()
    aa = {}
    wtdec = np.zeros((n, len(DECILES)), np.float32)
    for r, mut in enumerate(df["mutant"]):
        sc = []
        for wt_aa, pos, mut_aa in parse_muts(mut):
            wi = aa.setdefault(wt_aa, tok.convert_tokens_to_ids(wt_aa))
            mi = aa.setdefault(mut_aa, tok.convert_tokens_to_ids(mut_aa))
            sc.append(logp_at[pos][mi] - logp_at[pos][wi])
        wtdec[r] = np.percentile(np.array(sc), DECILES)
    return emb, pll.reshape(-1, 1), wtdec


def score_assay(df, X, out_dir, dms_id):
    y = df["DMS_score"].to_numpy(np.float64)
    for scheme in CV_SCHEMES:
        folds = df[scheme].to_numpy()
        y_true = np.full(len(df), np.nan); y_pred = np.full(len(df), np.nan)
        for f in np.unique(folds):
            tr = np.where(folds != f)[0]; te = folds == f
            if len(tr) > TRAIN_CAP:
                rng = np.random.default_rng(abs(hash((dms_id, scheme, int(f)))) % 2**32)
                tr = np.sort(rng.choice(tr, TRAIN_CAP, replace=False))
            mu, sd = y[tr].mean(), (y[tr].std() or 1.0)
            reg = TabICLRegressor(random_state=42)
            reg.fit(X[tr], (y[tr] - mu) / sd)
            y_pred[te] = reg.predict(X[te]); y_true[te] = (y[te] - mu) / sd
        d = os.path.join(out_dir, scheme); os.makedirs(d, exist_ok=True)
        pd.DataFrame({"mutant": df["mutant"], "y": y_true, "y_pred": y_pred, "fold": folds}
                     ).to_csv(os.path.join(d, dms_id + ".csv"), index=False)
        rho = spearmanr(y_true, y_pred).correlation
        print(f"  {dms_id} {scheme} Spearman={rho:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dms_reference", required=True)
    ap.add_argument("--dms_folder", required=True, help="folder of per-assay CV-fold CSVs")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--dms_index", type=int, default=-1, help="score one assay (row in reference); default all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ref = pd.read_csv(args.dms_reference)
    ids = [ref["DMS_id"].iloc[args.dms_index]] if args.dms_index >= 0 else list(ref["DMS_id"])
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, dtype=torch.float32).to(args.device).eval()

    for dms_id in ids:
        fn = ref.loc[ref.DMS_id == dms_id, "DMS_filename"].values[0]
        df = pd.read_csv(os.path.join(args.dms_folder, fn))
        emb, pll, wtdec = featurize(df, tok, model, args.device)
        X = np.concatenate([emb, pll, wtdec], axis=1).astype(np.float32)
        score_assay(df, X, args.output_dir, dms_id)


if __name__ == "__main__":
    main()
