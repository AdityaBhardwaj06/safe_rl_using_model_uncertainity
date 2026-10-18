"""Shared helpers for the cost-head experiments (Exp 1 and Exp 2)."""
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.ensemble_cost_head import EpistemicCostEnsemble, make_bootstrap_masks  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))


def load_split(path, test_frac=0.2, seed=0):
    """Split by EPISODE (not by transition) so test states come from unseen layouts."""
    d = np.load(path)
    eps = np.unique(d["episode"])
    rng = np.random.default_rng(seed)
    test_eps = set(rng.choice(eps, size=max(1, int(len(eps) * test_frac)), replace=False).tolist())
    is_test = np.isin(d["episode"], list(test_eps))
    # The cost at step t is incurred at the state reached AFTER the step -> use next_obs.
    X, y, dist, ep = d["next_obs"], d["cost"], d["dist"], d["episode"]
    return dict(
        X_tr=X[~is_test], y_tr=y[~is_test], dist_tr=dist[~is_test], ep_tr=ep[~is_test],
        X_te=X[is_test], y_te=y[is_test], dist_te=dist[is_test],
    )


class Normalizer:
    def __init__(self, X):
        self.mu = X.mean(0, keepdims=True)
        self.sd = np.maximum(X.std(0, keepdims=True), 1e-2)  # floor: some sensor dims are ~constant

    def __call__(self, X):
        return torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32)


def train_ensemble(X, y, num_heads=5, steps=3000, batch=256, lr=1e-3, bootstrap="none",
                   prior_scale=0.0, hidden=256, layers=2, seed=0):
    torch.manual_seed(seed)
    ens = EpistemicCostEnsemble(X.shape[1], hidden, num_heads, layers, prior_scale=prior_scale)
    opt = torch.optim.Adam([p for p in ens.parameters() if p.requires_grad], lr=lr)
    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
    g = torch.Generator().manual_seed(seed)
    masks = make_bootstrap_masks(num_heads, len(Xt), kind=bootstrap, generator=g)
    for _ in range(steps):
        idx = torch.randint(0, len(Xt), (min(batch, len(Xt)),), generator=g)
        loss, _ = ens.loss(Xt[idx], yt[idx], masks[:, idx].unsqueeze(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return ens


@torch.no_grad()
def predict(ens, X, kappa=1.0, chunk=8192):
    probs, logits = [], []
    for i in range(0, len(X), chunk):
        lg = ens.head_logits(X[i:i + chunk]).squeeze(-1)  # (K, n)
        logits.append(lg)
        probs.append(torch.sigmoid(lg))
    P, L = torch.cat(probs, 1).numpy(), torch.cat(logits, 1).numpy()
    mu = P.mean(0)
    sigma = P.std(0, ddof=1) if P.shape[0] > 1 else np.zeros_like(mu)
    sigma_logit = L.std(0, ddof=1) if L.shape[0] > 1 else np.zeros_like(mu)
    mean_logit = L.mean(0)
    # c_tilde_logit: pessimism applied BEFORE the sigmoid (proposed fix for sigmoid saturation);
    # sigmoid is monotone, so ranking-based metrics (AUROC/TPR) can use the logit value directly.
    return dict(mu=mu, sigma=sigma, sigma_logit=sigma_logit, c_tilde=mu + kappa * sigma,
                mean_logit=mean_logit, c_tilde_logit=mean_logit + kappa * sigma_logit, P=P)


def tpr_at_fpr(score_pos, score_neg, fpr=0.01):
    """Fraction of positives above the threshold that flags only `fpr` of negatives."""
    thr = np.quantile(score_neg, 1 - fpr)
    return float((score_pos > thr).mean())
