"""
Epistemic Cost Ensemble (K heads) -- fixed version.
Project: Safe Reinforcement Learning using Model Uncertainty (CS F317, BITS Pilani)

Changes vs. the original file:
  1. Heads output LOGITS and the loss uses binary_cross_entropy_with_logits
     (numerically stable). Sigmoid is applied only when predicting.
  2. Bootstrap masks are actually usable: per-head, per-sample weights
     (Poisson(1) "online bootstrap" or Bernoulli(0.5)), and the loss is normalised
     by the mask sum, so masking really re-weights the data.
  3. Optional randomized prior functions (Osband et al., NeurIPS 2018): each head adds a
     FIXED, untrained random network to its logit. This is a standard trick that keeps
     ensemble members disagreeing away from the training data.
  4. `predict_heads` returns per-head probabilities for analysis.

forward(z, kappa) keeps the original signature and returns (c_tilde, mu_c, sigma_c),
where c_tilde = mu_c + kappa * sigma_c.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden_dim, layers, out_dim=1):
    mods, d = [], in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden_dim), nn.ReLU()]
        d = hidden_dim
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class EpistemicCostEnsemble(nn.Module):
    def __init__(self, latent_dim=1024, hidden_dim=256, num_heads=5, layers=2, prior_scale=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.prior_scale = prior_scale
        self.heads = nn.ModuleList([_mlp(latent_dim, hidden_dim, layers) for _ in range(num_heads)])
        if prior_scale > 0:
            self.priors = nn.ModuleList([_mlp(latent_dim, hidden_dim, layers) for _ in range(num_heads)])
            for p in self.priors.parameters():
                p.requires_grad_(False)  # priors are never trained
        else:
            self.priors = None

    def head_logits(self, z):
        """Per-head logits, shape (K, *batch, 1)."""
        out = []
        for k, head in enumerate(self.heads):
            logit = head(z)
            if self.priors is not None:
                logit = logit + self.prior_scale * self.priors[k](z)
            out.append(logit)
        return torch.stack(out, dim=0)

    def predict_heads(self, z):
        """Per-head hazard probabilities in [0, 1], shape (K, *batch, 1)."""
        return torch.sigmoid(self.head_logits(z))

    def forward(self, z, kappa=1.0):
        probs = self.predict_heads(z)
        mu_c = probs.mean(dim=0)
        sigma_c = probs.std(dim=0, unbiased=True) if self.num_heads > 1 else torch.zeros_like(mu_c)
        return mu_c + kappa * sigma_c, mu_c, sigma_c

    def loss(self, z, target_costs, bootstrap_masks=None):
        """
        z:               (B, latent_dim)
        target_costs:    (B, 1) binary hazard indicator in {0, 1}
        bootstrap_masks: (K, B, 1) non-negative per-head sample weights, or None
        Returns (summed loss over heads, list of per-head losses).
        """
        logits = self.head_logits(z)  # (K, B, 1)
        target = target_costs.unsqueeze(0).expand_as(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if bootstrap_masks is None:
            per_head = bce.mean(dim=(1, 2))
        else:
            w = bootstrap_masks
            per_head = (bce * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(1.0)
        return per_head.sum(), per_head.detach().tolist()


def make_bootstrap_masks(num_heads, num_samples, kind="poisson", generator=None):
    """Fixed per-sample bootstrap weights, shape (K, N). Index columns with batch indices."""
    if kind == "poisson":
        return torch.poisson(torch.ones(num_heads, num_samples), generator=generator)
    if kind == "bernoulli":
        return torch.bernoulli(torch.full((num_heads, num_samples), 0.5), generator=generator)
    if kind == "none":
        return torch.ones(num_heads, num_samples)
    raise ValueError(kind)
