"""
Epistemic Cost Ensemble Module for SafeDreamer
Project: Safe Reinforcement Learning using Model Uncertainty (CS F317, BITS Pilani)

Replaces SafeDreamer's single deterministic MLP cost head with a K-head ensemble
trained with bootstrap masking and diverse weight initialization.
"""

import torch
import torch.nn as nn


class EpistemicCostEnsemble(nn.Module):
    """
    K-Head Ensemble Cost Predictor sharing a common latent state z_t.
    Computes empirical mean and standard deviation across heads to evaluate
    epistemic model uncertainty during imagined rollouts.
    """
    def __init__(self, latent_dim=1024, hidden_dim=256, num_heads=5, layers=2):
        super().__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim

        # K independent MLPs with diverse random weight initialization
        heads = []
        for _ in range(num_heads):
            mlp_layers = []
            in_dim = latent_dim
            for _ in range(layers):
                mlp_layers.append(nn.Linear(in_dim, hidden_dim))
                mlp_layers.append(nn.ReLU())
                in_dim = hidden_dim
            mlp_layers.append(nn.Linear(hidden_dim, 1))
            mlp_layers.append(nn.Sigmoid())  # Cost indicator in [0, 1]
            heads.append(nn.Sequential(*mlp_layers))

        self.heads = nn.ModuleList(heads)

    def forward(self, z, kappa=1.0):
        """
        Forward pass over latent states z.
        Args:
            z (torch.Tensor): Latent state batch of shape (batch_size, latent_dim)
                              or (horizon, batch_size, latent_dim).
            kappa (float): Conservatism multiplier for model disagreement.
        Returns:
            c_pessimistic (torch.Tensor): Pessimistic cost mu_c + kappa * sigma_c.
            mu_c (torch.Tensor): Empirical mean predicted cost across heads.
            sigma_c (torch.Tensor): Epistemic standard deviation (disagreement).
        """
        # Pass latent state through each head: shape (num_heads, *batch_shape, 1)
        preds = torch.stack([head(z) for head in self.heads], dim=0)

        # Empirical mean across ensemble heads
        mu_c = preds.mean(dim=0)

        # Empirical sample standard deviation (epistemic uncertainty)
        if self.num_heads > 1:
            sigma_c = preds.std(dim=0, unbiased=True)
        else:
            sigma_c = torch.zeros_like(mu_c)

        # Pessimistic cost penalty for Lagrangian Actor-Critic planning
        c_pessimistic = mu_c + kappa * sigma_c

        return c_pessimistic, mu_c, sigma_c

    def loss(self, z, target_costs, bootstrap_masks=None):
        """
        Computes training loss across ensemble heads.
        Args:
            z (torch.Tensor): Latent state inputs from replay buffer.
            target_costs (torch.Tensor): Ground-truth binary hazard costs in {0, 1}.
            bootstrap_masks (torch.Tensor, optional): Binary masks for bootstrap sampling.
        Returns:
            total_loss (torch.Tensor): Summed loss across all heads.
            losses_per_head (list): Individual head BCE losses.
        """
        criterion = nn.BCELoss(reduction='none')
        total_loss = 0.0
        head_losses = []

        for i, head in enumerate(self.heads):
            pred = head(z)
            bce = criterion(pred, target_costs)
            if bootstrap_masks is not None:
                bce = bce * bootstrap_masks[i]
            loss_i = bce.mean()
            total_loss += loss_i
            head_losses.append(loss_i.item())

        return total_loss, head_losses
