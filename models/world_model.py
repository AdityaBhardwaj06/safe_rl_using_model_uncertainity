"""
A small deterministic latent world model for VECTOR observations (Stage 2 of the plan).

This is NOT DreamerV3's RSSM. It is a minimal, honest stand-in that lets us test the
cost-ensemble idea end-to-end on CPU:

    encoder     z_t   = LN(enc(o_t))
    dynamics    z_t+1 = LN(z_t + f([z_t, a_t]))          (residual MLP)
    decoder     o_hat = dec(z)
    reward head r_hat = g([z_t, a_t])
    cost heads  c~(z_t+1) = mu + kappa * sigma            (EpistemicCostEnsemble on the NEXT latent,
                                                           because Safety-Gymnasium charges the cost
                                                           at the state reached after the step)

Training (world-model loss only -- the actor loss NEVER updates these parameters):
    open-loop multi-step prediction over sequences of length L:
      latent loss   || z_hat_t+1 - sg(enc(o_t+1)) ||^2
      decoder loss  || dec(z_hat_t+1) - o_t+1 ||^2  +  || dec(enc(o_t)) - o_t ||^2
      reward loss   || r_hat_t - r_t ||^2            (on posterior latents)
      cost loss     BCE of the K heads on sg(enc(o_t+1)) with bootstrap weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ensemble_cost_head import EpistemicCostEnsemble


def mlp(i, h, o, n=2, act=nn.ELU):
    mods, d = [], i
    for _ in range(n):
        mods += [nn.Linear(d, h), act()]
        d = h
    mods.append(nn.Linear(d, o))
    return nn.Sequential(*mods)


class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim=64, hidden=256, num_cost_heads=5,
                 cost_hidden=256, prior_scale=0.0):
        super().__init__()
        self.enc = mlp(obs_dim, hidden, latent_dim)
        self.enc_ln = nn.LayerNorm(latent_dim)
        self.dyn = mlp(latent_dim + act_dim, hidden, latent_dim)
        self.dyn_ln = nn.LayerNorm(latent_dim)
        self.dec = mlp(latent_dim, hidden, obs_dim)
        self.reward = mlp(latent_dim + act_dim, hidden, 1)
        self.cost = EpistemicCostEnsemble(latent_dim, cost_hidden, num_cost_heads, 2, prior_scale=prior_scale)
        self.latent_dim = latent_dim

    def encode(self, o):
        return self.enc_ln(self.enc(o))

    def step(self, z, a):
        return self.dyn_ln(z + self.dyn(torch.cat([z, a], -1)))

    def reward_fn(self, z, a):
        return self.reward(torch.cat([z, a], -1)).squeeze(-1)

    def loss(self, obs, act, rew, cost, boot_w=None):
        """
        obs: (B, L+1, obs_dim)  act: (B, L, act_dim)  rew, cost: (B, L)  boot_w: (K, B, L) or None
        """
        B, L = act.shape[:2]
        z_post = self.encode(obs)                                   # (B, L+1, Z)
        recon = F.mse_loss(self.dec(z_post), obs)
        z = z_post[:, 0]
        lat, pred = 0.0, 0.0
        for t in range(L):
            z = self.step(z, act[:, t])
            lat = lat + F.mse_loss(z, z_post[:, t + 1].detach())
            pred = pred + F.mse_loss(self.dec(z), obs[:, t + 1])
        lat, pred = lat / L, pred / L
        r_loss = F.mse_loss(self.reward_fn(z_post[:, :-1].detach(), act), rew)
        zc = z_post[:, 1:].detach().reshape(B * L, -1)
        w = None if boot_w is None else boot_w.reshape(boot_w.shape[0], B * L, 1)
        c_loss, _ = self.cost.loss(zc, cost.reshape(B * L, 1), w)
        total = recon + pred + lat + r_loss + c_loss / self.cost.num_heads
        return total, dict(recon=recon.item(), pred=pred.item(), latent=lat.item(),
                           reward=r_loss.item(), cost=c_loss.item() / self.cost.num_heads)
