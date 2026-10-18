"""
Lagrangian actor-critic trained in imagination (Dreamer-style dynamics backpropagation).

Key correctness rule (this was the bug in the original smoke test):
    the actor loss is minimised ONLY with respect to ACTOR parameters.
    World-model parameters are frozen during imagination (requires_grad = False), so gradients
    flow THROUGH the learned dynamics/reward/cost to the actions, but the model itself cannot
    "cheat" by inflating predicted reward or deflating predicted cost.

Imagination from start latents z_0 (detached posterior states from replay):
    a_t ~ pi(z_t),  r_t = r_hat(z_t, a_t),  z_{t+1} = f(z_t, a_t),  c_t = c~(z_{t+1}) = mu + kappa*sigma
Lambda-returns (lambda = 0.95) for reward and cost, bootstrapped with target critics V_R, V_C.
Actor objective:   maximise  E[ R^lam_t  -  lag * C^lam_t ] / (1 + lag)   + entropy bonus
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .world_model import mlp


class Actor(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden=256, min_std=0.1, max_std=1.0):
        super().__init__()
        self.net = mlp(latent_dim, hidden, 2 * act_dim)
        self.min_std, self.max_std = min_std, max_std

    def dist(self, z):
        mean, raw = self.net(z).chunk(2, -1)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(raw + 2.0)
        return torch.distributions.Normal(mean, std)

    def sample(self, z, deterministic=False):
        d = self.dist(z)
        u = d.mean if deterministic else d.rsample()
        a = torch.tanh(u)
        # entropy of the pre-tanh Gaussian (simple, standard approximation)
        return a, d.entropy().sum(-1)


def lambda_return(rew, val, gamma, lam):
    """rew: (H, B); val: (H+1, B) -> returns (H, B)."""
    H = rew.shape[0]
    out, nxt = [], val[-1]
    for t in reversed(range(H)):
        nxt = rew[t] + gamma * ((1 - lam) * val[t + 1] + lam * nxt)
        out.append(nxt)
    return torch.stack(out[::-1])


class LagrangianActorCritic(nn.Module):
    def __init__(self, latent_dim, act_dim, hidden=256, gamma=0.99, lam=0.95, horizon=15,
                 entropy_coef=3e-4, actor_lr=8e-5, critic_lr=3e-4, tau=0.02):
        super().__init__()
        self.actor = Actor(latent_dim, act_dim, hidden)
        self.v_r = mlp(latent_dim, hidden, 1)
        self.v_c = mlp(latent_dim, hidden, 1)
        self.v_r_targ, self.v_c_targ = copy.deepcopy(self.v_r), copy.deepcopy(self.v_c)
        for p in list(self.v_r_targ.parameters()) + list(self.v_c_targ.parameters()):
            p.requires_grad_(False)
        self.gamma, self.lam, self.H, self.ent, self.tau = gamma, lam, horizon, entropy_coef, tau
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.v_r.parameters()) + list(self.v_c.parameters()), lr=critic_lr)

    def imagine(self, wm, z0, kappa):
        zs, rs, cs, sig, ents = [z0], [], [], [], []
        z = z0
        for _ in range(self.H):
            a, ent = self.actor.sample(z)
            r = wm.reward_fn(z, a)
            z = wm.step(z, a)
            c_tilde, mu, sigma = wm.cost(z, kappa)
            zs.append(z); rs.append(r); cs.append(c_tilde.squeeze(-1)); sig.append(sigma.squeeze(-1)); ents.append(ent)
        return torch.stack(zs), torch.stack(rs), torch.stack(cs), torch.stack(sig), torch.stack(ents)

    def update(self, wm, z0, lag, kappa):
        # ---- actor: world model frozen, gradients flow through it to the actions ----
        req = [p.requires_grad for p in wm.parameters()]
        for p in wm.parameters():
            p.requires_grad_(False)
        zs, rs, cs, sig, ents = self.imagine(wm, z0.detach(), kappa)
        vr = self.v_r_targ(zs).squeeze(-1)
        vc = self.v_c_targ(zs).squeeze(-1)
        ret_r = lambda_return(rs, vr, self.gamma, self.lam)
        ret_c = lambda_return(cs, vc, self.gamma, self.lam)
        objective = (ret_r - lag * ret_c) / (1.0 + lag)
        actor_loss = -(objective.mean() + self.ent * ents.mean())
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.actor_opt.step()
        for p, r in zip(wm.parameters(), req):
            p.requires_grad_(r)

        # ---- critics: regress on detached lambda-returns ----
        zs_d = zs[:-1].detach()
        critic_loss = F.mse_loss(self.v_r(zs_d).squeeze(-1), ret_r.detach()) + \
            F.mse_loss(self.v_c(zs_d).squeeze(-1), ret_c.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for net, targ in [(self.v_r, self.v_r_targ), (self.v_c, self.v_c_targ)]:
                for p, pt in zip(net.parameters(), targ.parameters()):
                    pt.mul_(1 - self.tau).add_(self.tau * p)

        disc = self.gamma ** torch.arange(self.H, dtype=torch.float32).unsqueeze(-1)
        return dict(
            actor_loss=actor_loss.item(), critic_loss=critic_loss.item(),
            img_reward_H=(disc * rs).sum(0).mean().item(),        # discounted reward over the H imagined steps
            img_cost_H=(disc * cs).sum(0).mean().item(),          # discounted pessimistic cost over H steps
            img_cost_value=ret_c[0].mean().item(),                # lambda-return incl. bootstrap (discounted J^C estimate)
            sigma=sig.mean().item(), max_step_cost=cs.max().item(), min_step_cost=cs.min().item(),
        )
