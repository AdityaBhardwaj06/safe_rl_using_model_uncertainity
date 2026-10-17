"""
World Model Wrapper & Imagined Rollout Simulation
Project: Safe Reinforcement Learning using Model Uncertainty (CS F317, BITS Pilani)

Demonstrates how the RSSM dynamics, reward head, and 5-head epistemic cost ensemble
simulate imagined trajectories and compute Lagrangian Actor-Critic losses.
"""

import torch
import torch.nn as nn
from .ensemble_cost_head import EpistemicCostEnsemble
from .lagrangian import LagrangianMultiplier


class SimplifiedWorldModelPlanner(nn.Module):
    """
    Modular wrapper integrating:
    1. Recurrent State-Space Transition Dynamics P_hat(z_{t+1} | z_t, a_t)
    2. Reward Predictor R_hat(z_t)
    3. Epistemic Cost Ensemble C_hat(z_t) = mu_c + kappa * sigma_c
    4. Reward Critic V^R(z) and Cost Critic V^C(z)
    5. Actor Policy pi(a | z)
    """
    def __init__(self, obs_dim=60, action_dim=2, latent_dim=256, horizon=15, kappa=1.0, cost_limit=0.0):
        super().__init__()
        self.horizon = horizon
        self.kappa = kappa
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Encoder (Vector MLP or CNN for pixels)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Transition Dynamics: z_{t+1} = f(z_t, a_t)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Reward Head: R(z_t)
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # 5-Head Epistemic Cost Ensemble
        self.cost_ensemble = EpistemicCostEnsemble(latent_dim=latent_dim, hidden_dim=64, num_heads=5)

        # Reward Critic V^R(z) and Cost Critic V^C(z)
        self.reward_critic = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.cost_critic   = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 1))

        # Actor Policy: pi(a | z)
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        # Lagrangian Multiplier
        self.lagrangian = LagrangianMultiplier(cost_limit=cost_limit, lr=1e-2)

    def imagine_rollouts(self, initial_obs):
        """
        Simulates an H-step imagined trajectory inside the latent world model.
        Args:
            initial_obs (torch.Tensor): Real starting observations (batch_size, obs_dim).
        Returns:
            dict containing imagined states, actions, rewards, costs, and value estimates.
        """
        batch_size = initial_obs.shape[0]
        z_curr = self.encoder(initial_obs)

        imagined_z = [z_curr]
        imagined_actions = []
        imagined_rewards = []
        imagined_pessimistic_costs = []
        imagined_uncertainties = []

        for step in range(self.horizon):
            # Actor samples action in latent space
            action = self.actor(z_curr)
            imagined_actions.append(action)

            # Predict task reward
            r_pred = self.reward_head(z_curr)
            imagined_rewards.append(r_pred)

            # Predict pessimistic cost via ensemble
            c_pess, mu_c, sigma_c = self.cost_ensemble(z_curr, kappa=self.kappa)
            imagined_pessimistic_costs.append(c_pess)
            imagined_uncertainties.append(sigma_c)

            # Predict next latent state
            trans_in = torch.cat([z_curr, action], dim=-1)
            z_curr = self.transition(trans_in)
            imagined_z.append(z_curr)

        # Stack rollouts: shape (horizon, batch_size, ...)
        return {
            "z": torch.stack(imagined_z[:-1], dim=0),
            "next_z": torch.stack(imagined_z[1:], dim=0),
            "actions": torch.stack(imagined_actions, dim=0),
            "rewards": torch.stack(imagined_rewards, dim=0),
            "pessimistic_costs": torch.stack(imagined_pessimistic_costs, dim=0),
            "uncertainties": torch.stack(imagined_uncertainties, dim=0),
            "final_reward_value": self.reward_critic(imagined_z[-1]),
            "final_cost_value": self.cost_critic(imagined_z[-1])
        }

    def compute_actor_lagrangian_loss(self, rollouts):
        """
        Computes the Lagrangian saddle-point actor loss over imagined rollouts:
        Loss = -(Expected Reward Return - lambda * Expected Pessimistic Cost Return)
        """
        # Sum rewards and pessimistic costs over horizon
        gamma = 0.99
        horizon = rollouts["rewards"].shape[0]

        discounted_rewards = torch.zeros_like(rollouts["rewards"][0])
        discounted_costs   = torch.zeros_like(rollouts["pessimistic_costs"][0])

        for t in reversed(range(horizon)):
            discounted_rewards = rollouts["rewards"][t] + gamma * discounted_rewards
            discounted_costs   = rollouts["pessimistic_costs"][t] + gamma * discounted_costs

        # Bootstrap with critics at step H
        total_reward_return = discounted_rewards + (gamma ** horizon) * rollouts["final_reward_value"]
        total_cost_return   = discounted_costs   + (gamma ** horizon) * rollouts["final_cost_value"]

        lam = self.lagrangian.value
        # Lagrangian objective: maximize R - lambda * C => minimize -(R - lambda * C)
        actor_objective = total_reward_return - lam * total_cost_return
        actor_loss = -actor_objective.mean()

        return actor_loss, total_reward_return.mean().item(), total_cost_return.mean().item()
