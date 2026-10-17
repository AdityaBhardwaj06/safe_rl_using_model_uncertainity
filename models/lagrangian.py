"""
Lagrangian Multiplier Module for Safe RL
Project: Safe Reinforcement Learning using Model Uncertainty (CS F317, BITS Pilani)

Implements the adaptive dual multiplier update for Constrained MDPs:
lambda_{k+1} = max(0, lambda_k + alpha_lambda * (J^C(pi) - b))
"""

import torch
import torch.nn as nn


class LagrangianMultiplier(nn.Module):
    """
    Manages the adaptive Lagrange multiplier lambda >= 0.
    Dynamically balances reward maximization and cost constraint satisfaction.
    """
    def __init__(self, cost_limit=0.0, init_lambda=0.0, lr=1e-3, lambda_max=50.0):
        super().__init__()
        self.cost_limit = float(cost_limit)
        self.lr = float(lr)
        self.lambda_max = float(lambda_max)

        # Parameterize via unconstrained log_lambda or raw parameter clamped at 0
        self.raw_lambda = nn.Parameter(torch.tensor(float(init_lambda), dtype=torch.float32))

    @property
    def value(self):
        """Current clamped non-negative multiplier value."""
        return torch.clamp(self.raw_lambda, min=0.0, max=self.lambda_max)

    def update(self, current_cost_return):
        """
        Dual gradient update step based on constraint violation.
        Args:
            current_cost_return (float or torch.Tensor): Expected cumulative cost J^C.
        Returns:
            violation (float): J^C - b.
            new_lambda (float): Updated multiplier value.
        """
        if isinstance(current_cost_return, torch.Tensor):
            cost_val = current_cost_return.detach().item()
        else:
            cost_val = float(current_cost_return)

        violation = cost_val - self.cost_limit

        # Dual ascent: if violation > 0, increase lambda; else decay toward 0
        new_val = self.raw_lambda.data + self.lr * violation
        self.raw_lambda.data = torch.clamp(new_val, min=0.0, max=self.lambda_max)

        return violation, self.value.item()
