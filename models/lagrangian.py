"""
Dual (Lagrange multiplier) update for the CMDP  max J^R  s.t.  J^C <= b.

    lambda_{k+1} = clip( lambda_k + alpha * (J^C_hat - b),  0,  lambda_max )

Note: lambda is capped at lambda_max, so it cannot diverge to infinity in this code.

Budget scaling. Safety-Gymnasium budgets b are UNDISCOUNTED costs per T = 1000-step episode.
If a critic estimates DISCOUNTED cost (sum_t gamma^t c_t), compare it to the matching
discounted budget for a constant per-step cost rate b / T:
    d = (b / T) * (1 - gamma^T) / (1 - gamma)        (~ b / 10 for gamma = 0.99, T = 1000)
"""


def discounted_budget(b, T=1000, gamma=0.99):
    return (b / T) * (1 - gamma ** T) / (1 - gamma)


class LagrangeMultiplier:
    def __init__(self, init=0.0, lr=0.01, max_value=50.0):
        self.value = float(init)
        self.lr = float(lr)
        self.max_value = float(max_value)

    def update(self, cost_estimate, budget):
        violation = float(cost_estimate) - float(budget)
        self.value = min(max(0.0, self.value + self.lr * violation), self.max_value)
        return violation, self.value
