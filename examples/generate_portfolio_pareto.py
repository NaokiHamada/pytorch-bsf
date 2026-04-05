import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# 3-asset mean-variance portfolio optimization
# x in R^3: portfolio weights (unconstrained with L2 regularization for strong convexity)
mu = np.array([0.05, 0.12, 0.08])  # expected returns
sigma_diag = np.array([0.10, 0.15, 0.08])  # variances


def f1(x):
    # Negative expected return + L2 regularization
    return -mu @ x + 0.05 * np.sum(x ** 2)


def f2(x):
    # Portfolio variance + L2 regularization
    return x @ np.diag(sigma_diag) @ x + 0.05 * np.sum(x ** 2)


def f(x, w):
    return w[0] * f1(x) + w[1] * f2(x)


# Sample 10 weights on the 1-simplex from w=(1,0) to w=(0,1)
n_samples = 10
weights = np.linspace(0.0, 1.0, n_samples)
optimals = []
for t in weights:
    w = np.array([1.0 - t, t])
    result = minimize(
        lambda x: f(x, w),
        x0=np.zeros(3),
        method="L-BFGS-B",
    )
    x_opt = result.x
    optimals.append((w, x_opt, f1(x_opt), f2(x_opt)))

# Prepare training data for Bezier simplex fitting
w1 = np.array([p[0][0] for p in optimals])
w2 = np.array([p[0][1] for p in optimals])
X = np.column_stack([w1, w2])
y = np.column_stack([
    np.array([p[2] for p in optimals]),
    np.array([p[3] for p in optimals]),
])

# Fit a degree-3 Bezier simplex to the Pareto front
regressor = BezierSimplexRegressor(degree=3)
regressor.fit(X, y)

# Generate smooth curve for visualization
t_smooth = np.linspace(0.0, 1.0, 200)
X_smooth = np.column_stack([1.0 - t_smooth, t_smooth])
y_smooth = regressor.predict(X_smooth)

# Compute approximation error on training points
pred_train = regressor.predict(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y[:, 0], y[:, 1], color="blue", label="Optimization-derived Pareto points")
ax.plot(y_smooth[:, 0], y_smooth[:, 1], color="red", label="Fitted Bézier simplex")
ax.set_xlabel("Negative Expected Return + L2 (f₁)")
ax.set_ylabel("Portfolio Variance + L2 (f₂)")
ax.set_title("3-Asset Portfolio Pareto Front: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/portfolio_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

# Fit Bézier simplex to Pareto set (portfolio allocation space, first two assets)
z_targets = np.array([p[1] for p in optimals])
regressor_set = BezierSimplexRegressor(degree=3)
regressor_set.fit(X, z_targets)
z_smooth = regressor_set.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(z_targets[:, 0], z_targets[:, 1], color="green", label="Optimal allocations (samples)", s=50)
ax2.plot(z_smooth[:, 0], z_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.set_xlabel("x₁ (Asset 1 Weight)")
ax2.set_ylabel("x₂ (Asset 2 Weight)")
ax2.set_title("Pareto Set: Portfolio Allocations with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/portfolio_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
