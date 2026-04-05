import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# 2-generator 2-objective: generation cost vs. emissions
# P = [P1, P2] in R^2 (power outputs)


def f1(P):
    # Generation cost (strongly convex)
    return 0.5 * P[0] ** 2 + 0.3 * P[1] ** 2 + 0.05 * (P[0] + P[1]) ** 2


def f2(P):
    # Emissions (strongly convex)
    return 0.2 * P[0] ** 2 + 0.6 * P[1] ** 2 + 0.05 * (P[0] + P[1]) ** 2


def f(P, w):
    return w[0] * f1(P) + w[1] * f2(P)


# Sample 10 weights on the 1-simplex from w=(1,0) to w=(0,1)
n_samples = 10
weights = np.linspace(0.0, 1.0, n_samples)
optimals = []
for t in weights:
    w = np.array([1.0 - t, t])
    result = minimize(
        lambda P: f(P, w),
        x0=np.ones(2),
        method="L-BFGS-B",
    )
    P_opt = result.x
    optimals.append((w, P_opt, f1(P_opt), f2(P_opt)))

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
ax.set_xlabel("Generation Cost (f₁)")
ax.set_ylabel("Emissions (f₂)")
ax.set_title("Smart Grid Pareto Front: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/smart_grids_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

# Fit Bézier simplex to Pareto set (power output space)
P_targets = np.array([p[1] for p in optimals])
regressor_set = BezierSimplexRegressor(degree=3)
regressor_set.fit(X, P_targets)
P_smooth = regressor_set.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(P_targets[:, 0], P_targets[:, 1], color="green", label="Optimal power outputs (samples)", s=50)
ax2.plot(P_smooth[:, 0], P_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.set_xlabel("P₁ (Generator 1 Output)")
ax2.set_ylabel("P₂ (Generator 2 Output)")
ax2.set_title("Pareto Set: Generator Power Outputs with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/smart_grids_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
