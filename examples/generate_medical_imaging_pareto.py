import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# CT reconstruction: fidelity vs. regularization (L-curve experiment)
# n=8 pixels
n = 8
rng = np.random.default_rng(42)

# Random projection matrix K (shape 6x8) and finite-difference matrix L (shape 7x8)
K = rng.standard_normal((6, n))
L = np.zeros((n - 1, n))
for i in range(n - 1):
    L[i, i] = -1.0
    L[i, i + 1] = 1.0

# True image: piecewise constant
x_true = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0])

# Noisy measurement
y_obs = K @ x_true + 0.05 * rng.standard_normal(6)


def f1(x):
    # Data fidelity + small L2 regularization
    return np.sum((K @ x - y_obs) ** 2) + 0.01 * np.sum(x ** 2)


def f2(x):
    # Smoothness (total variation via finite differences) + small L2
    return np.sum((L @ x) ** 2) + 0.01 * np.sum(x ** 2)


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
        x0=np.zeros(n),
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
ax.set_xlabel("Data Fidelity (f₁)")
ax.set_ylabel("Smoothness Regularization (f₂)")
ax.set_title("CT Reconstruction L-Curve: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/medical_imaging_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

# Fit Bézier simplex to Pareto set (reconstruction space, showing first two pixels)
z_targets = np.array([p[1] for p in optimals])
regressor_set = BezierSimplexRegressor(degree=3)
regressor_set.fit(X, z_targets)
z_smooth = regressor_set.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(z_targets[:, 0], z_targets[:, 1], color="green", label="Optimal reconstructions (samples)", s=50)
ax2.plot(z_smooth[:, 0], z_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.axhline(x_true[1], color="gray", linestyle="--", alpha=0.5, label=f"True pixel 2 = {x_true[1]:.1f}")
ax2.axvline(x_true[0], color="gray", linestyle=":", alpha=0.5, label=f"True pixel 1 = {x_true[0]:.1f}")
ax2.set_xlabel("Pixel 1 (x₁)")
ax2.set_ylabel("Pixel 2 (x₂)")
ax2.set_title("Pareto Set: CT Reconstruction Parameters with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/medical_imaging_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
