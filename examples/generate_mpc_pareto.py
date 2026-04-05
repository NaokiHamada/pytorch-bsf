import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Simple 2-objective MPC: tracking vs. control smoothness
# 1D system: x(k+1) = x(k) + u(k), target r=1, horizon H=5
# U = [u0, u1, u2, u3, u4] in R^5, x(0) = 0
r = 1.0
H = 5


def simulate(U):
    """Return state trajectory x(1)..x(H) given x(0)=0 and inputs U."""
    x = np.zeros(H + 1)
    for k in range(H):
        x[k + 1] = x[k] + U[k]
    return x


def f1(U):
    # Tracking error: sum_{k=1}^{H} (x(k) - r)^2
    x = simulate(U)
    return np.sum((x[1:] - r) ** 2)


def f2(U):
    # Control smoothness: sum_{k=0}^{H-2} (U(k+1) - U(k))^2
    return np.sum(np.diff(U) ** 2)


def f(U, w):
    return w[0] * f1(U) + w[1] * f2(U)


# Sample 10 weights on the 1-simplex from w=(1,0) to w=(0,1)
n_samples = 10
weights = np.linspace(0.0, 1.0, n_samples)
optimals = []
for t in weights:
    w = np.array([1.0 - t, t])
    result = minimize(
        lambda U: f(U, w),
        x0=np.zeros(H),
        method="L-BFGS-B",
    )
    U_opt = result.x
    optimals.append((w, U_opt, f1(U_opt), f2(U_opt)))

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
ax.set_xlabel("Tracking Error (f₁)")
ax.set_ylabel("Control Smoothness (f₂)")
ax.set_title("MPC Pareto Front: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/mpc_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

# Fit Bézier simplex to Pareto set (control input space, showing first two steps)
U_targets = np.array([p[1] for p in optimals])
regressor_set = BezierSimplexRegressor(degree=3)
regressor_set.fit(X, U_targets)
U_smooth = regressor_set.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(U_targets[:, 0], U_targets[:, 1], color="green", label="Optimal control inputs (samples)", s=50)
ax2.plot(U_smooth[:, 0], U_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.set_xlabel("u₀ (Control Input at Step 0)")
ax2.set_ylabel("u₁ (Control Input at Step 1)")
ax2.set_title("Pareto Set: MPC Control Inputs with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/mpc_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
