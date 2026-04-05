import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Two-task federated learning with quadratic losses
# theta in R^2: shared model parameters
theta1_star = np.array([1.0, 0.5])   # Task 1 target
theta2_star = np.array([-0.5, 1.0])  # Task 2 target
lam = 0.1


def f1(theta):
    # Task 1 regularized loss
    return np.sum((theta - theta1_star) ** 2) + lam * np.sum(theta ** 2)


def f2(theta):
    # Task 2 regularized loss
    return np.sum((theta - theta2_star) ** 2) + lam * np.sum(theta ** 2)


def f(theta, w):
    return w[0] * f1(theta) + w[1] * f2(theta)


# Sample 10 weights on the 1-simplex from w=(1,0) to w=(0,1)
n_samples = 10
weights = np.linspace(0.0, 1.0, n_samples)
optimals = []
for t in weights:
    w = np.array([1.0 - t, t])
    result = minimize(
        lambda theta: f(theta, w),
        x0=np.zeros(2),
        method="L-BFGS-B",
    )
    theta_opt = result.x
    optimals.append((w, theta_opt, f1(theta_opt), f2(theta_opt)))

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
ax.set_xlabel("Task 1 Regularized Loss (f₁)")
ax.set_ylabel("Task 2 Regularized Loss (f₂)")
ax.set_title("Federated Learning Pareto Front: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/federated_learning_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

# Fit Bézier simplex to Pareto set (model parameter space)
theta_targets = np.array([p[1] for p in optimals])
regressor_set = BezierSimplexRegressor(degree=3)
regressor_set.fit(X, theta_targets)
theta_smooth = regressor_set.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(theta_targets[:, 0], theta_targets[:, 1], color="green", label="Optimal parameters (samples)", s=50)
ax2.plot(theta_smooth[:, 0], theta_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.scatter([theta1_star[0]], [theta1_star[1]], color="blue", marker="*", s=200, zorder=5, label="Task 1 target θ₁*")
ax2.scatter([theta2_star[0]], [theta2_star[1]], color="red", marker="*", s=200, zorder=5, label="Task 2 target θ₂*")
ax2.set_xlabel("θ₁")
ax2.set_ylabel("θ₂")
ax2.set_title("Pareto Set: Model Parameters with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/federated_learning_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
