import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Define three convex objective functions for a simplified routing problem
# x = [x1, x2] representing routing allocations

def f1(x):
    return x[0] ** 2 + x[1] ** 2


def f2(x):
    prev = np.array([0.5, 0.5])
    return np.sum((x - prev) ** 2)


def f3(x):
    target = np.array([1.0, 0.2])
    return 0.8 * np.sum((x - target) ** 2)


def f(x, w):
    return w[0] * f1(x) + w[1] * f2(x) + w[2] * f3(x)

# 10 deterministic weights on the 2-simplex
w_list = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1 / 3, 1 / 3, 1 / 3],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
    ]
)

# Solve the convex weighted optimization problem for each weight
optimals = []
for w in w_list:
    x0 = np.array([0.5, 0.5])
    result = minimize(lambda x: f(x, w), x0, method="L-BFGS-B")
    x_opt = result.x
    optimals.append((w, x_opt, f1(x_opt), f2(x_opt), f3(x_opt)))

optimal_values = np.array([[p[2], p[3], p[4]] for p in optimals])

# Fit a Bézier simplex to the weight-objective mapping
X = w_list
y = optimal_values
regressor = BezierSimplexRegressor(degree=3)
regressor.fit(X, y)

# Generate a smooth set of weights across the simplex edge for visualization
n_plot = 200
alpha = np.linspace(0, 1, n_plot)
beta = np.linspace(0, 1, n_plot)
weights_smooth = []
for a in alpha:
    for b in beta:
        if a + b <= 1.0:
            weights_smooth.append([1 - a - b, a, b])
weights_smooth = np.array(weights_smooth)

predicted = regressor.predict(weights_smooth)

# Compare fitted values on training samples
pred_train = regressor.predict(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training point error: {max_error:.4f}")

# Plot 3D Pareto front approximation
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(y[:, 0], y[:, 1], y[:, 2], color="blue", label="Optimization-derived points", s=40)
ax.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], color="red", alpha=0.15, s=10, label="Bézier simplex approximation")
ax.set_xlabel("f₁: Congestion Excess")
ax.set_ylabel("f₂: Route Volatility")
ax.set_zlabel("f₃: Path Imbalance")
ax.set_title("Three-objective Pareto Front: Optimization vs Bézier Simplex")
ax.legend()

plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/communication_pareto_3obj.png", dpi=150, bbox_inches="tight")

# Fit Bézier simplex to Pareto set (routing allocation space)
x_targets = np.array([p[1] for p in optimals])
regressor_set = BezierSimplexRegressor(degree=3)
regressor_set.fit(X, x_targets)
x_smooth = regressor_set.predict(weights_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x_targets[:, 0], x_targets[:, 1], color="green", label="Optimal routing allocations (samples)", s=50)
ax2.scatter(x_smooth[:, 0], x_smooth[:, 1], color="orange", alpha=0.15, s=10, label="Bézier simplex approximation")
ax2.set_xlabel("x₁ (Routing Allocation 1)")
ax2.set_ylabel("x₂ (Routing Allocation 2)")
ax2.set_title("Pareto Set: Routing Allocations with Bézier Approximation (3 Objectives)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/communication_pareto_set_3obj.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")