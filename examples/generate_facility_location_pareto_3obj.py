import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Define a three-objective facility location problem in 2D
# x represents the facility location
# a1, a2, a3 are three demand centers

a1 = np.array([0.0, 0.0])
a2 = np.array([1.0, 0.0])
a3 = np.array([0.5, 1.0])


def f1(x):
    return np.sum((x - a1) ** 2)


def f2(x):
    return np.sum((x - a2) ** 2)


def f3(x):
    return np.sum((x - a3) ** 2)


def f(x, w):
    return w[0] * f1(x) + w[1] * f2(x) + w[2] * f3(x)

# Sample 10 weight vectors on the 2-simplex
w_list = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.5],
    [1/3, 1/3, 1/3],
    [0.7, 0.2, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
])

optimals = []
for w in w_list:
    result = minimize(lambda x: f(x, w), x0=np.array([0.5, 0.5]), method="L-BFGS-B")
    x_opt = result.x
    optimals.append((w, x_opt, f1(x_opt), f2(x_opt), f3(x_opt)))

# Prepare training data for Bézier simplex fitting
X = np.array([p[0] for p in optimals])
y = np.array([[p[2], p[3], p[4]] for p in optimals])

# Fit a degree-3 Bézier simplex to the Pareto front
regressor = BezierSimplexRegressor(degree=3)
regressor.fit(X, y)

# Generate smooth weights for visualization
n_plot = 50
weights_smooth = []
for i in range(n_plot):
    for j in range(n_plot - i):
        w1 = i / (n_plot - 1)
        w2 = j / (n_plot - 1)
        w3 = 1 - w1 - w2
        if w3 >= 0:
            weights_smooth.append([w1, w2, w3])
weights_smooth = np.array(weights_smooth)

predicted = regressor.predict(weights_smooth)

print("Predicted shape:", predicted.shape)

# Compute approximation error on training points
pred_train = regressor.predict(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")

# Visualize in 3D
print("Generating plot...")
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(y[:, 0], y[:, 1], y[:, 2], color="blue", label="Optimization-derived Pareto points", s=40)
ax.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], color="red", alpha=0.15, s=10, label="Bézier simplex approximation")
ax.set_xlabel("Distance to demand center A (f₁)")
ax.set_ylabel("Distance to demand center B (f₂)")
ax.set_zlabel("Distance to demand center C (f₃)")
ax.set_title("Three-Objective Facility Location Pareto Front: Optimization vs Bézier Simplex")
ax.legend()

plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/facility_location_pareto_3obj.png", dpi=150, bbox_inches="tight")
print("Plot saved.")
# plt.show()

# Visualize the Pareto set (x locations)
x_targets = np.array([p[1] for p in optimals])  # x_opt for each w
regressor_x = BezierSimplexRegressor(degree=3)
regressor_x.fit(X, x_targets)

# Predict smooth x locations
x_smooth = regressor_x.predict(weights_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x_targets[:, 0], x_targets[:, 1], color="green", label="Pareto set locations (samples)", s=50)
ax2.scatter(x_smooth[:, 0], x_smooth[:, 1], color="orange", alpha=0.15, s=10, label="Bézier simplex approximation")
ax2.scatter([a1[0], a2[0], a3[0]], [a1[1], a2[1], a3[1]], color="red", marker="x", s=100, label="Demand centers")
ax2.set_xlabel("x₁")
ax2.set_ylabel("x₂")
ax2.set_title("Pareto Set: Facility Locations with Bézier Approximation (3 Objectives)")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/facility_location_pareto_set_3obj.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")