import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Define a two-objective facility location problem in 2D
# x represents the facility location
# a1 and a2 are two demand centers

a1 = np.array([0.0, 0.0])
a2 = np.array([1.0, 1.0])


def f1(x):
    return np.sum((x - a1) ** 2)


def f2(x):
    return np.sum((x - a2) ** 2)


def f(x, w):
    return w[0] * f1(x) + w[1] * f2(x)

# Sample 10 weights from w=(1,0) to w=(0,1)
weights = np.linspace(0.0, 1.0, 10)
optimals = []
for t in weights:
    w = np.array([1.0 - t, t])
    result = minimize(lambda x: f(x, w), x0=np.array([0.5, 0.5]), method="L-BFGS-B")
    x_opt = result.x
    optimals.append((w, x_opt, f1(x_opt), f2(x_opt)))

# Prepare training data for Bézier simplex fitting
w1 = np.array([p[0][0] for p in optimals])
w2 = np.array([p[0][1] for p in optimals])
X = np.column_stack([w1, w2])
y = np.column_stack([np.array([p[2] for p in optimals]), np.array([p[3] for p in optimals])])

# Fit a degree-3 Bézier simplex to the Pareto front
regressor = BezierSimplexRegressor(degree=3)
regressor.fit(X, y)

# Visualize the optimization-derived Pareto points and the fitted Bézier curve
t_smooth = np.linspace(0.0, 1.0, 200)
w1_smooth = 1.0 - t_smooth
w2_smooth = t_smooth
X_smooth = np.column_stack([w1_smooth, w2_smooth])
y_smooth = regressor.predict(X_smooth)

# Compute approximation error on training points
pred_train = regressor.predict(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y[:, 0], y[:, 1], color="blue", label="Optimization-derived Pareto points")
ax.plot(y_smooth[:, 0], y_smooth[:, 1], color="red", label="Fitted Bézier simplex")
ax.set_xlabel("Distance to demand center A (f₁)")
ax.set_ylabel("Distance to demand center B (f₂)")
ax.set_title("Two-Objective Facility Location Pareto Front: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
fig.savefig("docs/_static/facility_location_pareto_2obj.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")
# Visualize the Pareto set (x locations)
x_targets = np.array([p[1] for p in optimals])  # x_opt for each w
regressor_x = BezierSimplexRegressor(degree=3)
regressor_x.fit(X, x_targets)

# Predict smooth x locations
x_smooth = regressor_x.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x_targets[:, 0], x_targets[:, 1], color="green", label="Pareto set locations (samples)", s=50)
ax2.plot(x_smooth[:, 0], x_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.scatter([a1[0], a2[0]], [a1[1], a2[1]], color="red", marker="x", s=100, label="Demand centers")
ax2.set_xlabel("x₁")
ax2.set_ylabel("x₂")
ax2.set_title("Pareto Set: Facility Locations with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/facility_location_pareto_set_2obj.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")