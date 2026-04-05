import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

# Define objective functions for communication routing example
# x = [x1, x2] representing routing allocations
def f1(x):
    # Congestion excess: penalize deviation from target capacity
    return x[0]**2 + x[1]**2

def f2(x):
    # Route volatility: penalize deviation from previous allocation (assume previous was [0.5, 0.5])
    prev = np.array([0.5, 0.5])
    return np.sum((x - prev)**2)

def f(x, w):
    # Weighted sum: w[0]*f1 + w[1]*f2
    return w[0] * f1(x) + w[1] * f2(x)

# Sample weights from w=(1,0) to w=(0,1)
n_samples = 10
weights = np.linspace(0, 1, n_samples)
w_list = [(1-w, w) for w in weights]

# For each weight, find optimal x
optimal_points = []
for w in w_list:
    # Initial guess
    x0 = np.array([0.5, 0.5])
    # Minimize f(x, w)
    result = minimize(lambda x: f(x, w), x0, method='L-BFGS-B')
    x_opt = result.x
    f1_val = f1(x_opt)
    f2_val = f2(x_opt)
    optimal_points.append((w, x_opt, f1_val, f2_val))

# Extract data for fitting
w1_vals = np.array([p[0][0] for p in optimal_points])
w2_vals = np.array([p[0][1] for p in optimal_points])
f1_targets = np.array([p[2] for p in optimal_points])
f2_targets = np.array([p[3] for p in optimal_points])

# Fit Bezier simplex (multi-output: predict f1 and f2)
X = np.column_stack([w1_vals, w2_vals])
y = np.column_stack([f1_targets, f2_targets])
regressor = BezierSimplexRegressor(degree=3)
regressor.fit(X, y)

# Generate smooth curve for visualization
t_smooth = np.linspace(0, 1, 100)
w1_smooth = 1 - t_smooth
w2_smooth = t_smooth
X_smooth = np.column_stack([w1_smooth, w2_smooth])
y_smooth = regressor.predict(X_smooth)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(f1_targets, f2_targets, color='blue', label='Optimal points from optimization')
ax.plot(y_smooth[:, 0], y_smooth[:, 1], 'r-', label='Fitted Bézier simplex')
ax.set_xlabel('Congestion Excess (f₁)')
ax.set_ylabel('Route Volatility (f₂)')
ax.set_title('Pareto Front: Optimization vs Bézier Simplex Fitting')
ax.legend()
ax.grid(True, alpha=0.3)

# Calculate approximation error on the training points
y_pred = regressor.predict(X)
errors = np.sqrt((y_pred[:, 0] - y[:, 0])**2 +
                 (y_pred[:, 1] - y[:, 1])**2)
max_error = np.max(errors)
print(f'Max approximation error: {max_error:.4f}')

plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig('docs/_static/communication_fitting.png', dpi=150, bbox_inches='tight')

# Fit Bézier simplex to Pareto set (routing allocation space)
x_targets = np.array([p[1] for p in optimal_points])
regressor_set = BezierSimplexRegressor(degree=3)
regressor_set.fit(X, x_targets)
x_smooth = regressor_set.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x_targets[:, 0], x_targets[:, 1], color="green", label="Optimal routing allocations (samples)", s=50)
ax2.plot(x_smooth[:, 0], x_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.set_xlabel("x₁ (Routing Allocation 1)")
ax2.set_ylabel("x₂ (Routing Allocation 2)")
ax2.set_title("Pareto Set: Routing Allocations with Bézier Approximation")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/communication_pareto_set.png", dpi=150, bbox_inches='tight')
print("Pareto set plot saved.")