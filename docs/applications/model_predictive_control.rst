Multi-Objective Model Predictive Control
========================================

In industrial automation and robotics, Model Predictive Control (MPC) and Linear Quadratic Regulators (LQR) are foundational algorithms used to optimize future behavior in real-time. Operating chemical plant processes, for instance, requires rigorously balancing product quality tracking (setpoint proximity) against the suppression of abrupt heater and valve modifications (machinery wear). 

Similarly, omnidirectional mobile robots traversing complex environments (e.g., hospitals or warehouses) must optimize trajectory tracking while minimizing energy consumption and avoiding collisions. However, they face significant physical constraints like nonlinear friction modes. Over a predictive horizon :math:`H`, these systems evaluate competing objectives to find the strictly optimal control sequence :math:`U`:

.. math:: 
   
   f_{track}(U) &= \sum_{k=1}^H \|y_k - r_k\|_Q^2 \\
   f_{smooth}(U) &= \sum_{k=0}^{H-1} \|\Delta u_k\|_R^2 \\
   f_{energy}(U) &= \sum_{k=0}^{H-1} \|u_k\|_S^2

where :math:`y_k` denotes system states/outputs, :math:`r_k` the target setpoints, :math:`u_k` the operational inputs, and :math:`\Delta u_k` the input change rates.

Crucially, as long as the weighting matrices (e.g., :math:`R`) acting on the independent control inputs are uniformly positive definite (:math:`R \succ 0`), the resulting multi-objective performance stage cost is strongly convex with respect to the continuous input decisions :math:`U`. Even in complex scenarios where discrete mode switching (like robot friction statuses) transforms the formulation into a Mixed-Integer Quadratic Program (MIQP), the continuous sub-component remains undeniably strongly convex, stabilizing bounded-node optimization searches.

In practice, resolving these multi-objective QP or MIQP problems during an active millisecond control loop requires prohibitive amounts of computational power. Since the strong convexity naturally spans a continuous simplicial Pareto front, controllers equipped with a Bézier simplex representation can bypass the online QP solver entirely. The polynomial evaluation provides an analytic, instantaneous multi-objective control response, enabling ultra-fast, high-frequency continuous re-optimization in highly unstable physical environments.

Numerical Experiments
---------------------

We demonstrate Bézier simplex fitting on a simple two-objective MPC problem over a horizon of :math:`H = 5` steps.

**Problem Setup:**

- 1D system: :math:`x(k+1) = x(k) + u(k)`, initial state :math:`x(0) = 0`, target :math:`r = 1`
- Control sequence: :math:`U = [u_0, u_1, u_2, u_3, u_4] \in \mathbb{R}^5`
- Competing objectives:

.. math::

   f_1(U) &= \sum_{k=0}^{4} (x(k+1) - r)^2 \quad \text{(tracking error)} \\
   f_2(U) &= \sum_{k=0}^{3} (u_{k+1} - u_k)^2 \quad \text{(control smoothness)}

**Experiment Procedure:**

1. Sample 10 weight vectors :math:`w = (w_1, w_2)` on the 1-simplex from :math:`(1,0)` to :math:`(0,1)`.
2. For each :math:`w`, solve :math:`U^*(w) = \arg\min_U [w_1 f_1(U) + w_2 f_2(U)]` using L-BFGS-B.
3. Collect the Pareto front points :math:`(f_1(U^*(w)), f_2(U^*(w)))`.
4. Fit a degree-3 Bézier simplex to the weight–objective pairs.
5. Visualize the fitted Bézier curve against the optimization-derived Pareto front.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/mpc_pareto_set.png
         :alt: Pareto set for MPC (control input space)
         :width: 100%
     - .. image:: ../_static/mpc_pareto.png
         :alt: Bézier simplex fitting to MPC Pareto front
         :width: 100%
   * - Pareto set: optimal control inputs :math:`U^*(w)` in decision space, traced as the weight :math:`w` moves from tracking-only to smoothness-only.
     - Pareto front: optimization-derived points (blue) and Bézier simplex approximation (red curve) in objective space.

The complete example script is available at :file:`examples/generate_mpc_pareto.py`.
