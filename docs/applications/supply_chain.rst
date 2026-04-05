Supply Chain and Logistics Optimization
=======================================

In modern supply chain management (SCM) and operations research, isolating physical material logistics from global financial flows regularly causes massive inefficiencies. Supply Chain Finance algorithms attempt to computationally integrate factory outputs, inventory storage, and monetary streams to achieve a single unified, profitable coordination strategy. 

Recent operational formulations actively model this extensive corporate balancing act as a purely convex, constrained, finite-horizon Linear Quadratic Regulator (LQR) problem over discrete stages.

The optimization generally maps out three tightly competing operational objectives across the logistics network:

.. math:: 
   
   f_1(U) &= \sum_{k=0}^{H} \|X_k - X_k^{\text{target}}\|_Q^2 \quad \text{(State deviations)} \\
   f_2(U) &= \sum_{k=0}^{H-1} \|U_k\|_R^2 \quad \text{(Financial/operational inputs)} \\
   f_3(U) &= \sum_{k=0}^{H-1} \|\Delta U_k\|_S^2 \quad \text{(Policy change instability)}

Where :math:`X_k` represents the aggregate cash and inventory states, constrained to remain close to their targets without causing disruptive shortages, and :math:`U_k` are the direct financial interventions or operational accelerations.

For manufacturing processes and logistics, abrupt disruptions in financing inputs or delivery capacities incur massive overheads, making the penalization of policy volatility (:math:`\|\Delta U_k\|_S^2`) practically vital for securing stable operational margins. 

Because the entire LQR overarching formulation relies exclusively on positive definite penalty matrices (:math:`Q, R, S \succ 0`), the extensive supply chain network optimization mathematically remains entirely and globally strongly convex.

When corporate managers visualize these strategic relationships statically, they often suffer from disjointed discrete scenario planning. Implementing a Bézier simplex surrogate that models the multi-objective continuous terrain flawlessly allows disparate corporate departments—ranging from financial officers to warehouse managers—to visually navigate through a single, continuous interactive Pareto topology in boardroom meetings, instantly resolving the optimal, stable path for navigating market supply shocks without requiring live, intensive QP solver re-computations.

Numerical Experiments
---------------------

We demonstrate Bézier simplex fitting on a two-objective inventory LQR problem balancing state deviation against policy volatility over a horizon of :math:`H = 5` periods.

**Problem Setup:**

- Ordering decisions: :math:`U = [u_0, u_1, u_2, u_3, u_4] \in \mathbb{R}^5`
- Inventory dynamics: :math:`x(k+1) = x(k) + u(k)`, :math:`x(0) = 0`, target :math:`x^* = 1`
- Competing objectives:

.. math::

   f_1(U) &= \sum_{k=0}^{4} (x(k+1) - x^*)^2 \quad \text{(state deviation)} \\
   f_2(U) &= \sum_{k=1}^{4} (u_k - u_{k-1})^2 + 0.01\|U\|^2 \quad \text{(policy volatility)}

**Experiment Procedure:**

1. Sample 10 weight vectors :math:`w = (w_1, w_2)` on the 1-simplex from :math:`(1,0)` to :math:`(0,1)`.
2. For each :math:`w`, solve :math:`U^*(w) = \arg\min_U [w_1 f_1(U) + w_2 f_2(U)]` using L-BFGS-B.
3. Collect the Pareto front points :math:`(f_1(U^*(w)), f_2(U^*(w)))`.
4. Fit a degree-3 Bézier simplex to the weight–objective pairs.
5. Visualize the fitted Bézier curve against the optimization-derived Pareto front.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/supply_chain_pareto_set.png
         :alt: Pareto set for supply chain LQR (ordering decision space)
         :width: 100%
     - .. image:: ../_static/supply_chain_pareto.png
         :alt: Bézier simplex fitting to supply chain LQR Pareto front
         :width: 100%
   * - Pareto set: optimal order quantities :math:`U^*(w)` in decision space, traced as the weight :math:`w` moves from state-deviation-only to policy-volatility-only.
     - Pareto front: optimization-derived points (blue) and Bézier simplex approximation (red curve) in objective space.

The complete example script is available at :file:`examples/generate_supply_chain_pareto.py`.
