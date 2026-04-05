Robust Portfolio Management
===========================

In financial engineering and asset management, the most classical real-world application is Markowitz's mean-variance portfolio optimization :cite:p:`markowitz1952portfolio`. Investors and fund managers seek to simultaneously maximize expected returns and minimize risk (often modeled as the variance of returns). This framework is widely utilized by institutional pension funds (e.g., GPIF), treasury departments, and risk management teams to quantify trade-offs and build consensus on asset allocation decisions.

However, in practice, estimating the covariance matrix from limited observations or highly collinear assets often leads to numerical instability. To resolve this, it is common to introduce a strongly convex regularization term, such as an :math:`L_2` norm penalty on the asset allocation weights or a turnover penalty to suppress excessive trading compared to a previous portfolio :math:`x^{\text{prev}}` :cite:p:`qi2026optimal`.

This formulation effectively creates a three-objective optimization problem over the allocation weights :math:`x \in \mathbb{R}^n`: 

.. math::

   \text{Expected Return: } & f_1(x) = -\mu^T x \\
   \text{Risk (Variance): } & f_2(x) = x^T \Sigma x \\
   \text{Stability (Turnover): } & f_3(x) = \lambda \|x - x^{\text{prev}}\|_2^2

Subject to classical constraints such as the budget constraint :math:`\mathbf{1}^T x = 1` and long-only constraints :math:`x \ge 0`.

Because the regularization term is strongly convex, the resulting scalarized objective function becomes strictly strongly convex. Even if the covariance matrix :math:`\Sigma` is only positive semi-definite, the addition of the turnover penalty guarantees that the Hessian :math:`\nabla^2 f(x) \succeq 2\lambda I` is strictly positive definite. According to the theorems in :cite:p:`mizota2021unconstrained`, this strongly convex problem is guaranteed to be weakly simplicial. 

Fitting a Bézier simplex to this problem allows practitioners to continuously map the entire robust Pareto front—namely, the continuous efficient frontier—guaranteeing unique solutions. Instead of computing disconnected discrete point clouds using weighted sums, analysts obtain a functionally continuous mapping of allocations versus risk. Utilizing sensitivity-based Newton path-following, the full Pareto front can be computed with an extremely efficient iteration complexity of :math:`O(p \log(1/\varepsilon))` :cite:p:`bergou2021complexity`. 
Solvers like MOSEK, Gurobi, and OSQP are widely used to efficiently compute these strongly convex QP subproblems. Modern extensions also include CVaR (Conditional Value-at-Risk) portfolio optimization and robust multi-objective portfolio frameworks under data uncertainty.

Numerical Experiments
---------------------

To illustrate Bézier simplex fitting on a portfolio Pareto front, we conducted an experiment using a three-asset mean-variance problem with L2 regularization to ensure strong convexity.

**Problem Setup:**

- Decision variable: :math:`x \in \mathbb{R}^3` (portfolio weights, unconstrained)
- Expected returns: :math:`\mu = [0.05, 0.12, 0.08]`
- Asset variances: :math:`\text{diag}(\Sigma) = [0.10, 0.15, 0.08]`
- Regularized objectives:

.. math::

   f_1(x) &= -\mu^T x + 0.05\|x\|^2 \quad \text{(negative return + L2)} \\
   f_2(x) &= x^T \mathrm{diag}(\Sigma) x + 0.05\|x\|^2 \quad \text{(variance + L2)}

**Experiment Procedure:**

1. Sample 10 weight vectors :math:`w = (w_1, w_2)` uniformly on the 1-simplex from :math:`(1,0)` to :math:`(0,1)`.
2. For each :math:`w`, solve :math:`x^*(w) = \arg\min_x [w_1 f_1(x) + w_2 f_2(x)]` using L-BFGS-B.
3. Collect the Pareto front points :math:`(f_1(x^*(w)), f_2(x^*(w)))`.
4. Fit a degree-3 Bézier simplex to the weight–objective pairs.
5. Visualize the fitted Bézier curve against the optimization-derived Pareto front.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/portfolio_pareto_set.png
         :alt: Pareto set for 3-asset portfolio (asset allocation space)
         :width: 100%
     - .. image:: ../_static/portfolio_pareto.png
         :alt: Bézier simplex fitting to 3-asset portfolio Pareto front
         :width: 100%
   * - Pareto set: optimal asset allocations :math:`x^*(w)` traced in decision space as the weight :math:`w` moves from :math:`(1,0)` to :math:`(0,1)`.
     - Pareto front: optimization-derived points (blue) and Bézier simplex approximation (red curve) in objective space.

The complete example script is available at :file:`examples/generate_portfolio_pareto.py`.
