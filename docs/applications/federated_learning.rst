Multi-Task and Federated Learning
=================================

In modern machine learning, models are frequently tasked with balancing strictly competing performance indicators. In multi-task learning (MTL), a single shared neural representation must simultaneously optimize losses across different tasks, often leading to "gradient conflict" where improving one task substantially degrades another. This same underlying tension appears when training models with explicit fairness constraints, where the expected loss over different population subgroups forms a multi-objective optimization problem. 

Furthermore, this principle is critically important in Federated Learning (FL), such as the MOCHA framework, where massive data is distributed across edge devices and a global model must balance diverse local losses against consensus.

By imposing strongly convex regularizations (e.g., Tikhonov/Ridge) on the model parameters alongside the local losses, the optimization formulations become naturally resolvable:

.. math:: 
   
   f_1(\theta) &= \mathcal{L}_1(\theta) + \lambda\|\theta\|^2 \\
   f_2(\theta) &= \mathcal{L}_2(\theta) + \lambda\|\theta\|^2 \\
               &\vdots \\
   f_m(\theta) &= \mathcal{L}_m(\theta) + \lambda\|\theta\|^2

The :math:`L_2` regularization penalty :math:`\lambda\|\theta\|^2` ensures that the Hessian incorporates a positive definite matrix :math:`2\lambda I \succ 0`. This global penalty forcefully transforms the generic convex loss landscapes (common in logistic/ridge regressions or SVMs) into strictly strongly convex objectives. Under strong convexity, algorithms like the Stochastic Multi-Gradient (SMG) descent achieve dramatic convergence accelerations, improving from sub-linear to linear :math:`O(1/n)` or :math:`O(\exp(-\mu T))` convergence rates :cite:p:`liu2024stochastic`.

Rather than computing single compromised aggregations, the global model's Pareto front is modeled as a continuous Bézier simplex. This facilitates a "train once, customize anywhere" paradigm. A global Bézier simplex allows individual edge devices or network nodes to download the analytic continuous trade-off surface, enabling them to instantly select an optimal model variant tailored precisely to their local power limits, fairness constraints, or inference needs, simply by navigating the simplex variables—representing a drastic improvement over constant network-wide retraining.

Numerical Experiments
---------------------

We demonstrate Bézier simplex fitting on a two-task federated learning problem with quadratic losses and Tikhonov regularization.

**Problem Setup:**

- Model parameters: :math:`\theta \in \mathbb{R}^2`
- Task targets: :math:`\theta_1^* = [1.0, 0.5]`, :math:`\theta_2^* = [-0.5, 1.0]`
- Regularized task losses with :math:`\lambda = 0.1`:

.. math::

   f_1(\theta) &= \|\theta - \theta_1^*\|^2 + 0.1\|\theta\|^2 \\
   f_2(\theta) &= \|\theta - \theta_2^*\|^2 + 0.1\|\theta\|^2

**Experiment Procedure:**

1. Sample 10 weight vectors :math:`w = (w_1, w_2)` on the 1-simplex from :math:`(1,0)` to :math:`(0,1)`.
2. For each :math:`w`, solve :math:`\theta^*(w) = \arg\min_\theta [w_1 f_1(\theta) + w_2 f_2(\theta)]` using L-BFGS-B.
3. Collect the Pareto front points :math:`(f_1(\theta^*(w)), f_2(\theta^*(w)))`.
4. Fit a degree-3 Bézier simplex to the weight–loss pairs.
5. Visualize the fitted Bézier curve against the optimization-derived Pareto front.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/federated_learning_pareto_set.png
         :alt: Pareto set for two-task federated learning (model parameter space)
         :width: 100%
     - .. image:: ../_static/federated_learning_pareto.png
         :alt: Bézier simplex fitting to two-task federated learning Pareto front
         :width: 100%
   * - Pareto set: optimal model parameters :math:`\theta^*(w)` in parameter space, with task targets :math:`\theta_1^*` and :math:`\theta_2^*` shown as stars.
     - Pareto front: optimization-derived points (blue) and Bézier simplex approximation (red curve) in objective space.

The complete example script is available at :file:`examples/generate_federated_learning_pareto.py`.
