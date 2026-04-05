Medical Imaging and Radiation Therapy
=====================================

Optimizing diagnostic clarity and therapeutic safety simultaneously is paramount in medical fields, making multi-objective optimization frameworks indispensable.

For instance, Computed Tomography (CT) image reconstruction represents an ill-posed multi-objective inverse problem where doctors must minimize a data fidelity term (noise residual) against a regularization term safeguarding spatial continuity. Utilizing the L-Curve method :cite:p:`hansen1993use`, this is routinely formulated mathematically as balancing:

.. math:: 
   
   f_1(x) &= \|Kx - y\|_2^2 \quad \text{(Fidelity)} \\
   f_2(x) &= \|Lx\|_2^2     \quad \text{(Regularization)}

Similarly, Intensity-Modulated Radiation Therapy (IMRT) requires treating patients via highly constrained inverse planning :cite:p:`breedveld2007novel`. Planners must aggressively deliver prescriptive radiation doses to tumors while strictly repressing the doses affecting adjacent healthy Organs At Risk (OAR). The optimization minimizes heavily competing quadratic objectives:

.. math:: 
   
   s(f) = \sum_{v \in V} \xi_v (Hf - d_v^p)^T \tilde{\eta}_v (Hf - d_v^p) + \kappa (Mf)^T (Mf)

where the first component penalizes dose errors via beamlet fluence :math:`f`, and the second component applies spatial smoothing. 

In both clinical environments, strong convexity critically anchors the computation. For CT reconstructions, the explicit :math:`L_2` regularizations (via the parameter :math:`\lambda^2 \|Lx\|^2_2`) stabilize singular projection matrices, resulting in a definitively positive-definite system matrix :math:`\nabla^2 f \succeq \lambda I`. Correspondingly, in IMRT, smoothing parameters (like :math:`\kappa`) combined with dense beamlet mappings force the objective Hessian :math:`2(H^T \tilde{\eta} H + \kappa M^T M)` to remain strictly positive definite. 

In traditional practices, resolving the exact trade-offs requires hours of discrete algorithmic recalculations, delaying urgent therapies. However, by substituting the Pareto front mapping with a Bézier simplex, radiation oncologists and radiologists are granted an interactive, fully continuous digital slider. They can instantaneously sweep through the entire geometric space—trading off tumor destruction versus organ preservation precisely—in real-time without recalculating a single QP iteration.

Numerical Experiments
---------------------

We illustrate Bézier simplex fitting on a small CT reconstruction problem, tracing the L-curve between data fidelity and spatial smoothness.

**Problem Setup:**

- Image: :math:`x \in \mathbb{R}^8` (piecewise-constant signal :math:`x^* = [0,1,1,2,2,1,1,0]`)
- Forward operator: :math:`K \in \mathbb{R}^{6 \times 8}` (random projection, seed 42)
- Finite-difference operator: :math:`L \in \mathbb{R}^{7 \times 8}`
- Noisy measurement: :math:`y = Kx^* + \epsilon` (:math:`\epsilon \sim \mathcal{N}(0, 0.05^2)`)
- Regularized objectives:

.. math::

   f_1(x) &= \|Kx - y\|^2 + 0.01\|x\|^2 \quad \text{(fidelity)} \\
   f_2(x) &= \|Lx\|^2 + 0.01\|x\|^2 \quad \text{(smoothness)}

**Experiment Procedure:**

1. Sample 10 weight vectors :math:`w = (w_1, w_2)` on the 1-simplex from :math:`(1,0)` to :math:`(0,1)`.
2. For each :math:`w`, solve :math:`x^*(w) = \arg\min_x [w_1 f_1(x) + w_2 f_2(x)]` using L-BFGS-B.
3. Collect the Pareto front points :math:`(f_1(x^*(w)), f_2(x^*(w)))`.
4. Fit a degree-3 Bézier simplex to the weight–objective pairs.
5. Visualize the fitted Bézier curve against the L-curve sampled by optimization.

.. list-table::
   :widths: 50 50
   :align: center

   * - .. image:: ../_static/medical_imaging_pareto_set.png
         :alt: Pareto set for CT reconstruction (pixel space)
         :width: 100%
     - .. image:: ../_static/medical_imaging_pareto.png
         :alt: Bézier simplex fitting to CT reconstruction L-curve
         :width: 100%
   * - Pareto set: optimal reconstruction parameters :math:`x^*(w)` in pixel space, traced as the weight :math:`w` moves from fidelity-only to smoothness-only.
     - Pareto front: optimization-derived L-curve (blue dots) and Bézier simplex approximation (red curve) in objective space.

The complete example script is available at :file:`examples/generate_medical_imaging_pareto.py`.
