Frequently Asked Questions
==========================

This page provides answers to common questions about PyTorch-BSF, its mathematical foundations, and practical usage.


General Information
-------------------

How does PyTorch-BSF differ from other hyperparameter optimization tools?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The key difference is that PyTorch-BSF **exploits problem structure** rather than treating the objective as a black box.

*   **Dramatically fewer evaluations:** Black-box methods such as Bayesian optimization make no assumptions about the objective and must explore the search space from scratch. Approximating a Pareto front to reasonable accuracy can require hundreds of evaluations. Because PyTorch-BSF assumes the problem is *weakly simplicial*, it can often recover the entire Pareto front from as few as 50 points with higher accuracy.
*   **Regression-based approach:** Unlike search methods that find discrete points, PyTorch-BSF fits a continuous parametric surface (a Bézier simplex). Once trained, you can evaluate any point on the trade-off surface instantly.
*   **Dimension-free convergence:** When data lie along a low-dimensional manifold embedded in a high-dimensional space, the convergence rate depends on the **intrinsic dimension** of the simplex, not on the ambient space dimension. This avoids the curse of dimensionality common in black-box methods.

What applications are there beyond multi-objective optimization?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While primarily used for Pareto front approximation, Bézier simplex fitting is a general-purpose regression technique for any continuous map from a simplex to a Euclidean space. Potential applications include:

*   **Interpolation of parametric families:** When a model's behavior varies continuously with coefficients on a simplex (e.g., mixture weights, regularization strengths in Elastic Net), a Bézier simplex can compactly represent the entire family.
*   **Shape modeling:** Bézier simplices generalize Bézier triangles used in CAD and computer graphics; they can represent smooth curved surfaces of any dimension.
*   **Solution manifolds:** Any problem whose solution set forms a continuous simplex-structured manifold is a candidate for fitting.
*   **Scientific data fitting:** Modeling physical phenomena where constraints naturally form a simplex (e.g., chemical concentrations in a mixture).


Mathematical Foundations
------------------------

What is the "weakly simplicial" assumption?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A problem is **weakly simplicial** if its Pareto set (and Pareto front) is the continuous image of a standard simplex. Topologically, this means the set of optimal trade-offs has no "holes" or disconnected components and can be "stretched" or "bent" from a simplex.

This assumption is remarkably broad. For example, it has been mathematically proven that **all unconstrained strongly convex optimization problems are weakly simplicial**. This covers a wide class of practical problems, including Elastic Net regression, Ridge regression, and many regularized empirical risk minimization tasks. See the :doc:`whatis` section for formal definitions.

Can I verify the "weakly simplicial" assumption for my problem?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes. If your problem is unconstrained and strongly convex, it is guaranteed to be weakly simplicial. For other cases, you can use a **data-driven statistical test** based on persistent homology.

The test checks whether the topology of the sampled Pareto set is consistent with a simplex structure. If the test rejects the simplicial hypothesis, a Bézier simplex model may not be appropriate. If it does not reject, you have statistical evidence supporting the use of PyTorch-BSF. Detailed information on these tests can be found in :cite:t:`hamada2018data` and :cite:t:`hamada2020test`.


Practical Usage
---------------

How do I choose the degree and estimate the required sample size?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The complexity of a Bézier simplex is determined by its **degree** (:math:`D`) and the number of **objectives** (:math:`M`). The number of control points is given by the formula:

.. math:: N_{cp} = \binom{D+M-1}{M-1}

**Guidelines:**

*   **Start low:** A degree of 2 or 3 is usually a good starting point. Low-degree models are less prone to overfitting and faster to train.
*   **Sample size:** You need at least as many training samples as there are control points (:math:`N_{cp}`) for the problem to be well-determined. In practice, having **2 to 3 times as many samples** as control points leads to more stable and reliable fits.
*   **Refine as needed:** If the residuals (fitting errors) are high, increase the degree. If the model overfits (low training error but poor generalization), increase the sample size or decrease the degree.

How do I normalize my parameters or values?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``fit()`` function requires that each row of the ``params`` tensor sums to 1 (i.e., lies on the standard simplex :math:`\Delta^{M-1}`). If your raw parameters don't satisfy this, you must normalize them manually.

Additionally, normalizing the output ``values`` can improve fitting stability and accuracy. PyTorch-BSF provides several options for automatic value normalization in the CLI/MLflow interface.

Please refer to the :doc:`advanced/normalization` page for detailed instructions on how to normalize your parameter and value tensors.

Can I use GPU or multi-node training?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Yes.** Since PyTorch-BSF is built on PyTorch and PyTorch Lightning, it supports hardware acceleration and distributed training out of the box.

Please refer to the :doc:`advanced/acceleration` page for detailed instructions on using GPUs (single or multiple), multi-node clusters, and mixed-precision training.

How do I save and load a trained model?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can save and load models using the ``torch_bsf.bezier_simplex`` module. Supported formats include ``.pt`` (PyTorch), ``.csv``, ``.tsv``, ``.json``, and ``.yaml``.

.. code-block:: python

   from torch_bsf.bezier_simplex import save, load

   # Save the trained model
   save("my_model.pt", bs)

   # Load it back
   bs = load("my_model.pt")

How can I perform cross-validation or grid search?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

PyTorch-BSF includes built-in tools for model selection and specific task automation:

*   **K-Fold Cross-Validation:** Run `python -m torch_bsf.model_selection.kfold` to evaluate model performance across different data splits.
*   **Elastic Net Grid Search:** Run `python -m torch_bsf.model_selection.elastic_net_grid` to generate parameter grids specifically for Elastic Net regularization paths.

When should I use the ``freeze`` argument?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``freeze`` argument allows you to hold specific control points constant during training. This is useful for:

*   **Boundary constraints:** If you know the exact values at the vertices of the simplex (e.g., results of single-objective optimizations), freeze those vertices and fit only the interior.
*   **Incremental refinement:** Fit a low-degree model first, then use its control points as initialization for a higher-degree model, freezing the well-estimated parts to stabilize training.
*   **Encoding prior knowledge:** If theoretical or physical constraints dictate the value at certain parameter combinations, you can pin those points to ensure the model respects them.

What is Bézier simplex splitting (subdivision)?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Bézier simplex **splitting** (or **subdivision**) is a technique for refining a fitted Bézier simplex model by recursively dividing the parameter domain (the simplex) into smaller sub-simplices.

**How it works:**

1.  A Bézier simplex maps a standard parameter simplex to a family of objective values.
2.  Splitting subdivides the parameter simplex by choosing an edge and inserting a new vertex on that edge (by default at the midpoint, ``s=0.5``), which forms smaller sub-simplices.
3.  Each resulting sub-simplex inherits control points from the parent in a way that preserves continuity and smoothness across the subdivision.
4.  Repeating this process creates a hierarchical decomposition, where different regions can be approximated with different levels of detail.

**Why use splitting?**

*   **Local refinement:** Focus computational effort on regions of interest without uniformly increasing the global degree.
*   **Adaptive complexity:** Automatically increase model complexity only where the surface is complex or "wiggly."
*   **Reduced overfitting:** By keeping low degree in simple regions and increasing it only where needed, you avoid unnecessary parameters.
*   **Efficient multi-resolution analysis:** Create a coarse-to-fine hierarchy useful for progressive fitting or visualization.

**Practical usage:**

Splitting is particularly valuable when:

*   The Pareto front has **varying curvature**—smooth in some regions, highly curved in others.
*   Computational budget is limited and you want to allocate samples intelligently.
*   You need a **hierarchical representation** for visualization or downstream analysis.
*   High-degree fits suffer from oscillations or overfitting, but low-degree models miss important features.

See the :doc:`advanced` documentation for implementation details and examples.


Troubleshooting
---------------

Are approximation results always reliable?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not necessarily. While the Universal Approximation Theorem guarantees that a Bézier simplex *can* approximate any continuous map, a model with a **fixed** degree might not be sufficient for highly complex or "wiggly" surfaces.

To ensure reliability:
1.  **Check Residuals:** Large fitting errors indicate the degree might be too low.
2.  **Cross-Validation:** Use the built-in k-fold tools to ensure the model generalizes well to unseen data.
3.  **Visualization:** If the dimension allows, plot the resulting surface against the training points.
4.  **Domain Knowledge:** Verify that the predicted trade-offs make sense according to the physics or logic of your problem.

How can I improve fitting convergence or accuracy?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you encounter poor accuracy or optimization issues, try these steps:

1.  **Check Data Quality:** Ensure ``params`` sum to 1 and that you have enough samples relative to the degree.
2.  **Adjust the Degree:** If the surface is too complex, increase the degree. If the model is oscillating or overfitting, decrease it.
3.  **Better Initialization:** Use the ``init`` argument to provide a better starting point, perhaps from a coarse fit or domain knowledge.
4.  **Increase Training Epochs:** For complex surfaces, the L-BFGS optimizer might need more iterations. In the CLI, use ``--max_epochs``.
5.  **Data Normalization:** Scale your output values (``values``) to a similar range (e.g., using ``--normalize std``) to help the optimizer converge faster.
