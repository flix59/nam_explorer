# NAM explanation text for documentation/demos
NAM_EXPLANATION = """
## What is a Neural Additive Model (NAM)?

**Neural Additive Models** are neural versions of classic Generalized Additive Models (GAMs).
They learn a separate **shape function** \\(g_i(x_i)\\) for each input feature and combine them additively:

\\[
\\hat{y} = \\sum_{i=1}^{d} g_i(x_i) + b
\\]

Each \\(g_i\\) is a **tiny neural network** (usually an MLP) that maps a *single* feature to a contribution.
This keeps the model **interpretable**: you can visualize each feature's effect as a 1D curve.

### Why are NAMs useful?
- **Interpretability**: inspect a per-feature shape function instead of opaque weights.
- **Faithful partial effects**: curves are learned directly, not post-hoc approximations.
- **Flexible**: each \\(g_i\\) can be nonlinear and smooth.
- **Regularization options**: sparsity, smoothness, monotonicity constraints per feature.

### Great use cases
- **Tabular** regression/classification (healthcare risk, credit scoring, pricing, demand forecasting).
- **Policy/analytics** where explanations matter (what drives predictions, how much, and in which direction).
- **Feature auditing**: detect spurious patterns or saturations in single features.

### Less ideal when
- **Strong interactions** dominate (NAMs are additive by default).
  *Tip:* you can add a few **pairwise** terms \\(g_{ij}(x_i, x_j)\\) when needed.

### How classification works
For classification, NAMs typically model the **logit** (or log-odds) additively, then apply a final **link** (sigmoid/softmax).
"""