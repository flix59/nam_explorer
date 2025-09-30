"""Visualization utilities for Neural Additive Models."""

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib import patheffects

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None


def get_shape_function_values(model, X, feature_names=None, scaler=None, num_points=200):
    """
    Extract shape function values across the range of input features.

    Args:
        model: Trained NAM model
        X: Input data (numpy array or torch tensor) of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        scaler: Optional sklearn StandardScaler used to transform the data
        num_points: Number of points to evaluate each shape function

    Returns:
        dict: Mapping from feature name/index to (x_range, y_values) tuples
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()

    num_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    values = {}
    for index, feature in enumerate(feature_names):
        # Range of values in original space
        x_raw = X[:, index]
        x_range = np.linspace(x_raw.min(), x_raw.max(), num_points)

        # Scale to model input space if scaler provided
        if scaler is not None:
            # Create dummy data with only this feature varying
            x_scaled = scaler.transform(
                np.array([x_range if i == index else np.zeros_like(x_range)
                         for i in range(num_features)]).T
            )
            x_tensor = torch.tensor(x_scaled[:, index].reshape(-1, 1), dtype=torch.float32)
        else:
            x_tensor = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float32)

        with torch.no_grad():
            shape_fn = model.shape_functions[index]
            y = shape_fn(x_tensor).view(-1).numpy()

        values[feature] = (x_range, y)

    return values


def plot_shape_functions(model, X, feature_names=None, scaler=None, figsize=(10, 5), save_dir=None):
    """
    Plot all shape functions for a trained NAM model.

    Args:
        model: Trained NAM model
        X: Input data (numpy array or torch tensor) of shape (n_samples, n_features)
        feature_names: Optional list of feature names
        scaler: Optional sklearn StandardScaler used to transform the data
        figsize: Figure size for each plot
        save_dir: Optional directory path to save plots

    Returns:
        dict: Mapping from feature name to matplotlib figure
    """
    if isinstance(X, torch.Tensor):
        X_np = X.numpy()
    else:
        X_np = X
        X = torch.tensor(X, dtype=torch.float32)

    num_features = X.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    figures = {}

    # Get shape function values
    values = get_shape_function_values(model, X_np, feature_names, scaler)

    for feature_name in feature_names:
        x_range, y = values[feature_name]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x_range, y, linewidth=2)
        ax.set_title(f'Shape Function: {feature_name}')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Contribution to Prediction')
        ax.grid(True, alpha=0.3)

        figures[feature_name] = fig

        if save_dir is not None:
            from pathlib import Path
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / f'shape_function_{feature_name}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    return figures


def make_nam_architecture_figure(
    feature_names,
    hidden_layers=(32, 32),
    task="regression",
    example_inputs=None,
    example_outputs=None,
    figsize=None
):
    """
    Draw a NAM architecture diagram showing the additive structure.

    Args:
        feature_names: List of feature names
        hidden_layers: Tuple of hidden layer sizes (not fully used, for documentation)
        task: "regression" or "classification"
        example_inputs: Optional list of example input values
        example_outputs: Optional list of contribution values for each feature
        figsize: Optional figure size tuple (width, height)

    Returns:
        matplotlib.figure.Figure: The architecture diagram
    """
    if figsize is None:
        figsize = (9, max(4, int(len(feature_names) * 0.6)))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    num_features = len(feature_names)

    # Layout coordinates
    x_input = 0.05
    x_mlp = 0.35
    x_sum = 0.72
    x_out = 0.88
    y_top = 0.9
    y_bottom = 0.1
    ys = np.linspace(y_top, y_bottom, num_features)

    # Styles
    box_style = dict(edgecolor="black", facecolor="white", lw=1.2)
    neuron_style = dict(edgecolor="black", facecolor="lightgray", lw=0.8)

    # Draw features and MLPs
    for i, (fname, y) in enumerate(zip(feature_names, ys)):
        # Feature box
        ax.text(x_input - 0.02, y, fname, ha="center", va="center", fontsize=10, weight="bold")

        # Draw first MLP in detail
        if i == 0:
            mlp_width = 0.2
            mlp_height = 0.14
            ax.add_patch(Rectangle((x_mlp, y - mlp_height/2), mlp_width, mlp_height, **box_style))
            ax.text(x_mlp + mlp_width/2, y + mlp_height/2 + 0.02, r"$g_{%d}(%s)$" % (i+1, fname), ha="center", va="bottom", fontsize=9)

            # neuron coords: 1 input -> 5 hidden -> 3 hidden -> 1 output
            layer_xs = np.linspace(x_mlp + 0.02, x_mlp + mlp_width - 0.02, 4)
            layer_sizes = [1, 5, 3, 1]
            neuron_radius = 0.008

            neuron_coords = []
            for lx, size in zip(layer_xs, layer_sizes):
                ys_layer = np.linspace(y - 0.04, y + 0.04, size)
                coords = [(lx, yy) for yy in ys_layer]
                neuron_coords.append(coords)
                for (nx, ny) in coords:
                    ax.add_patch(Circle((nx, ny), neuron_radius, **neuron_style))

            # Draw connections
            for coords_a, coords_b in zip(neuron_coords, neuron_coords[1:]):
                for (xa, ya) in coords_a:
                    for (xb, yb) in coords_b:
                        ax.plot([xa, xb], [ya, yb], color="gray", lw=0.5)

        else:
            # Simple MLP box
            ax.add_patch(Rectangle((x_mlp, y - 0.05), 0.2, 0.1, **box_style))
            ax.text(x_mlp + 0.1, y, rf"$g_{i+1}({fname})$", ha="center", va="center", fontsize=9)

        # Arrow: feature -> MLP
        ax.add_patch(FancyArrowPatch(
            (x_input + 0.12, y), (x_mlp, y),
            arrowstyle='-|>', mutation_scale=10, lw=1
        ))

        # Arrow: MLP -> SUM
        con_style = dict(arrowstyle='-|>', mutation_scale=10, lw=1)
        ax.add_patch(FancyArrowPatch(
            (x_mlp + 0.2, y), (x_sum, 0.5),
            **con_style
        ))

        # Show example values
        if example_inputs is not None:
            ax.add_patch(Rectangle((x_input + 0.1, y - 0.02), 0.05, 0.05, **box_style))
            ax.text(x_input + 0.14, y, f"{example_inputs[i]:.2f}", ha="right", va="center", fontsize=9,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])
        if example_outputs is not None:
            mid_x = (x_mlp + 0.2 + x_sum) / 2
            mid_y = (y + 0.5) / 2
            ax.text(mid_x, mid_y, f"{example_outputs[i]:+.2f}", ha="center", va="center", fontsize=9,
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])

    # Summation node
    ax.add_patch(Circle((x_sum, 0.5), 0.03, **box_style))
    ax.text(x_sum, 0.5, r"$\sum$", ha="center", va="center", fontsize=12)

    # Arrow: SUM -> Output
    ax.add_patch(FancyArrowPatch((x_sum + 0.03, 0.5), (x_out, 0.5),
                                 arrowstyle='-|>', mutation_scale=12, lw=1))

    if example_outputs is not None:
        total = sum(example_outputs)
        ax.text(x_out + 0.02, 0.5, f"{total:.2f}", ha="left", va="center", fontsize=11,
                path_effects=[patheffects.withStroke(linewidth=2, foreground="white")])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


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