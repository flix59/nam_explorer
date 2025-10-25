import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import matplotlib.patheffects as patheffects
# Deprecated: shape function utilities moved to nam.visualization
# This file is kept for backward compatibility with old experiments

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
