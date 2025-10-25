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
