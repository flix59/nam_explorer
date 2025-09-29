# NAM Explorer

Neural Additive Models (NAM) library for interpretable machine learning with shape function visualization.

## Features

- **Interpretable ML**: Build Neural Additive Models that decompose predictions into individual feature contributions
- **Shape Functions**: Visualize how each feature affects the model's predictions
- **PyTorch-based**: Efficient training with automatic differentiation
- **Model Persistence**: Save and load trained models
- **Extensible API**: Easy to integrate into your ML workflows

## Installation

```bash
pip install nam-explorer
```

For visualization features (Gradio, Plotly):
```bash
pip install nam-explorer[viz]
```

For development:
```bash
pip install nam-explorer[dev]
```

## Quick Start

### Training a NAM

```python
import torch
from nam_explorer import NAM, train_nam

# Prepare your data
X = torch.randn(1000, 5)  # 1000 samples, 5 features
y = X[:, 0] * 2 + X[:, 1] ** 2 - X[:, 2]  # Some non-linear relationship

# Train the model
model = train_nam(X, y, num_features=5, hidden_dim=32, depth=5, epochs=1000)

# Make predictions
predictions = model(X)
```

### Using the Model

```python
from nam_explorer import NAM

# Create model
model = NAM(num_features=5, hidden_dim=32, depth=5)

# Forward pass
output = model(X)

# Save model
model.save_model('my_nam_model.pth')

# Load model
loaded_model = NAM.load_model('my_nam_model.pth')
```

### Visualizing Shape Functions

Each feature's contribution can be visualized by evaluating the corresponding shape function across its range:

```python
import torch
import matplotlib.pyplot as plt

# Get shape function for feature 0
feature_idx = 0
x_range = torch.linspace(-3, 3, 100).reshape(-1, 1)
contributions = model.shape_functions[feature_idx](x_range)

plt.plot(x_range.numpy(), contributions.detach().numpy())
plt.xlabel('Feature Value')
plt.ylabel('Contribution to Prediction')
plt.title(f'Shape Function for Feature {feature_idx}')
plt.show()
```

## What are Neural Additive Models?

Neural Additive Models (NAMs) are interpretable machine learning models that learn a separate neural network for each input feature. The final prediction is the sum of all feature contributions:

```
y = f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ)
```

This additive structure makes it easy to understand how each feature affects predictions, while still capturing non-linear relationships.

## Examples

See the `experiments/` directory for complete examples including:
- Housing price prediction with interactive Gradio visualization
- Shape function plotting
- Model training and evaluation

## API Reference

### `NAM`

Main model class implementing a Neural Additive Model.

**Parameters:**
- `num_features` (int): Number of input features
- `hidden_dim` (int): Hidden dimension for shape function networks
- `depth` (int): Depth of each shape function network

**Methods:**
- `forward(x)`: Forward pass returning predictions
- `save_model(path)`: Save model to disk
- `load_model(path)`: Load model from disk (classmethod)

### `train_nam`

Train a NAM model on your data.

**Parameters:**
- `X` (torch.Tensor): Input features (n_samples, n_features)
- `y` (torch.Tensor): Target values (n_samples,)
- `num_features` (int): Number of features
- `hidden_dim` (int): Hidden dimension (default: 32)
- `depth` (int): Network depth (default: 5)
- `epochs` (int): Training epochs (default: 1000)
- `lr` (float): Learning rate (default: 0.01)
- `verbose` (bool): Print progress (default: True)

**Returns:**
- Trained NAM model

## Contributing

Contributions are welcome! Please open issues or pull requests at the [GitHub repository](https://github.com/yourusername/nam_explorer).

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{nam_explorer,
  title = {NAM Explorer: Neural Additive Models for Interpretable Machine Learning},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/nam_explorer}
}
```