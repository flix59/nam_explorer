# NAM - Neural Additive Models

A PyTorch library for building and visualizing Neural Additive Models (NAMs) - interpretable machine learning models that decompose predictions into individual feature contributions.

## Features

- **Interpretable ML**: Build Neural Additive Models that decompose predictions into individual feature contributions
- **Shape Function Visualization**: Plot and analyze how each feature affects predictions
- **Architecture Diagrams**: Generate visual explanations of the NAM structure
- **PyTorch-based**: Efficient training with automatic differentiation
- **Model Persistence**: Save and load trained models
- **Easy to Use**: Simple API for training and inference

## Installation

```bash
pip install nam
```

## Quick Start

### Training a NAM

```python
import torch
from nam import NAM, train_nam

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
from nam import NAM

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

Use the built-in visualization tools to understand feature contributions:

```python
from nam import plot_shape_functions, get_shape_function_values
import matplotlib.pyplot as plt

# Plot all shape functions
figures = plot_shape_functions(model, X, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])

# Or get the raw values for custom plotting
values = get_shape_function_values(model, X, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])
for feature_name, (x_range, y_values) in values.items():
    plt.figure()
    plt.plot(x_range, y_values)
    plt.title(f'Shape Function: {feature_name}')
    plt.show()
```

### Architecture Visualization

Generate publication-ready architecture diagrams:

```python
from nam import make_nam_architecture_figure

fig = make_nam_architecture_figure(
    feature_names=['feature1', 'feature2', 'feature3'],
    example_inputs=[0.5, -1.2, 0.8],
    example_outputs=[0.3, -0.5, 0.2]
)
fig.savefig('nam_architecture.png', dpi=300, bbox_inches='tight')
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

Contributions are welcome! Please open issues or pull requests at the [GitHub repository](https://github.com/flix59/nam_explorer).

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

## References

Based on the Neural Additive Models paper:
- Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R., & Hinton, G. E. (2021). Neural additive models: Interpretable machine learning with neural nets. *NeurIPS*.