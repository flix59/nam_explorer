import torch
from .model import NAM

def train_nam(X, y, num_features, hidden_dim=32, depth=5, epochs=1000, lr=0.01, verbose=True):
    """
    Train a Neural Additive Model.

    Args:
        X (torch.Tensor): Input features of shape (n_samples, n_features)
        y (torch.Tensor): Target values of shape (n_samples,)
        num_features (int): Number of input features
        hidden_dim (int): Hidden dimension for shape functions
        depth (int): Depth of shape function networks
        epochs (int): Number of training epochs
        lr (float): Learning rate
        verbose (bool): Print training progress

    Returns:
        NAM: Trained model
    """
    model = NAM(num_features=num_features, hidden_dim=hidden_dim, depth=depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = torch.nn.functional.mse_loss(outputs, y)
        loss.backward()
        optimizer.step()

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return model

if __name__ == "__main__":
    # Example usage
    test_data = torch.randn(100, 5)
    test_targets = test_data[:, 1] * 25
    model = train_nam(test_data, test_targets, num_features=5, hidden_dim=10, depth=3)