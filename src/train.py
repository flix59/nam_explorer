import torch
from src.nam import NAM

def train_nam():
    test_data = torch.randn(100, 5)  # Example test data
    test_targets = test_data[:, 1] * 25
    model = NAM(num_features=5, hidden_dim=10, depth=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(test_data)
        loss = torch.nn.functional.mse_loss(outputs, test_targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    train_nam()