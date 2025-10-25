from pathlib import Path
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import pandas as pd
from experiments.housing.dataset import HousingDataset
from nam import NAM


def plot_shape_functions(model: NAM, dataset: HousingDataset ):
    import matplotlib.pyplot as plt
    import numpy as np
    test_data = dataset.test_data.numpy()
    x_values = []
    y_values = []

    for feature_index, feature_name in enumerate(dataset.features):
        shape_function = model.shape_functions[feature_index]
        x_tensor = torch.tensor(test_data[:, feature_index], dtype=torch.float32)
        y = shape_function(x_tensor.view(-1,1)).detach().view(-1).numpy()
        x_values.append(x_tensor.numpy())
        y_values.append(y)
    xs = np.stack(x_values, axis=-1)
    unscaled_xs = dataset.scaler.inverse_transform(xs)
    ys = np.stack(y_values, axis=-1)

    for feature_index, feature_name in enumerate(dataset.features):
        x = unscaled_xs[:, feature_index]
        y = ys[:, feature_index]
        plt.figure(figsize=(10, 5))
        plt.scatter(x, y, label=f'Shape Function {feature_index}', s=10)
        plt.title(f'Shape Function of {feature_name}')
        plt.xlabel(feature_name)
        plt.ylabel('Output')
        plt.legend()
        plt.grid()
        plt.savefig(f'shape_function_{feature_name}.png')
        plt.close()

def train_nam(path: Path,hidden_dim=32, depth=5, epochs=100):
    dataset = HousingDataset(csv_file=path)
    model_name = path.name
    model_name = f"{model_name}_{hidden_dim}_{depth}"
    model = NAM(num_features=len(dataset.features), hidden_dim=hidden_dim, depth=depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(dataset.train_data)
        loss = torch.nn.functional.mse_loss(outputs, dataset.train_targets)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    save_dir = path.parent / f"models/"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir/"{model_name}.pth"
    model.save_model(save_path)
    return model, save_path


def load_and_evaluate_model(model_name: str, dataset_path: Path):
    dataset = HousingDataset(csv_file=dataset_path)
    model = NAM.load_model(pwd / f"models/{model_name}.pth")
    model.eval()
    with torch.no_grad():
        outputs = model(dataset.test_data)
        loss = torch.nn.functional.mse_loss(outputs, dataset.test_targets)
        print(f"Test Loss: {loss.item()}")
    # plot shape functions
    plot_shape_functions(model, dataset)

if __name__ == "__main__":
    pwd = Path(__file__).parent
    model, model_path = train_nam(pwd / 'data/housing.csv')
    load_and_evaluate_model(model_path, pwd / 'data/housing.csv')