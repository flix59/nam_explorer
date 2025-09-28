
# Plot shape functions
import numpy as np
import torch
from experiments.housing.dataset import HousingDataset
from src.nam import NAM


def get_shape_function_values(model: NAM, dataset: HousingDataset):
        values = {}
        for index, feature in enumerate(dataset.features):
    
            # Range of values in original space
            x_raw = dataset.data[feature].values
            x_range = np.linspace(x_raw.min(), x_raw.max(), 200)
    
            # Scale to model input space
            x_scaled = dataset.scaler.transform(np.array([x_range if i == index else np.zeros_like(x_range)
                                                  for i in range(len(dataset.features))]).T)

            x_tensor = torch.tensor(x_scaled[:, index].reshape(-1, 1), dtype=torch.float32)
            with torch.no_grad():
                shape_fn = model.shape_functions[index]
                y = shape_fn(x_tensor).view(-1).numpy()
            values[feature] = (x_range, y)
        
        return values