from pathlib import Path
import torch
from torch.nn import Module, Sequential

class ShapeFunction(Module):
    def __init__(self, input_dim, hidden_dim, depth=2, dropout=0.0):
        """
        Initializes a Multi-Layer Perceptron (MLP) model.
        
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output layer.
            depth (int): Number of hidden layers.
            activation (str): Activation function to use ('relu', 'tanh', etc.).
            dropout (float): Dropout rate for regularization.
        """
        super(ShapeFunction, self).__init__()
        layers = []
        if depth < 2:
            raise ValueError("Depth must be at least 2")
        else:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            for _ in range(depth - 2):
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout))
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))

            layers.append(torch.nn.ReLU() )
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Linear(hidden_dim, 1))
        self.model = Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

class NAM(Module):
    def __init__(self, num_features: int, hidden_dim: int, depth=5):
        super(NAM, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.shape_functions = torch.nn.ModuleList([
            ShapeFunction(1, hidden_dim=hidden_dim, depth=depth) for _ in range(num_features)
        ])
        
    def forward(self, x):
        features = []
        for i in range(len(self.shape_functions)):
            features.append(self.shape_functions[i](x[:, i:i+1]))

        return torch.cat(features, dim=1).sum(dim=1)
    
    def save_model(self, path: Path):
        """Saves the model state dict to the specified path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.state_dict()
        state_dict["num_features"] = self.num_features
        state_dict["hidden_dim"] = self.hidden_dim
        state_dict["depth"] = self.depth
        torch.save(state_dict, path)
    
    @classmethod
    def load_model(cls, path):
        state_dict = torch.load(path)
        num_features = state_dict.pop("num_features")
        hidden_dim = state_dict.pop("hidden_dim")
        depth = state_dict.pop("depth")
        model = cls(num_features=num_features, hidden_dim=hidden_dim, depth=depth)
        model.load_state_dict(state_dict)
        return model
