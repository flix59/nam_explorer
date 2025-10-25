from .model import NAM, ShapeFunction
from .train import train_nam
from .visualization import (
    plot_shape_functions,
    get_shape_function_values,
)

__version__ = "0.1.0"
__all__ = [
    "NAM",
    "ShapeFunction",
    "train_nam",
    "plot_shape_functions",
    "get_shape_function_values"
]