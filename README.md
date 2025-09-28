# nam_explorer
Nam Explorer is a Python package for exploring and visualizing Network Animator (NAM) files, which are used to represent network simulations.
## Features

- Parse and analyze NAM (Network Animator) files.
- Visualize network topologies and simulation events.
- Extract node, link, and event information for further analysis.
- Command-line interface for quick exploration.
- Extensible Python API for custom workflows.

## Installation

```bash
pip install nam_explorer
```

## Usage

### Command Line

```bash
nam_explorer visualize path/to/your.nam
```

### Python API

```python
from nam_explorer import NamParser

parser = NamParser('path/to/your.nam')
network = parser.parse()
network.visualize()
```

## Example

```python
from nam_explorer import NamParser

parser = NamParser('example.nam')
network = parser.parse()

print(f"Nodes: {network.nodes}")
print(f"Links: {network.links}")
network.visualize()
```

## Documentation

See the [documentation](docs/) for detailed API usage and examples.

## Contributing

Contributions are welcome! Please open issues or pull requests.

## License

This project is licensed under the MIT License.