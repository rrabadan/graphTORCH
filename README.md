# graphTORCH

graphTORCH is a PyTorch-based framework for applying Graph Neural Networks (GNNs) to hit association in the LHCb TORCH detector.

## Features

- **Data Processing**: Converts ROOT files into PyTorch Geometric `HeteroData`.
- **Physics-Aware Preprocessing**: Includes ToF (Time of Flight) calculations and spatial scaling.
- **Graphs**: Models relationships between `track` and `hit` nodes.

## Installation

This project uses `uv` for dependency management. To set up the environment:

```bash
uv sync
```

Or using pip:

```bash
pip install -e .
```

## Usage

### 1. Data Preparation
To process ROOT files into a format suitable for training:

```bash
graphtorch-dataset --root_dir ./data --input_file path/to/your/file.root --tree_name events
```

### 2. Training
Run the training script:

```bash
python main.py
```

### 3. Exploration
Check `inputs.ipynb` for data exploration and visualization.

## Dependencies

- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Uproot](https://uproot.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/)
