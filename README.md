# graphTORCH

graphTORCH is a PyTorch-based framework for applying Graph Neural Networks (GNNs) to Track-Hit association in the LHCb TORCH detector. It models the detector data as a bipartite graph where `track` and `hit` nodes are connected, and GNNs are used for edge classification to associate hits with their corresponding tracks.

## Features

- **Data Processing**: Converts ROOT files into PyTorch Geometric `HeteroData` objects.
- **Physics-Aware Preprocessing**: Includes Time-of-Flight (ToF) calculations, spatial scaling, and delta-feature construction.
- **Heterogeneous Graphs**: Models relationships between `track` and `hit` nodes using a bipartite graph structure.

## Project Structure

```text
graphTORCH/
├── data/               # Raw and processed dataset storage
├── models/             # Saved model checkpoints (.pth)
├── src/graphtorch/     # source code
│   ├── dataset.py      # ROOT to PyG dataset conversion
│   ├── model.py        # GNN model architecture
│   ├── train.py        # Training logic
│   └── evaluate.py     # Evaluation and basic plotting
├── main.py             # Simple entry point for training
└── pyproject.toml      # Project dependencies and script entry points
```

## Installation

This project uses `uv` for dependency management. To set up the environment:

```bash
uv sync
```

Alternatively, you can install the package in editable mode using pip:

```bash
pip install -e .
```

## Usage

The project provides three main command-line interfaces (CLIs) defined in `pyproject.toml`.

### 1. Data Preparation
Convert ROOT files into a processed PyTorch Geometric dataset:

```bash
graphtorch-dataset --root_dir ./data --input_file path/to/your/file.root --tree_name events
```

### 2. Training
Train the GNN model on the processed dataset:

```bash
graphtorch-train --input_file path/to/your/file.root --epochs 20 --batch_size 64
```
*Note: The model will be saved to `models/graphtorch_model.pth` by default.*

### 3. Evaluation
Evaluate a trained model and generate a ROC curve:

```bash
graphtorch-evaluate --input_file path/to/your/file.root --model_path models/graphtorch_model.pth
```
This script will output performance metrics and save a `roc_curve.png` plot.

## Dependencies

- **PyTorch & PyTorch Geometric**: Core GNN framework.
- **Uproot**: For reading ROOT files.
- **Awkward Array**: For handling ragged physics data.
- **Scikit-learn**: For evaluation metrics.
- **Matplotlib & Seaborn**: For visualization.
