# Optimising a Vision Transformer for Weather Forecasting

This project is part of the **Machine Learning at Scale** coursework, focusing on optimizing a Deep Neural Network (DNN) for improved performance and reduced runtime. The model predicts future weather fields from historical weather data using a **Vision Transformer (ViT)** architecture implemented in PyTorch Lightning.

## Coursework Context

The primary goal was to:
- Profile and analyze bottlenecks in the model.
- Propose and implement optimizations.
- Measure improvements in both runtime and prediction accuracy.


## Model Overview

The core of the model is based on the **Vision Transformer** architecture as introduced in [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929). Key characteristics:
- Custom multi-head self-attention modules.
- Patch embedding for 2D weather field inputs.
- Image generation via learned positional encodings and transformer blocks.

The model reads weather fields from HDF5 files and trains to generate a time-advanced version of the field.

## Project Structure

```plaintext
.
├── config/
│   └── coursework_transformer.yaml       # YAML config file for training parameters
├── install.sh                            # Environment setup script for Cirrus
├── model/
│   ├── lit_transformer.py                # PyTorch Lightning wrapper for ViT
│   ├── transformer.py                    # Vision Transformer architecture
│   └── utils.py                          # DropPath, truncated normal init, etc.
├── README.md                             # You're here!
├── requirements.txt                      # Python package dependencies
├── run_coursework.sh                     # SLURM batch script for Cirrus
├── train.py                              # Main training and evaluation script
└── utils/
    ├── data_loader.py                    # ERA5 dataset loader (HDF5 format)
    ├── __init__.py
    ├── logging_utils.py                  # Rank-aware logging + version tracking
    ├── loss.py                           # Normalized L2 loss
    ├── metrics.py                        # Latitude-weighted RMSE & accuracy
    ├── parser.py                         # Argument parsing + parameter injection
    ├── plots.py                          # Plotting of predictions vs targets
    ├── synthesizer.py                    # Synthetic dataset generator (HDF5)
    └── y_params.py                       # YAML-based parameter management
```

## Running the Model

Run the training script with the following command:

```bash
python3 train.py --config short
```

The `short` configuration runs a lightweight training cycle (~10–20 minutes). For a full-scale training run:

```bash
python3 train.py --config base
```

## Cirrus Setup

Load the necessary modules and install dependencies as follows:

```bash
module load nvidia/cudnn/8.6.0-cuda-11.8
module load python/3.10.8-gpu
module load libsndfile/1.0.28

export PYTHONUSERBASE=/work/m24oc/m24oc/$USER/python-installs
python3 -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
python3 -m pip install protobuf==3.20
python3 -m pip install --upgrade tensorboard
python3 -m pip install h5py
```

**Disclaimer:** This codebase is for educational purposes under the University of Edinburgh's Machine Learning at Scale module.
