# FlyVis Tutorial

**Connectome-constrained models of the fruit fly visual system**

Tutorial material for working with [FlyVis](https://turagalab.github.io/flyvis/) models at the [Winter School on Computational Approaches in Biological Sciences (SJCABS)](https://sjcabs.com/).

## Overview

| Tutorial | Topic | Description |
|----------|-------|-------------|
| 0 | **Using the model** | Load a pretrained model, stimulate it, plot neural responses, and update parameters with gradient descent |
| 1 | **Mechanism discovery** | Apply UMAP dimensionality reduction and Gaussian mixture clustering to discover computational strategies |
| 2 | **Deep stimulus design** | Find optimal naturalistic stimuli and generate artificial optimal stimuli using gradient-based optimization |

## Installation

### Google Colab Enterprise

Run this in the first cell of your notebook:

```bash
!pip install "git+https://github.com/sjcabs/flyvis_tutorial.git"
!flyvis download-pretrained
```

### Local Installation

Requires Python 3.11.

```bash
# Clone the repository
git clone https://github.com/sjcabs/flyvis_tutorial.git
cd flyvis_tutorial

# Create a conda environment (recommended)
conda create -n flyvis_tutorial python=3.11
conda activate flyvis_tutorial

# Install the package
pip install -e .

# Download pretrained models
flyvis download-pretrained
```

## Acknowledgments

This tutorial was adapted from the [FlyVis visual system tutorial](https://github.com/TuragaLab/flysim_tutorials/tree/main/visual_system_tutorial) by Janne Lappalainen.

## References

Lappalainen, J. K. et al. Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634, 1132–1140 (2024). https://doi.org/10.1038/s41586-024-07939-3

FlyVis Documentation: https://turagalab.github.io/flyvis/
