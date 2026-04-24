# Kronecker Structured Gaussian Process Regression

This repository implements an **Optimized Kronecker-Structured Gaussian Process (GP) Regression** model, specifically applied to the SARCOS dataset. The implementation demonstrates significant acceleration in training and inference by leveraging the Kronecker structure of the covariance matrices, overcoming the traditional $\mathcal{O}(N^3)$ computational bottleneck of GPs.

## Features

- **True Kronecker Acceleration**: Implements sparse approximation with Kronecker structure.
- **Eigendecomposition Optimization**: Reduces time complexity from $\mathcal{O}(M^3)$ to $\mathcal{O}(Dm^3)$ and memory from $\mathcal{O}(M^2)$ to $\mathcal{O}(Dm^2)$.
- **GPU Acceleration Support**: Includes CUDA-accelerated versions via CuPy for massive speedups on compatible hardware.
- **Baseline Comparison**: Includes a standard brute-force GP implementation to benchmark performance, memory, and training time improvements.

## Repository Structure

- `src/`: Contains the core Python scripts for the GP models.
  - `sarcos_gp_kronecker.py`: CPU implementation of the Optimized Kronecker GP.
  - `sarcos_gp_kronecker_gpu.py`: GPU-accelerated implementation using CuPy.
  - `sarcos_gp_bruteforce.py`: Baseline standard GP model.
  - `load_sarcos_data.py`: Utilities for loading and preprocessing the SARCOS dataset.
- `notebooks/`: Jupyter Notebooks for exploration and testing.
  - `bruteforce0.ipynb`: Notebook demonstrating the baseline model.
- `docs/`: Presentations and documentation.
- `results/`: Performance plots, residual histograms, and uncertainty visualizations.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Scikit-Learn
- Matplotlib
- CuPy (Optional, for GPU acceleration)

## Usage

1. **Dataset Preparation**: 
   Ensure you have the SARCOS dataset (`sarcos_inv.mat` and `sarcos_inv_test.mat`) in a `Datasets/` or `data/` directory relative to the scripts.

2. **Run Optimized GP (CPU)**:
   ```bash
   python src/sarcos_gp_kronecker.py
   ```

3. **Run Optimized GP (GPU)**:
   ```bash
   python src/sarcos_gp_kronecker_gpu.py
   ```

4. **Run Baseline GP**:
   ```bash
   python src/sarcos_gp_bruteforce.py
   ```

## Results

By leveraging the Kronecker-structured eigendecomposition, our optimized implementation achieves comparable predictive performance to the exact GP while drastically reducing the memory footprint (by up to 80% compared to standard sparse or full GPs) and accelerating the training and prediction times. Check the `results/` folder for visualizations.
