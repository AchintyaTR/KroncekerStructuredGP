"""
Standard Brute-Force Gaussian Process Regression on SARCOS Dataset
Baseline for comparison with Kronecker-structured GP.

WARNING: This implementation has O(N^3) time and O(N^2) memory complexity.
Using the full dataset (44k samples) will likely cause OOM on standard machines.
We default to a subset of the data.
"""

import numpy as np
import scipy.io
import sys
import time
from scipy.spatial.distance import cdist
from scipy.linalg import cholesky, solve_triangular
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class BruteForceGP:
    """
    Standard Gaussian Process Regression.
    Complexity: O(N^3) time, O(N^2) memory.
    """
    
    def __init__(self, sigma_f=1.0, sigma_n=0.1, length_scale=1.0):
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.length_scale = length_scale
        self.X_train = None
        self.L = None
        self.alpha = None
        self.memory_usage = {}
        
    def _kernel(self, X1, X2):
        """Standard RBF Kernel: k(x, x') = σ_f² * exp(-||x-x'||² / 2l²)"""
        # Efficient distance computation
        dists = cdist(X1, X2, 'sqeuclidean')
        return self.sigma_f**2 * np.exp(-dists / (2 * self.length_scale**2))
        
    def fit(self, X, y):
        """Fit the GP model using Cholesky decomposition."""
        n = X.shape[0]
        self.X_train = X
        y = y.reshape(-1, 1)
        
        print(f"Fitting Brute Force GP on {n} samples...")
        sys.stdout.flush()
        t0 = time.time()
        
        # 1. Compute Kernel Matrix (O(N^2))
        print(f"  Computing Kernel Matrix ({n}x{n})...")
        sys.stdout.flush()
        K = self._kernel(X, X)
        
        # Memory check
        mem_K = K.nbytes / (1024**2)
        print(f"  Kernel matrix memory: {mem_K:.2f} MB")
        self.memory_usage['kernel_mb'] = mem_K
        
        # 2. Add noise to diagonal
        K[np.diag_indices_from(K)] += self.sigma_n**2
        
        # 3. Cholesky Decomposition (O(N^3))
        print("  Computing Cholesky Decomposition (in-place to save memory)...")
        sys.stdout.flush()
        try:
            # overwrite_a=True destroys K but saves memory (replaces K with L)
            # We don't need K again for prediction (we compute K_* separately)
            self.L = cholesky(K, lower=True, overwrite_a=True)
        except np.linalg.LinAlgError:
            print("  Cholesky failed! Matrix might not be positive definite. Adding jitter...")
            # If failed, K might be partially destroyed, so we might need to recompute or handle carefully.
            # However, for this script, we'll try to recover if possible, but real recovery implies recomputing K.
            # Since we can't easily undo overwrite_a, we'll raise for now or warn.
            raise RuntimeError("Cholesky failed with overwrite_a=True. Rerun with higher sigma_n.")
            
        # 4. Solve for alpha (O(N^2))
        # alpha = K^{-1} y = L^{-T} L^{-1} y
        print("  Solving for weights...")
        sys.stdout.flush()
        self.alpha = solve_triangular(self.L.T, solve_triangular(self.L, y, lower=True))
        
        t_fit = time.time() - t0
        print(f"Fitting done in {t_fit:.2f}s")
        self.memory_usage['total_mb'] = mem_K  # Dominant term
        
        return self
        
    def predict(self, X_test, return_std=False):
        """Predict on new data."""
        n_test = X_test.shape[0]
        print(f"Predicting on {n_test} samples...")
        sys.stdout.flush()
        t0 = time.time()
        
        # Compute K_star (covariance between train and test)
        K_star = self._kernel(self.X_train, X_test) # (N_train, N_test)
        
        # Mean prediction: mu = K_star^T @ alpha
        mu = K_star.T @ self.alpha
        
        if return_std:
            # Variance: k(x*, x*) - K_star^T @ K^{-1} @ K_star
            # K^{-1} = L^{-T} L^{-1}
            # v = L^{-1} @ K_star
            # var = k(x*, x*) - v^T @ v
            
            print("  Computing variance...")
            # Compute variance of test points (diagonal of kernel)
            # For RBF, k(x,x) = sigma_f^2
            k_star_diag = np.full(n_test, self.sigma_f**2)
            
            # Compute v = L^{-1} K_star
            v = solve_triangular(self.L, K_star, lower=True)
            
            # Predictive variance
            var = k_star_diag - np.sum(v**2, axis=0) + self.sigma_n**2 # Add noise variance for predictive distribution
            std = np.sqrt(np.maximum(var, 1e-10))
            
            print(f"Prediction done in {time.time() - t0:.2f}s")
            return mu.ravel(), std
            
        print(f"Prediction done in {time.time() - t0:.2f}s")
        return mu.ravel()

def main():
    print("\n" + "="*60)
    print("BRUTE FORCE GP ON SARCOS (BASELINE)")
    print("="*60 + "\n")
    
    # Load data
    try:
        train_data = scipy.io.loadmat('Datasets/sarcos_inv.mat')
        test_data = scipy.io.loadmat('Datasets/sarcos_inv_test.mat')
    except FileNotFoundError:
        print("Error: Dataset files not found in 'Datasets/' directory.")
        return

    train_key = [k for k in train_data.keys() if not k.startswith('__')][0]
    test_key = [k for k in test_data.keys() if not k.startswith('__')][0]
    
    X_train_full = train_data[train_key][:, :21]
    y_train_full = train_data[train_key][:, 21]
    X_test_full = test_data[test_key][:, :21]
    y_test_full = test_data[test_key][:, 21]
    
    # Normalize data
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train_full)
    X_test_s = scaler_X.transform(X_test_full)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train_full.reshape(-1, 1)).ravel()
    
    # -------------------------------------------------------------
    # CONFIGURATION
    # -------------------------------------------------------------
    # Use same feature selection as Kronecker for fair comparison
    # Kronecker code used: [0, 1, 2, 7, 8]
    feature_indices = [0, 1, 2, 7, 8] 
    
    # SUBSET SIZE
    # User requested full dataset run on 32GB RAM system.
    # We will use all training data.
    # N ~ 44,484
    # Memory for Kernel Matrix (float64): ~15 GB
    # We use overwrite_a=True in Cholesky to avoid doubling memory usage.
    
    print(f"Using FULL training set: {X_train_full.shape[0]} samples")
    print("WARNING: This will use ~15GB of RAM for the kernel matrix alone.")
    print("Ensure you have at least 32GB RAM available.")
    
    X_train = X_train_s[:, feature_indices]
    y_train = y_train_s
    
    # Use full test set but subset features
    X_test = X_test_s[:, feature_indices]
    y_test_orig = y_test_full
    
    n_test = len(X_test)
    
    print(f"Training Data: {X_train.shape}")
    print(f"Test Data:     {X_test.shape}")
    
    # Hyperparameters (same as Kronecker for rough comparison, though optimal might differ)
    # sigma_f=1.0, sigma_n=0.08, length_scale=1.2
    gp = BruteForceGP(sigma_f=1.0, sigma_n=0.08, length_scale=1.2)
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    # Hint to users about time
    print("Note: complexity is O(N^3). For N=44k, this may take hours.")
    gp.fit(X_train, y_train)
    
    # Predict
    print("\n" + "="*60)
    print("PREDICTION")
    print("="*60)
    y_pred_s, y_std_s = gp.predict(X_test, return_std=True)
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    
    # Metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    mse = mean_squared_error(y_test_orig, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_orig, y_pred)
    smse = mse / np.var(y_test_orig)
    
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    print(f"R²       : {r2:.4f}")
    print(f"SMSE     : {smse:.4f}")
    
    print("\nPerformance:")
    print(f"Kernel Memory: {gp.memory_usage.get('kernel_mb', 0):.2f} MB")
    
    # Plotting
    try:
        print("\nPlotting results...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pred vs True
        ax = axes[0]
        # Downsample for scatter plot if too many points
        plot_idx = np.random.choice(n_test, min(2000, n_test), replace=False)
        ax.scatter(y_test_orig[plot_idx], y_pred[plot_idx], alpha=0.5, s=10)
        lims = [min(y_test_orig.min(), y_pred.min()), max(y_test_orig.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', alpha=0.8)
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Brute Force GP (Full Data)\nR² = {r2:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Residuals
        ax = axes[1]
        residuals = y_test_orig - y_pred
        ax.hist(residuals, bins=50, edgecolor='k', alpha=0.7)
        ax.set_xlabel('Residual')
        ax.set_ylabel('Count')
        ax.set_title('Residual Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sarcos_gp_bruteforce_results.png', dpi=150)
        print("Saved plot to 'sarcos_gp_bruteforce_results.png'")
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)

if __name__ == "__main__":
    main()
