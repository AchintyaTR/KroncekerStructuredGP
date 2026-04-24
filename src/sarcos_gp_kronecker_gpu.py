"""
Optimized Kronecker-Structured GP Regression on SARCOS Dataset
TRUE KRONECKER ACCELERATION - GPU Version (PyTorch)
"""

import torch
import numpy as np
import scipy.io
import sys
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class OptimizedKroneckerGPGPU:
    """
    Optimized GP with TRUE Kronecker acceleration, running on the GPU using PyTorch.
    """
    
    def __init__(self, m=5, sigma_f=1.0, sigma_n=0.1, length_scale=1.0, device='cuda'):
        self.m = m
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.length_scale = length_scale
        self.device = device
        
        self.D = None
        self.Z = []
        self.Q_factors = []
        self.lambda_factors = []
        self.alpha = None
        self.memory_info = {}
        
    def _rbf_1d(self, x1, x2):
        """1D RBF kernel (PyTorch)"""
        x1 = x1.view(-1, 1)
        x2 = x2.view(-1, 1)
        dist = torch.cdist(x1, x2, p=2.0)**2
        return torch.exp(-dist / (2.0 * self.length_scale**2))
    
    def _compute_Kuf(self, X):
        """
        Compute K_uf using product kernel structure.
        """
        n = X.shape[0]
        M = self.m ** self.D
        
        K_factors = []
        for d in range(self.D):
            K_d = self._rbf_1d(self.Z[d], X[:, d])  # (m, n)
            K_factors.append(K_d)
        
        K_uf = torch.ones((M, n), dtype=torch.float64, device=self.device) * (self.sigma_f**2)
        
        for d in range(self.D):
            n_repeat = self.m ** d
            n_tile = self.m ** (self.D - d - 1)
            K_d_exp = torch.tile(K_factors[d], (n_tile, 1))
            K_d_exp = torch.repeat_interleave(K_d_exp, n_repeat, dim=0)
            K_uf *= K_d_exp
        
        return K_uf
    
    def _compute_Kuu_eigen(self):
        """
        Compute eigendecomposition of K_uu factors on GPU.
        """
        Q_factors = []
        lambda_factors = []
        
        for d in range(self.D):
            K_d = (self.sigma_f**(2.0/self.D)) * self._rbf_1d(self.Z[d], self.Z[d])
            K_d += 1e-6 * torch.eye(self.m, dtype=torch.float64, device=self.device)
            
            lambdas, Q = torch.linalg.eigh(K_d)
            lambda_factors.append(lambdas)
            Q_factors.append(Q)
        
        return Q_factors, lambda_factors
    
    def _kronecker_eigenvalues(self):
        """
        Compute Kronecker product of eigenvalues.
        """
        kron_lambda = self.lambda_factors[0]
        for d in range(1, self.D):
            kron_lambda = torch.kron(kron_lambda, self.lambda_factors[d])
        return kron_lambda
    
    def _kron_mv(self, X, transpose=False):
        """
        Efficiently compute Q @ X or Q^T @ X without full Kronecker tracking in PyTorch
        """
        M = self.m ** self.D
        is_vector = (X.dim() == 1)
        if is_vector:
            X = X.view(-1, 1)
            
        n_cols = X.shape[1]
        result = X.contiguous()
        
        shape = [self.m] * self.D + [n_cols]
        result = result.view(shape)
        
        for d in range(self.D):
            Q_d = self.Q_factors[d]
            if transpose:
                Q_d = Q_d.T
            
            result = torch.movedim(result, d, 0)
            original_shape = result.shape
            result = result.reshape(self.m, -1)
            result = Q_d @ result
            result = result.reshape(original_shape)
            result = torch.movedim(result, 0, d)
        
        result = result.reshape(M, n_cols)
        
        if is_vector:
            return result.flatten()
        return result
    
    def fit(self, X, y):
        n, self.D = X.shape
        M = self.m ** self.D
        y = y.flatten()
        
        print(f"Fitting GPU Kronecker GP on {self.device}")
        print(f"  Samples: {n}, Dimensions: {self.D}")
        print(f"  Inducing/dim: {self.m}, Total: {M:,}")
        sys.stdout.flush()
        t0 = time.time()
        
        self.Z = []
        for d in range(self.D):
            q = torch.linspace(0.05, 0.95, self.m, dtype=torch.float64, device=self.device)
            self.Z.append(torch.quantile(X[:, d], q))
        
        print("Computing Kronecker eigendecomposition on GPU...")
        t_eigen = time.time()
        self.Q_factors, self.lambda_factors = self._compute_Kuu_eigen()
        if self.device.type == 'cuda': torch.cuda.synchronize()
        print(f"  Eigendecomposition done in {time.time() - t_eigen:.2f}s")
        
        print("Computing K_uf on GPU...")
        K_uf = self._compute_Kuf(X)
        
        kron_lambda = self._kronecker_eigenvalues()
        
        print("Solving sparse GP system using eigenvalues...")
        print(f"  Transforming K_uf ({M} x {n})...")
        K_uf_tilde = self._kron_mv(K_uf, transpose=True)
        
        Sigma_tilde = torch.diag(kron_lambda) + K_uf_tilde @ K_uf_tilde.T / (self.sigma_n**2)
        Sigma_tilde += 1e-6 * torch.eye(M, dtype=torch.float64, device=self.device)
        
        L_tilde = torch.linalg.cholesky(Sigma_tilde)
        
        b_tilde = K_uf_tilde @ y / (self.sigma_n**2)
        
        # Solving the linear system L_tilde * L_tilde.T * alpha_tilde = b_tilde
        alpha_tilde = torch.cholesky_solve(b_tilde.unsqueeze(1), L_tilde).squeeze(1)
        
        self.alpha = self._kron_mv(alpha_tilde, transpose=False)
        self.L_tilde = L_tilde
        self.kron_lambda = kron_lambda
        
        if self.device.type == 'cuda': torch.cuda.synchronize()
        print(f"Fitting done in {time.time() - t0:.2f}s")
        return self
    
    def predict(self, X_test, return_std=False):
        n_test = X_test.shape[0]
        print(f"Predicting on {n_test} samples on GPU...")
        t0 = time.time()
        
        K_sf = self._compute_Kuf(X_test)
        y_pred = K_sf.T @ self.alpha
        
        if return_std:
            print("Computing uncertainties on GPU...")
            k_star = self.sigma_f**2
            
            K_sf_tilde = self._kron_mv(K_sf, transpose=True)
            
            v1 = torch.linalg.solve_triangular(self.L_tilde, K_sf_tilde, upper=False)
            
            lambda_inv_sqrt = 1.0 / torch.sqrt(self.kron_lambda + 1e-8)
            v2 = lambda_inv_sqrt.unsqueeze(1) * K_sf_tilde
            
            y_var = k_star - torch.sum(v2**2, dim=0) + torch.sum(v1**2, dim=0) + self.sigma_n**2
            y_std = torch.sqrt(torch.maximum(y_var, torch.tensor(1e-8, device=self.device)))
            
            if self.device.type == 'cuda': torch.cuda.synchronize()
            print(f"Prediction done in {time.time() - t0:.2f}s")
            return y_pred, y_std
            
        if self.device.type == 'cuda': torch.cuda.synchronize()
        print(f"Prediction done in {time.time() - t0:.2f}s")
        return y_pred


def main():
    print("\n" + "="*60)
    print("GPU ACCELERATED KRONECKER GP ON SARCOS")
    print("="*60 + "\n")
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_data = scipy.io.loadmat('Datasets/sarcos_inv.mat')
    test_data = scipy.io.loadmat('Datasets/sarcos_inv_test.mat')
    
    train_key = [k for k in train_data.keys() if not k.startswith('__')][0]
    test_key = [k for k in test_data.keys() if not k.startswith('__')][0]
    
    X_train_full = train_data[train_key][:, :21]
    y_train_full = train_data[train_key][:, 21]
    X_test_full = test_data[test_key][:, :21]
    y_test_full = test_data[test_key][:, 21]
    
    # Normalize
    scaler_X = StandardScaler()
    X_train_s = scaler_X.fit_transform(X_train_full)
    X_test_s = scaler_X.transform(X_test_full)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train_full.reshape(-1, 1)).ravel()
    
    # Convert exactly to correct PyTorch Tensors on device
    X_train_t = torch.tensor(X_train_s, dtype=torch.float64, device=device)
    y_train_t = torch.tensor(y_train_s, dtype=torch.float64, device=device)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float64, device=device)
    y_test_orig = y_test_full  # Keep a numpy copy for metrics
    
    # Configuration
    D = 5
    m = 6
    feature_indices = [0, 1, 2, 7, 8]
    
    X_train = X_train_t[:, feature_indices]
    y_train = y_train_t
    X_test = X_test_t[:, feature_indices]
    
    n_train = len(X_train)
    n_test = len(X_test)
    
    print(f"\nUsing: {n_train} train, {n_test} test, D={D}, m={m}")
    sys.stdout.flush()
    
    # Train
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    gp = OptimizedKroneckerGPGPU(m=m, sigma_f=1.0, sigma_n=0.08, length_scale=1.2, device=device)
    gp.fit(X_train, y_train)
    
    # Predict
    print("\n" + "="*60)
    print("PREDICTION")
    print("="*60)
    
    y_pred_s_t, y_std_s_t = gp.predict(X_test, return_std=True)
    
    # Offload back to CPU & numpy
    y_pred_s = y_pred_s_t.cpu().numpy()
    y_std_s = y_std_s_t.cpu().numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_std = y_std_s * scaler_y.scale_[0]
    
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
    
    # Memory metrics from torch
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"\nPeak GPU Memory Allocated: {max_mem:.2f} MB")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == "__main__":
    main()
