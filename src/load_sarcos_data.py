"""
Load and explore SARCOS inverse dynamics dataset
"""
import scipy.io
import numpy as np

# Load training data
print("Loading SARCOS training data...")
train_data = scipy.io.loadmat('Datasets/sarcos_inv.mat')

# Load test data
print("Loading SARCOS test data...")
test_data = scipy.io.loadmat('Datasets/sarcos_inv_test.mat')

# Explore the structure
print("\n" + "="*60)
print("TRAINING DATA STRUCTURE")
print("="*60)
for key in train_data.keys():
    if not key.startswith('__'):
        print(f"\nKey: {key}")
        print(f"  Type: {type(train_data[key])}")
        if hasattr(train_data[key], 'shape'):
            print(f"  Shape: {train_data[key].shape}")
            print(f"  Data type: {train_data[key].dtype}")

print("\n" + "="*60)
print("TEST DATA STRUCTURE")
print("="*60)
for key in test_data.keys():
    if not key.startswith('__'):
        print(f"\nKey: {key}")
        print(f"  Type: {type(test_data[key])}")
        if hasattr(test_data[key], 'shape'):
            print(f"  Shape: {test_data[key].shape}")
            print(f"  Data type: {test_data[key].dtype}")

# Extract actual data arrays
# SARCOS dataset typically has inputs and outputs combined
# Usually the first 21 columns are inputs (7 joint positions, 7 velocities, 7 accelerations)
# and the last 7 columns are outputs (7 joint torques)

# Find the main data array
train_key = [k for k in train_data.keys() if not k.startswith('__')][0]
test_key = [k for k in test_data.keys() if not k.startswith('__')][0]

train_array = train_data[train_key]
test_array = test_data[test_key]

print("\n" + "="*60)
print("DATASET ANALYSIS")
print("="*60)
print(f"\nTraining set shape: {train_array.shape}")
print(f"Test set shape: {test_array.shape}")

# Assuming standard SARCOS format: 21 inputs + 7 outputs = 28 columns
if train_array.shape[1] == 28:
    X_train = train_array[:, :21]  # First 21 columns are inputs
    y_train = train_array[:, 21:]  # Last 7 columns are outputs
    
    X_test = test_array[:, :21]
    y_test = test_array[:, 21:]
    
    print(f"\nInput dimensions: {X_train.shape[1]}")
    print(f"Output dimensions: {y_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    print(f"\nInput statistics (training):")
    print(f"  Mean: {X_train.mean(axis=0)[:5]}... (showing first 5)")
    print(f"  Std: {X_train.std(axis=0)[:5]}... (showing first 5)")
    print(f"  Min: {X_train.min(axis=0)[:5]}... (showing first 5)")
    print(f"  Max: {X_train.max(axis=0)[:5]}... (showing first 5)")
    
    print(f"\nOutput statistics (training):")
    print(f"  Mean: {y_train.mean(axis=0)}")
    print(f"  Std: {y_train.std(axis=0)}")
    print(f"  Min: {y_train.min(axis=0)}")
    print(f"  Max: {y_train.max(axis=0)}")
else:
    print(f"\nUnexpected data format. Expected 28 columns, got {train_array.shape[1]}")
    print("Please inspect the data manually.")

print("\n" + "="*60)
print("Data loading complete!")
print("="*60)
