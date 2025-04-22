"""
Quantum Preprocessing Script for NSL-KDD Dataset

For amplitude embedding with:
- MinMax scaling
- Unit L2 normalization
- Power-of-2 padding
- No one-hot encoding

Outputs clean CSVs and visualizations ready for quantum classifier training with adversarial training support.

Author: Muhammad Haffi Khalid 
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
TRAIN_ARFF = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTrain+_20Percent.arff"
TEST_ARFF  = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTest+.arff"

OUTPUT_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\quantum"
VISUAL_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\visuals_quantum"

TRAIN_OUT = os.path.join(OUTPUT_DIR, "quantum_train_preprocessed.csv")
TEST_OUT  = os.path.join(OUTPUT_DIR, "quantum_test_preprocessed.csv")

# === UTILITIES ===
def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def read_arff_as_dataframe(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    header_lines = [line.strip() for line in lines if line.lower().startswith('@attribute')]
    col_names = [line.split()[1].strip("'") for line in header_lines]  # strip quotes
    data_start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data")
    data_lines = [line.strip() for line in lines[data_start + 1:] if line.strip()]
    data = [line.split(',') for line in data_lines]
    df = pd.DataFrame(data, columns=col_names)
    return df

def next_power_of_two(n):
    return 1 if n == 0 else 2**int(np.ceil(np.log2(n)))

def pad_to_power_of_two(X):
    current_dim = X.shape[1]
    target_dim = next_power_of_two(current_dim)
    if current_dim == target_dim:
        return X
    padded = np.zeros((X.shape[0], target_dim))
    padded[:, :current_dim] = X
    print(f"[INFO] Padded from {current_dim} to {target_dim} features.")
    return padded

def visualize_norms(norms, name):
    plt.figure(figsize=(7, 4))
    sns.histplot(norms, kde=True)
    plt.title(f'L2 Norm Distribution - {name}')
    plt.xlabel('L2 Norm')
    plt.ylabel('Density')
    path = os.path.join(VISUAL_DIR, f'{name}_norms.png')
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved norm plot: {path}")

def visualize_labels(labels, name):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels)
    plt.title(f'Label Distribution - {name}')
    plt.xlabel('Class (0=Normal, 1=Attack)')
    plt.ylabel('Count')
    path = os.path.join(VISUAL_DIR, f'{name}_label_distribution.png')
    plt.savefig(path)
    plt.close()
    print(f"[INFO] Saved label plot: {path}")

# === MAIN ===
def preprocess_and_save(path, out_path, name):
    print(f"[INFO] Processing: {name}")
    df = read_arff_as_dataframe(path)
    print(f"[DEBUG] Columns in DataFrame: {df.columns.tolist()}")


    drop_cols = ['protocol_type', 'service', 'flag']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    df['class'] = df['class'].apply(lambda x: 0.0 if 'normal' in str(x).lower() else 1.0)
    labels = df['class'].astype(np.float32).values

    feature_cols = [col for col in df.columns if col != 'class']
    features = df[feature_cols].astype(np.float32).values

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    norms = norm(features, axis=1)
    visualize_norms(norms, name)
    visualize_labels(labels, name)

    features = features / norms[:, np.newaxis]
    features = pad_to_power_of_two(features)

    final_df = pd.DataFrame(features)
    final_df['class'] = labels
    final_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Saved to {out_path}\n")

if __name__ == "__main__":
    ensure_dirs(OUTPUT_DIR, VISUAL_DIR)

    preprocess_and_save(TRAIN_ARFF, TRAIN_OUT, "train")
    preprocess_and_save(TEST_ARFF, TEST_OUT, "test")
