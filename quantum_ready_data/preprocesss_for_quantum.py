"""
Preprocess CSV Files for Quantum Classifier using Amplitude Encoding
----------------------------------------------------------------------
This script loads the already preprocessed NSL-KDD training and test CSV files,
then converts each sample's features into a form ready for amplitude encoding.
Specifically, for each sample:
  1. The feature vector (all columns except the label) is normalized to unit norm.
  2. If the number of features is not a power of 2, it is padded with zeros until
     its length equals the next power of 2.
  
Additional visuals are generated along the way (e.g., distribution of the original L2 norm
and the final padded vector dimensions) for documentation purposes.

Output:
  - A quantum-ready training CSV file.
  - A quantum-ready test CSV file.

Author: [Your Name]
Date: [Date]
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Configuration: Input and Output Paths
# -------------------------------------------------------------------

# Input file paths (adjust if needed)
TRAIN_CSV_INPUT = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\nslkdd_train_preprocessed.csv"
TEST_CSV_INPUT  = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\nslkdd_test_preprocessed.csv"

# Output directory for quantum-ready CSVs
OUTPUT_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\quantum_ready_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file names
TRAIN_CSV_OUTPUT = os.path.join(OUTPUT_DIR, "quantum_train_preprocessed.csv")
TEST_CSV_OUTPUT  = os.path.join(OUTPUT_DIR, "quantum_test_preprocessed.csv")

# Output directory for additional visuals
VISUALS_DIR = OUTPUT_DIR + os.sep + "visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------

def is_power_of_two(n):
    """Check if an integer n is a power of 2."""
    return (n & (n - 1) == 0) and n != 0

def next_power_of_two(n):
    """Return the next power of two greater than or equal to n."""
    return 1 if n == 0 else 2**(n - 1).bit_length()

def pad_vector(vec):
    """
    Pad a 1D numpy array vec with zeros so its length becomes the next power of two.
    """
    current_length = len(vec)
    target_length = next_power_of_two(current_length)
    if target_length == current_length:
        return vec
    padded = np.zeros(target_length)
    padded[:current_length] = vec
    return padded

def normalize_vector(vec):
    """
    Normalize a 1D numpy array to unit norm. If the norm is zero, return the vector as-is.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def visualize_vector_lengths(lengths_before, lengths_after, output_path):
    """
    Plot histograms of vector L2 norms before and after normalization.
    """
    plt.figure(figsize=(10,4))

    plt.subplot(1,2,1)
    sns.histplot(lengths_before, bins=20, kde=True)
    plt.title("Original L2 Norms")

    plt.subplot(1,2,2)
    sns.histplot(lengths_after, bins=20, kde=True)
    plt.title("After Normalization (Should be ~1)")

    plt.suptitle("Distribution of L2 Norms Before and After Normalization")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved vector norm distribution plot to {output_path}")

def visualize_feature_sample(sample, output_path):
    """
    Plot a line plot of a single feature vector (after padding) to visualize its structure.
    """
    plt.figure(figsize=(10,4))
    plt.plot(sample, marker='o')
    plt.title("Sample Feature Vector After Normalization & Padding")
    plt.xlabel("Feature Index")
    plt.ylabel("Value")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved sample feature vector plot to {output_path}")

# -------------------------------------------------------------------
# 3. Preprocessing Function for Quantum Data
# -------------------------------------------------------------------

def preprocess_for_quantum(input_csv):
    """
    Reads a CSV file (with preprocessed features) and prepares the data for quantum modeling.
    For each sample (row):
      - Extracts the feature vector (all columns except 'class').
      - Records its original L2 norm.
      - Normalizes the feature vector to unit norm.
      - Pads the vector to the next power of two.
    
    Returns:
      - A new DataFrame containing the quantum-ready features with column names f0, f1, ..., f(N-1)
      - A numpy array or pandas Series of labels from the 'class' column.
      - Also returns arrays of original and normalized L2 norms for visualization.
    """
    df = pd.read_csv(input_csv)
    print(f"[INFO] Loaded {input_csv} with shape {df.shape}")
    
    # Separate features and label (assuming label column is called 'class')
    features_df = df.drop(columns=['class'])
    labels = df['class']
    
    # Convert to numpy array (each row is a sample)
    X = features_df.to_numpy()
    
    original_norms = np.linalg.norm(X, axis=1)
    
    # Process each sample
    quantum_features = []
    normalized_norms = []  # should be 1 for each sample after normalization
    
    for i, row in enumerate(X):
        normed = normalize_vector(row)
        normalized_norms.append(np.linalg.norm(normed))
        padded = pad_vector(normed)
        quantum_features.append(padded)
        
    quantum_features = np.array(quantum_features)
    print(f"[INFO] Processed features shape: {quantum_features.shape}")
    
    # Create a DataFrame for the quantum features with generated column names (f0, f1, ..., f_{N-1})
    num_features = quantum_features.shape[1]
    col_names = [f"f{i}" for i in range(num_features)]
    df_quantum = pd.DataFrame(quantum_features, columns=col_names)
    df_quantum['class'] = labels.reset_index(drop=True)
    
    # Generate visualizations:
    # 1. Norm distributions before and after normalization
    norm_plot_path = os.path.join(VISUALS_DIR, "vector_norm_distribution.png")
    visualize_vector_lengths(original_norms, normalized_norms, norm_plot_path)
    
    # 2. Visualize one sample feature vector after normalization and padding
    sample_plot_path = os.path.join(VISUALS_DIR, "sample_feature_vector.png")
    visualize_feature_sample(quantum_features[0], sample_plot_path)
    
    return df_quantum

# -------------------------------------------------------------------
# 4. Main Function
# -------------------------------------------------------------------

def main():
    print("[INFO] Starting Quantum Preprocessing...")
    # Preprocess training data for quantum model
    df_quantum_train = preprocess_for_quantum(TRAIN_CSV_INPUT)
    df_quantum_test  = preprocess_for_quantum(TEST_CSV_INPUT)
    
    # Save the resulting DataFrames to CSV
    df_quantum_train.to_csv(TRAIN_CSV_OUTPUT, index=False)
    df_quantum_test.to_csv(TEST_CSV_OUTPUT, index=False)
    
    print(f"[INFO] Saved quantum-ready training data to {TRAIN_CSV_OUTPUT}")
    print(f"[INFO] Saved quantum-ready testing data to {TEST_CSV_OUTPUT}")
    print("[INFO] Quantum Preprocessing complete.")

if __name__ == "__main__":
    main()
