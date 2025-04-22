
# preprocess_cnn.py

# This script preprocesses the original NSL-KDD ARFF files for use with a CNN classifier.
# It reads the ARFF files, performs the following operations:
#   1. Maps the class label to binary: "normal" -> 0; all attack types -> 1.
#   2. One-hot encodes the categorical features: 'protocol_type', 'service', 'flag'.
#   3. Scales numeric features using Min-Max scaling.
  
# Unlike the quantum pre-processing script, this version does NOT:
#   - Normalize each data sample to unit L2 norm.
#   - Pad the feature vector to a specific length.

# The output is two CSV files:
#   - A training CSV file for the 20% training set.
#   - A testing CSV file for the test set.
  
# These files will be saved in the folder:
#   C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data

# Folder structure is assumed to be as follows:
  
# NSL-KDD Quantum Classifier/
#   ├── pre-processed_data/
#   │     ├── cnn_train_preprocessed.csv
#   │     └── cnn_test_preprocessed.csv
#   ├── (other folders and files)
  
# Requirements:
#   pip install liac-arff pandas numpy scikit-learn matplotlib seaborn

# Author: [Your Name]
# Date: [Date]

import os
import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Configuration: File Paths & Output Files
# -----------------------------
# Input ARFF file paths
TRAIN_ARFF = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTrain+_20Percent.arff"
TEST_ARFF  = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTest+.arff"

# Expected standard column names in NSL-KDD (41 features + 1 label)
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"  # label column
]

# Output CSV folder & file names (for CNN preprocessed data)
OUTPUT_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_CSV_OUTPUT = os.path.join(OUTPUT_DIR, "cnn_train_preprocessed.csv")
TEST_CSV_OUTPUT  = os.path.join(OUTPUT_DIR, "cnn_test_preprocessed.csv")

# -----------------------------
# 2. Helper Functions
# -----------------------------
def read_arff(file_path):
    """
    Reads an ARFF file using liac-arff.
    
    To handle extraneous quotes, this function removes single quotes from
    lines starting with '@attribute'. Returns a pandas DataFrame with COLUMN_NAMES.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # Remove extra single quotes from attribute lines for compatibility.
        if line.strip().lower().startswith("@attribute"):
            line = line.replace("'", "")
        cleaned_lines.append(line)
    
    content = "".join(cleaned_lines)
    dataset = arff.loads(content)
    data = dataset['data']
    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    return df

def map_label(label):
    """
    Maps a class label to binary for CNN training:
      - 'normal' (case-insensitive) -> 0,
      - all others -> 1.
    """
    return 0 if str(label).strip().lower() == "normal" else 1

def generate_visualizations(df, title_suffix="", output_prefix="cnn_"):
    """
    Generates basic visualizations for the raw processed data.
    Creates:
      - A countplot of the class distribution.
      - Histograms for a selection of numeric features.
      - A correlation heatmap of numeric features.
    Visuals are saved in OUTPUT_DIR.
    """
    # Create a folder for visuals (if desired, you could set a separate visuals folder)
    visuals_dir = os.path.join(OUTPUT_DIR, "visuals_cnn")
    os.makedirs(visuals_dir, exist_ok=True)
    
    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="class", data=df)
    plt.title(f"Class Distribution {title_suffix}")
    count_path = os.path.join(visuals_dir, f"{output_prefix}class_distribution.png")
    plt.savefig(count_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved class distribution plot to {count_path}")
    
    # Histograms for numeric features (select first 6 numeric columns)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.drop("class", errors='ignore')
    sample_numeric = numeric_cols[:6]
    df[sample_numeric].hist(bins=20, figsize=(12, 8))
    plt.suptitle(f"Histograms of Sample Numeric Features {title_suffix}")
    hist_path = os.path.join(visuals_dir, f"{output_prefix}numeric_histograms.png")
    plt.savefig(hist_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved histograms to {hist_path}")
    
    # Correlation heatmap for numeric features
    plt.figure(figsize=(10,8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title(f"Correlation Heatmap {title_suffix}")
    heatmap_path = os.path.join(visuals_dir, f"{output_prefix}correlation_heatmap.png")
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved correlation heatmap to {heatmap_path}")

# -----------------------------
# 3. Preprocessing Function for CNN Data
# -----------------------------
def preprocess_for_cnn(train_arff, test_arff):
    """
    Reads the NSL-KDD ARFF files for training and testing,
    performs the following steps:
      1. Maps class labels to binary.
      2. One-hot encodes categorical columns: 'protocol_type', 'service', 'flag'.
      3. Extracts numeric features (all others) and converts to float.
      4. Scales all features (numeric and the one-hot encoded ones) using Min-Max scaling.
      5. Generates visualizations for the preprocessed data.
      6. Outputs two DataFrames (train, test) ready for training a CNN.
      
    Returns:
        (df_train_processed, df_test_processed)
    """
    # 3.1 Read the ARFF files.
    print("[INFO] Reading training ARFF file...")
    df_train = read_arff(train_arff)
    print(f"[INFO] Training data shape (raw): {df_train.shape}")
    
    print("[INFO] Reading testing ARFF file...")
    df_test = read_arff(test_arff)
    print(f"[INFO] Testing data shape (raw): {df_test.shape}")
    
    # 3.2 Map the class labels to binary.
    df_train["class"] = df_train["class"].apply(map_label)
    df_test["class"]  = df_test["class"].apply(map_label)
    
    # 3.3 Separate categorical and numeric features.
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in df_train.columns if c not in categorical_cols + ["class"]]
    
    # 3.4 One-hot encode categorical columns.
    df_train_cat = pd.get_dummies(df_train[categorical_cols], prefix=categorical_cols)
    df_test_cat = pd.get_dummies(df_test[categorical_cols], prefix=categorical_cols)
    
    # Align columns so that train and test have identical one-hot columns.
    df_train_cat, df_test_cat = df_train_cat.align(df_test_cat, join="outer", axis=1, fill_value=0)
    
    # 3.5 Extract numeric features.
    df_train_num = df_train[numeric_cols].astype(float)
    df_test_num = df_test[numeric_cols].astype(float)
    
    # 3.6 Concatenate numeric and one-hot features.
    df_train_combined = pd.concat([df_train_num, df_train_cat], axis=1)
    df_test_combined  = pd.concat([df_test_num, df_test_cat], axis=1)
    
    # 3.7 Scale features using Min-Max Scaling.
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(df_train_combined)
    test_scaled = scaler.transform(df_test_combined)
    
    # Convert back to DataFrame and preserve column names.
    df_train_scaled = pd.DataFrame(train_scaled, columns=df_train_combined.columns)
    df_test_scaled = pd.DataFrame(test_scaled, columns=df_train_combined.columns)
    
    # 3.8 Append the class column.
    df_train_processed = df_train_scaled.copy()
    df_train_processed["class"] = df_train["class"].reset_index(drop=True)
    
    df_test_processed = df_test_scaled.copy()
    df_test_processed["class"] = df_test["class"].reset_index(drop=True)
    
    # 3.9 Generate visualizations.
    generate_visualizations(df_train_processed, title_suffix=" (CNN Preprocessed - Train)", output_prefix="cnn_train_")
    generate_visualizations(df_test_processed, title_suffix=" (CNN Preprocessed - Test)", output_prefix="cnn_test_")
    
    return df_train_processed, df_test_processed

# -----------------------------
# 4. Main Entry Point
# -----------------------------
def main():
    print("[INFO] Starting CNN Preprocessing for NSL-KDD ARFF files...")
    train_df, test_df = preprocess_for_cnn(TRAIN_ARFF, TEST_ARFF)
    
    # Save the processed DataFrames as CSV files for CNN.
    train_df.to_csv(TRAIN_CSV_OUTPUT, index=False)
    test_df.to_csv(TEST_CSV_OUTPUT, index=False)
    
    print(f"[INFO] Saved CNN preprocessed training data to {TRAIN_CSV_OUTPUT}")
    print(f"[INFO] Saved CNN preprocessed testing data to {TEST_CSV_OUTPUT}")

if __name__ == "__main__":
    main()
