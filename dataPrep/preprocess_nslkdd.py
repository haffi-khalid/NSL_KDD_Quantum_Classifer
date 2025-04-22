"""
Preprocess NSL-KDD 20% Training and Test Data

This script reads the NSL-KDD ARFF files (20% training set + test file),
performs preprocessing for binary classification, and generates CSV files.
The script also saves visualizations (label distribution, histograms, heatmap)
to help you document each step for your final year project.

Author: [Your Name]
Date: [Date]
"""

import arff
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# -------------------------------------------------------------------
# 1. Configuration: Paths & File Names
# -------------------------------------------------------------------

# Standard NSL-KDD attribute names (41 features + 'class')
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
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "class"
]

# Paths to your ARFF files (20% train + test)
TRAIN_FILE = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTrain+_20Percent.arff"
TEST_FILE  = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTest+.arff"

# Folder for saving visualizations
VISUALS_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\preprocessing_visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

# Folder for saving the preprocessed CSV files
CSV_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data"
os.makedirs(CSV_DIR, exist_ok=True)

# Desired output CSV file names
TRAIN_CSV_NAME = "nslkdd_train_preprocessed.csv"
TEST_CSV_NAME  = "nslkdd_test_preprocessed.csv"

# Full paths for the output CSV files
TRAIN_CSV_PATH = os.path.join(CSV_DIR, TRAIN_CSV_NAME)
TEST_CSV_PATH  = os.path.join(CSV_DIR, TEST_CSV_NAME)


# -------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------

def map_label_binary(label):
    """
    Maps a label string to binary:
    - 'normal' (case-insensitive) becomes 0,
    - anything else (an attack) becomes 1.
    """
    return 0 if str(label).strip().lower() == "normal" else 1

def read_arff_file(file_path):
    """
    Reads an ARFF file by first removing extraneous single quotes
    from the attribute definitions, then returns a pandas DataFrame
    with the standard NSL-KDD column names.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        # If the line starts with @attribute (case-insensitive), remove single quotes.
        if line.strip().lower().startswith("@attribute"):
            line = line.replace("'", "")
        cleaned_lines.append(line)
    
    file_content = "".join(cleaned_lines)
    # Use arff.loads() because we have a string in memory.
    dataset = arff.loads(file_content)
    data = dataset['data']
    import pandas as pd
    df = pd.DataFrame(data, columns=COLUMN_NAMES)
    return df


# def generate_visualizations(df, title_suffix="", output_prefix=""):
#     """
#     Generates and saves:
#       1) A label distribution plot (countplot).
#       2) Histograms for up to 6 numeric columns.
#       3) A correlation heatmap for numeric features.
#     All images are saved in VISUALS_DIR.
#     """
#     # 2.1 Label distribution
#     plt.figure(figsize=(6, 4))
#     sns.countplot(x="class", data=df)
#     plt.title(f"Label Distribution {title_suffix}")
#     plt.xlabel("Class (0=normal, 1=anomaly)")
#     plt.ylabel("Count")
#     label_dist_path = os.path.join(VISUALS_DIR, f"{output_prefix}label_distribution.png")
#     plt.savefig(label_dist_path)
#     plt.close()
#     print(f"[INFO] Saved label distribution to {label_dist_path}")

#     # 2.2 Histograms for numeric features
#     numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop("class", errors='ignore')
#     sample_numeric = numeric_cols[:6]  # pick first 6 for clarity
#     df[sample_numeric].hist(bins=20, figsize=(12, 8))
#     plt.suptitle(f"Histograms of Sample Numeric Features {title_suffix}")
#     hist_path = os.path.join(VISUALS_DIR, f"{output_prefix}numeric_histograms.png")
#     plt.savefig(hist_path)
#     plt.close()
#     print(f"[INFO] Saved numeric histograms to {hist_path}")

#     # 2.3 Correlation heatmap for numeric columns
#     if len(numeric_cols) > 1:
#         corr = df[numeric_cols].corr()
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
#         plt.title(f"Correlation Heatmap {title_suffix}")
#         heatmap_path = os.path.join(VISUALS_DIR, f"{output_prefix}correlation_heatmap.png")
#         plt.savefig(heatmap_path)
#         plt.close()
#         print(f"[INFO] Saved correlation heatmap to {heatmap_path}")
#     else:
#         print("[WARNING] Not enough numeric columns for a correlation heatmap.")


def generate_visualizations(df, title_suffix="", output_prefix=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    # Visual 1: Box Plots for a Subset of Numeric Features
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.drop("class", errors='ignore')
    sample_numeric = numeric_cols[:6]  # select first 6 numeric features for clarity
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[sample_numeric])
    plt.title(f"Box Plots of Sample Numeric Features {title_suffix}")
    boxplot_path = os.path.join(VISUALS_DIR, f"{output_prefix}boxplots.png")
    plt.savefig(boxplot_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved box plot visualization to {boxplot_path}")

    # Visual 2: Violin Plots for the Same Subset (with log scale on y-axis if needed)
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df[sample_numeric])
    plt.title(f"Violin Plots of Sample Numeric Features {title_suffix}")
    violin_path = os.path.join(VISUALS_DIR, f"{output_prefix}violinplots.png")
    plt.savefig(violin_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved violin plot visualization to {violin_path}")

    # Visual 3: KDE Plot with Log Scale for one numeric feature
    if len(sample_numeric) > 0:
        feature = sample_numeric[0]  # or pick any other representative feature
        plt.figure(figsize=(8, 6))
        sns.kdeplot(df[feature], fill=True)  # use fill=True instead of shade=True
        plt.xscale("log")
        plt.title(f"KDE Plot (Log Scale) for {feature} {title_suffix}")
        kde_path = os.path.join(VISUALS_DIR, f"{output_prefix}kde_{feature}_log.png")
        plt.savefig(kde_path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved KDE plot with log scale for {feature} to {kde_path}")


    # # Visual 4: Pair Plot (Scatter Matrix) for a subset of numeric features
    # if len(numeric_cols) >= 4:
    #     subset = df[numeric_cols[:4]]  # select 4 numeric features
    #     pairplot = sns.pairplot(subset)
    #     pairplot.fig.suptitle(f"Pair Plot of Sample Numeric Features {title_suffix}", y=1.02)
    #     pairplot_path = os.path.join(VISUALS_DIR, f"{output_prefix}pairplot.png")
    #     pairplot.savefig(pairplot_path, bbox_inches='tight')
    #     plt.close()
    #     print(f"[INFO] Saved pair plot visualization to {pairplot_path}")

    # # Visual 5: Correlation Heatmap (Exclude One-Hot Columns for Clarity)
    # exclude_prefixes = ("protocol_type_", "service_", "flag_")
    # numeric_cols_only = [col for col in df.columns if not col.startswith(exclude_prefixes) and col != "class"]
    # if len(numeric_cols_only) > 1:
    #     plt.figure(figsize=(10, 8))
    #     corr = df[numeric_cols_only].corr()
    #     sns.heatmap(corr, annot=False, cmap="coolwarm")
    #     plt.title(f"Correlation Heatmap {title_suffix} (Numeric Only)")
    #     heatmap_path = os.path.join(VISUALS_DIR, f"{output_prefix}correlation_heatmap_numeric.png")
    #     plt.savefig(heatmap_path, bbox_inches='tight')
    #     plt.close()
    #     print(f"[INFO] Saved numeric-only correlation heatmap to {heatmap_path}")
    # else:
    #     print("[WARNING] Not enough numeric columns for a numeric-only correlation heatmap.")


# -------------------------------------------------------------------
# 3. Main Preprocessing Function
# -------------------------------------------------------------------

def preprocess_nslkdd(train_file, test_file):
    """
    Reads the NSL-KDD (20% + test) ARFF files, then:
      1. Maps labels to binary.
      2. One-hot encodes categorical columns: [protocol_type, service, flag].
      3. Scales numeric features using Min-Max scaling.
      4. Generates dataset distribution visuals.
      5. Outputs CSV files for training and testing.

    Returns: Two DataFrames (train_preprocessed, test_preprocessed).
    """
    print("[INFO] Reading training file:", train_file)
    df_train = read_arff_file(train_file)
    print(f"[INFO] Training data shape (raw): {df_train.shape}")

    print("[INFO] Reading test file:", test_file)
    df_test = read_arff_file(test_file)
    print(f"[INFO] Test data shape (raw): {df_test.shape}")

    # 3.1 Convert labels to binary
    df_train["class"] = df_train["class"].apply(map_label_binary)
    df_test["class"]  = df_test["class"].apply(map_label_binary)

    # 3.2 Visualize label distribution & raw numeric data
    generate_visualizations(df_train, title_suffix="(Raw Training)", output_prefix="raw_train_")

    # 3.3 Identify categorical vs. numeric columns
    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in df_train.columns if c not in categorical_cols + ["class"]]

    # 3.4 One-hot encode categorical features
    df_train_cat = pd.get_dummies(df_train[categorical_cols], prefix=categorical_cols)
    df_test_cat  = pd.get_dummies(df_test[categorical_cols],  prefix=categorical_cols)

    # Make sure train/test have matching columns
    df_train_cat, df_test_cat = df_train_cat.align(df_test_cat, join="outer", axis=1, fill_value=0)

    # 3.5 Numeric features
    df_train_num = df_train[numeric_cols].astype(float)
    df_test_num  = df_test[numeric_cols].astype(float)

    # 3.6 Concatenate
    X_train = pd.concat([df_train_num, df_train_cat], axis=1)
    X_test  = pd.concat([df_test_num,  df_test_cat],  axis=1)

    # 3.7 Scale with Min-Max
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns)

    # 3.8 Re-append the label
    train_preprocessed = X_train_scaled.copy()
    train_preprocessed["class"] = df_train["class"].reset_index(drop=True)

    test_preprocessed = X_test_scaled.copy()
    test_preprocessed["class"] = df_test["class"].reset_index(drop=True)

    # 3.9 Visualize scaled data
    generate_visualizations(train_preprocessed, title_suffix="(Scaled Training)", output_prefix="scaled_train_")

    # 3.10 Print final shapes
    print(f"[INFO] Final training shape: {train_preprocessed.shape}")
    print(f"[INFO] Final testing shape:  {test_preprocessed.shape}")

    # 3.11 Save to CSV
    train_preprocessed.to_csv(TRAIN_CSV_PATH, index=False)
    test_preprocessed.to_csv(TEST_CSV_PATH, index=False)
    print(f"[INFO] Saved preprocessed training CSV => {TRAIN_CSV_PATH}")
    print(f"[INFO] Saved preprocessed testing CSV  => {TEST_CSV_PATH}")

    return train_preprocessed, test_preprocessed


# -------------------------------------------------------------------
# 4. Main Entry Point
# -------------------------------------------------------------------

def main():
    print("\n[INFO] Starting NSL-KDD Preprocessing...")
    preprocess_nslkdd(TRAIN_FILE, TEST_FILE)
    print("[INFO] Preprocessing complete.\n")

if __name__ == "__main__":
    main()
