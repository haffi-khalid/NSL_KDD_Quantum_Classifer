# Author: Muhammad Haffi Khalid
# Project: Effects of Quantum Embeddings on the Generalization of Adversarially Trained Quantum Classifiers

# Description:
# This script reads evaluation results from `metrics_quantum_vs_CNN.csv`, 
# and generates:
# - Accuracy vs Epsilon plots
# - Adversarial Test Loss vs Epsilon plots
# - Heatmaps for accuracy and loss
# - Subplot comparisons
# All outputs are saved in the `Quantum_vs_CNN` analysis folder structure.

# Requirements:
# - matplotlib
# - seaborn
# - pandas

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📂 File paths
METRICS_PATH = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\Quantum_vs_CNN\metrics\metrics_quantum_vs_CNN.csv"
BASE_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\Quantum_vs_CNN"

ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
HEATMAPS_DIR = os.path.join(BASE_DIR, "heatmaps")

# 📁 Create directories if not exist
for folder in [ANALYSIS_DIR, PLOTS_DIR, HEATMAPS_DIR]:
    os.makedirs(folder, exist_ok=True)

# 📊 Load metrics
df = pd.read_csv(METRICS_PATH)

# 🎨 Plot Accuracy vs Epsilon
plt.figure(figsize=(10, 6))
plt.plot(df["epsilon"], df["accuracy_on_quantum"], marker='o', label="Quantum Model")
plt.plot(df["epsilon"], df["accuracy_on_cnn"], marker='s', label="CNN Model")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon (Quantum-Generated Adversarial Examples)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_epsilon.png"))
plt.close()

# 🎨 Plot Loss vs Epsilon
plt.figure(figsize=(10, 6))
plt.plot(df["epsilon"], df["loss_on_quantum"], marker='o', label="Quantum Model")
plt.plot(df["epsilon"], df["loss_on_cnn"], marker='s', label="CNN Model")
plt.xlabel("Epsilon")
plt.ylabel("Adversarial Test Loss")
plt.title("Adversarial Test Loss vs Epsilon (Quantum-Generated Adversarial Examples)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(PLOTS_DIR, "loss_vs_epsilon.png"))
plt.close()

# 🔥 Heatmap for Accuracy
acc_matrix = df[["accuracy_on_quantum", "accuracy_on_cnn"]].T
acc_matrix.columns = df["epsilon"].round(2)
plt.figure(figsize=(10, 3))
sns.heatmap(acc_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "Accuracy"})
plt.title("Accuracy Heatmap")
plt.yticks(rotation=0)
plt.savefig(os.path.join(HEATMAPS_DIR, "accuracy_heatmap.png"))
plt.close()

# 🔥 Heatmap for Loss
loss_matrix = df[["loss_on_quantum", "loss_on_cnn"]].T
loss_matrix.columns = df["epsilon"].round(2)
plt.figure(figsize=(10, 3))
sns.heatmap(loss_matrix, annot=True, fmt=".2f", cmap="Reds", cbar_kws={"label": "Loss"})
plt.title("Loss Heatmap")
plt.yticks(rotation=0)
plt.savefig(os.path.join(HEATMAPS_DIR, "loss_heatmap.png"))
plt.close()

# 🧱 Subplot Comparison (Accuracy + Loss)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy subplot
axs[0].plot(df["epsilon"], df["accuracy_on_quantum"], marker='o', label="Quantum")
axs[0].plot(df["epsilon"], df["accuracy_on_cnn"], marker='s', label="CNN")
axs[0].set_title("Accuracy vs Epsilon")
axs[0].set_xlabel("Epsilon")
axs[0].set_ylabel("Accuracy")
axs[0].grid(True)
axs[0].legend()

# Loss subplot
axs[1].plot(df["epsilon"], df["loss_on_quantum"], marker='o', label="Quantum")
axs[1].plot(df["epsilon"], df["loss_on_cnn"], marker='s', label="CNN")
axs[1].set_title("Adversarial Test Loss vs Epsilon")
axs[1].set_xlabel("Epsilon")
axs[1].set_ylabel("Loss")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(ANALYSIS_DIR, "accuracy_and_loss_subplots.png"))
plt.close()

print("[OK] All plots and heatmaps saved successfully in their respective folders.")
