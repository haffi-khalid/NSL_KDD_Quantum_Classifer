import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from torch.nn import MSELoss
from numpy.linalg import norm
from adversaries.quantum_adversaries.fgsm_quantum import fgsm_attack
from model.quantum_model import create_quantum_model

# === CONFIGURATION ===
BASE = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier"
OUTPUT_DIR = os.path.join(BASE, "Quantum_Eval_AdvTraining")
os.makedirs(OUTPUT_DIR, exist_ok=True)

paths = {
    "clean_model": os.path.join(BASE, "train", "weights", "quantum_model_clean_10layers.pt"),
    "fgsm_model": os.path.join(BASE, "train", "weights", "quantum_model_fgsm_100%_10layers.pt"),
    "bim_model": os.path.join(BASE, "train", "weights", "quantum_model_bim_100%_10layers.pt"),
    "test_csv": os.path.join(BASE, "pre-processed_data", "quantum", "quantum_test_preprocessed.csv"),
    "csv_out": os.path.join(OUTPUT_DIR, "adv_training_results.csv"),
    "plot_acc": os.path.join(OUTPUT_DIR, "accuracy_vs_epsilon.png"),
    "plot_loss": os.path.join(OUTPUT_DIR, "loss_vs_epsilon.png"),
}

# === DEVICE SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD MODELS ===
def load_model(path):
    model = create_quantum_model(n_layers=10).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

print("[INFO] Loading models...")
model_clean = load_model(paths["clean_model"])
model_fgsm = load_model(paths["fgsm_model"])
model_bim = load_model(paths["bim_model"])

# === LOAD TEST DATA ===
print("[INFO] Loading test data...")
df_test = pd.read_csv(paths["test_csv"])
X_test = torch.tensor(df_test.drop(columns=["class"]).values, dtype=torch.float32)
y_test = torch.tensor(df_test["class"].values, dtype=torch.float32)

# === EVALUATION ===
epsilons = np.arange(0.0, 1.05, 0.05)
results = []

print("[INFO] Starting evaluation across epsilon range...")
for eps in epsilons:
    print(f"[*] Epsilon = {eps:.2f}")
    X_adv = fgsm_attack(model_clean, X_test, y_test, eps).detach()

    def evaluate(model, X, y):
        preds = model(X.to(device)).squeeze().cpu().detach().numpy()
        preds_label = (preds > 0.5).astype(int)
        acc = np.mean(preds_label == y.numpy())
        loss = tf.keras.losses.binary_crossentropy(y.numpy().astype(np.float32), preds).numpy().mean()
        return acc, loss

    acc_clean, loss_clean = evaluate(model_clean, X_adv, y_test)
    acc_fgsm, loss_fgsm = evaluate(model_fgsm, X_adv, y_test)
    acc_bim, loss_bim = evaluate(model_bim, X_adv, y_test)

    results.append({
        "epsilon": eps,
        "acc_clean": acc_clean,
        "acc_fgsm": acc_fgsm,
        "acc_bim": acc_bim,
        "loss_clean": loss_clean,
        "loss_fgsm": loss_fgsm,
        "loss_bim": loss_bim
    })

# === SAVE RESULTS ===
results_df = pd.DataFrame(results)
results_df.to_csv(paths["csv_out"], index=False)
print(f"[OK] Results saved to {paths['csv_out']}")

# === PLOT ACCURACY ===
plt.figure(figsize=(10, 6))
plt.plot(results_df["epsilon"], results_df["acc_clean"], label="Clean Model")
plt.plot(results_df["epsilon"], results_df["acc_fgsm"], label="FGSM-Trained Model")
plt.plot(results_df["epsilon"], results_df["acc_bim"], label="BIM-Trained Model")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon (FGSM Attacks)")
plt.legend()
plt.grid(True)
plt.savefig(paths["plot_acc"])
plt.close()
print(f"[OK] Accuracy plot saved to {paths['plot_acc']}")

# === PLOT LOSS ===
plt.figure(figsize=(10, 6))
plt.plot(results_df["epsilon"], results_df["loss_clean"], label="Clean Model")
plt.plot(results_df["epsilon"], results_df["loss_fgsm"], label="FGSM-Trained Model")
plt.plot(results_df["epsilon"], results_df["loss_bim"], label="BIM-Trained Model")
plt.xlabel("Epsilon")
plt.ylabel("Adversarial Test Loss")
plt.title("Adversarial Loss vs Epsilon (FGSM Attacks)")
plt.legend()
plt.grid(True)
plt.savefig(paths["plot_loss"])
plt.close()
print(f"[OK] Loss plot saved to {paths['plot_loss']}")
