import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from model.quantum_model import create_quantum_model
from adversaries.quantum_adversaries.fgsm_quantum import fgsm_attack
from dataPrep.preprocess_nslkdd_cnn import preprocess_for_cnn

# === PATH SETUP ===
BASE_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Quantum-vs-CNN-2ndtry"

TRAIN_ARFF = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTrain+_20Percent.arff"
TEST_ARFF  = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\archive\KDDTest+.arff"

paths = {
    "quantum_clean": os.path.join(BASE_DIR, "..", "train", "weights", "quantum_model_clean_10layers.pt"),
    "quantum_adv": os.path.join(BASE_DIR, "..", "train", "weights", "quantum_model_fgsm_100%_10layers.pt"),
    "cnn_clean": os.path.join(BASE_DIR, "..", "train-cnn", "weights", "cnn_model_clean.h5"),
    "cnn_adv": os.path.join(BASE_DIR, "..", "train-cnn", "weights", "cnn_model_fgsm_100%_final.h5"),
    "quantum_test_csv": os.path.join(BASE_DIR, "..", "pre-processed_data", "quantum", "quantum_test_preprocessed.csv"),
    "cnn_preprocessed": os.path.join(BASE_DIR, "quantum-generated-cnn-preprocessed"),
    "metrics": os.path.join(BASE_DIR, "metrics"),
    "plots": os.path.join(BASE_DIR, "plots"),
    "heatmaps": os.path.join(BASE_DIR, "heatmaps"),
}

for p in [paths["metrics"], paths["plots"], paths["heatmaps"], paths["cnn_preprocessed"]]:
    os.makedirs(p, exist_ok=True)

# === LOAD MODELS ===
print("[INFO] Loading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

quantum_clean = create_quantum_model(n_layers=10)
quantum_clean.load_state_dict(torch.load(paths["quantum_clean"]))
quantum_clean.to(device).eval()

quantum_adv = create_quantum_model(n_layers=10)
quantum_adv.load_state_dict(torch.load(paths["quantum_adv"]))
quantum_adv.to(device).eval()

cnn_clean = tf.keras.models.load_model(paths["cnn_clean"])
cnn_adv = tf.keras.models.load_model(paths["cnn_adv"])

# === LOAD QUANTUM TEST DATA ===
print("[INFO] Loading quantum test data...")
df_test = pd.read_csv(paths["quantum_test_csv"])
X_test = torch.tensor(df_test.drop(columns=["class"]).values, dtype=torch.float32)
y_test = torch.tensor(df_test["class"].values, dtype=torch.float32)

# === FIT SCALER ON CNN TRAIN DATA FOR CONVERSION ===
print("[INFO] Fitting CNN column alignment and scaler...")
df_train_cnn, _ = preprocess_for_cnn(TRAIN_ARFF, TEST_ARFF)
scaler = MinMaxScaler()
X_train_cnn = df_train_cnn.drop(columns=["class"])
scaler.fit(X_train_cnn)
cnn_columns = X_train_cnn.columns

def convert_quantum_adv_to_cnn_format(df_adv):
    df = df_adv.copy()
    X = df.drop(columns=["class"])
    y = df["class"]

    for col in cnn_columns:
        if col not in X.columns:
            X[col] = 0.0
    X = X[cnn_columns]
    X_scaled = scaler.transform(X)

    df_final = pd.DataFrame(X_scaled, columns=cnn_columns)
    df_final["class"] = y.values
    return df_final

# === STORAGE ===
epsilons = np.arange(0.0, 1.05, 0.05)
results = []

# === FGSM + Evaluation ===
for eps in epsilons:
    print(f"[*] Epsilon = {eps:.2f}")
    X_adv = fgsm_attack(quantum_clean, X_test, y_test, eps).detach()

    # Quantum Eval
    q_clean_probs = quantum_clean(X_adv.to(device)).squeeze().cpu().detach().numpy()
    q_adv_probs = quantum_adv(X_adv.to(device)).squeeze().cpu().detach().numpy()

    q_clean_preds = (q_clean_probs > 0.5).astype(int)
    q_adv_preds = (q_adv_probs > 0.5).astype(int)

    acc_q_clean = np.mean(q_clean_preds == y_test.numpy())
    acc_q_adv = np.mean(q_adv_preds == y_test.numpy())

    loss_q_clean = tf.keras.losses.binary_crossentropy(y_test.numpy().astype(np.float32), q_clean_probs).numpy().mean()
    loss_q_adv = tf.keras.losses.binary_crossentropy(y_test.numpy().astype(np.float32), q_adv_probs).numpy().mean()

    # CNN Eval (after conversion)
    X_adv_np = X_adv.numpy()
    X_adv_df = pd.DataFrame(X_adv_np)
    X_adv_df["class"] = y_test.numpy()
    cnn_ready = convert_quantum_adv_to_cnn_format(X_adv_df)

    # Save preprocessed converted data for CNN
    cnn_ready.to_csv(os.path.join(paths["cnn_preprocessed"], f"cnn_ready_eps_{eps:.2f}.csv"), index=False)

    cnn_X = cnn_ready.drop(columns=["class"]).values
    cnn_y = cnn_ready["class"].values.astype(np.float32)

    preds_clean = cnn_clean.predict(cnn_X, verbose=0).squeeze()
    preds_adv = cnn_adv.predict(cnn_X, verbose=0).squeeze()

    acc_c_clean = np.mean((preds_clean > 0.5).astype(int) == cnn_y.astype(int))
    acc_c_adv = np.mean((preds_adv > 0.5).astype(int) == cnn_y.astype(int))

    loss_c_clean = tf.keras.losses.binary_crossentropy(cnn_y, preds_clean).numpy().mean()
    loss_c_adv = tf.keras.losses.binary_crossentropy(cnn_y, preds_adv).numpy().mean()

    # Store Results
    results.append({
        "epsilon": eps,
        "acc_quantum_clean": acc_q_clean,
        "acc_quantum_adv": acc_q_adv,
        "loss_quantum_clean": loss_q_clean,
        "loss_quantum_adv": loss_q_adv,
        "acc_cnn_clean": acc_c_clean,
        "acc_cnn_adv": acc_c_adv,
        "loss_cnn_clean": loss_c_clean,
        "loss_cnn_adv": loss_c_adv,
    })

    # Confusion Matrices
    cm_cnn = confusion_matrix(cnn_y.astype(int), (preds_clean > 0.5).astype(int))
    cm_quantum = confusion_matrix(y_test.numpy().astype(int), q_clean_preds)

    sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - CNN Clean - Eps {eps:.2f}")
    plt.savefig(os.path.join(paths["heatmaps"], f"cnn_cm_eps_{eps:.2f}.png"))
    plt.close()

    sns.heatmap(cm_quantum, annot=True, fmt="d", cmap="Reds")
    plt.title(f"Confusion Matrix - Quantum Clean - Eps {eps:.2f}")
    plt.savefig(os.path.join(paths["heatmaps"], f"quantum_cm_eps_{eps:.2f}.png"))
    plt.close()

# === SAVE METRICS CSV ===
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(paths["metrics"], "metrics_quantum_vs_cnn_2nd_try.csv"), index=False)

# === PLOTS ===
plt.figure(figsize=(10, 6))
plt.plot(metrics_df["epsilon"], metrics_df["acc_quantum_clean"], label="Quantum Clean")
plt.plot(metrics_df["epsilon"], metrics_df["acc_quantum_adv"], label="Quantum Adv Trained")
plt.plot(metrics_df["epsilon"], metrics_df["acc_cnn_clean"], label="CNN Clean")
plt.plot(metrics_df["epsilon"], metrics_df["acc_cnn_adv"], label="CNN Adv Trained")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epsilon")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(paths["plots"], "accuracy_vs_epsilon.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(metrics_df["epsilon"], metrics_df["loss_quantum_clean"], label="Quantum Clean")
plt.plot(metrics_df["epsilon"], metrics_df["loss_quantum_adv"], label="Quantum Adv Trained")
plt.plot(metrics_df["epsilon"], metrics_df["loss_cnn_clean"], label="CNN Clean")
plt.plot(metrics_df["epsilon"], metrics_df["loss_cnn_adv"], label="CNN Adv Trained")
plt.xlabel("Epsilon")
plt.ylabel("Adversarial Test Loss")
plt.title("Adversarial Loss vs Epsilon")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(paths["plots"], "loss_vs_epsilon.png"))
plt.close()

print("[OK] All evaluations, plots, and converted datasets saved successfully.")
