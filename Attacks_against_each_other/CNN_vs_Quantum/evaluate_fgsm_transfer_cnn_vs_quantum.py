import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from model.quantum_model import create_quantum_model

# Epsilon values to sweep
epsilons = np.round(np.arange(0, 1.05, 0.05), 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === File Paths === #
# ✅ Clean CNN model (used to generate adversarial examples)
cnn_clean_path = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train-cnn\weights\cnn_model_clean.h5"

# ✅ Adversarially trained Quantum model (for evaluating robustness)
quantum_adv_path = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train\weights\quantum_model_fgsm_100%_10layers.pt"

# ✅ Clean CNN test data
cnn_test_data_path = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\cnn_test_preprocessed.csv"

# ✅ Where the quantum-compatible version of CNN adversarial data will be saved
preprocessed_quantum_csv_dir = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\CNN_vs_Quantum\cnn-produced-preprocessed-quantum-data"

# ✅ Where to save the evaluation metrics (accuracy/loss vs epsilon)
metrics_save_path = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\CNN_vs_Quantum\metrics\metrics_cnn_vs_quantum.csv"

# ✅ Folder for accuracy/loss line plots
plots_dir = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\CNN_vs_Quantum\plots"

# ✅ Folder for heatmaps
heatmaps_dir = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\CNN_vs_Quantum\heatmaps"

# ✅ Folder for confusion matrices and any analysis plots
analysis_dir = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\CNN_vs_Quantum\analysis"


# Ensure folders exist
for folder in [preprocessed_quantum_csv_dir, plots_dir, heatmaps_dir, analysis_dir, os.path.dirname(metrics_save_path)]:
    os.makedirs(folder, exist_ok=True)

# === Load Data === #
df = pd.read_csv(cnn_test_data_path)
X_cnn = df.drop(columns=["class"]).values
y = df["class"].values
y_tensor = torch.tensor(y, dtype=torch.float32)

# === Load Models === #
cnn_model = tf.keras.models.load_model(cnn_clean_path)
quantum_model = create_quantum_model(n_layers=10).to(device)
quantum_model.load_state_dict(torch.load(quantum_adv_path))
quantum_model.eval()

# === Loss Functions === #
loss_fn = tf.keras.losses.BinaryCrossentropy()
bce_torch = nn.BCELoss()

# === FGSM Attack & Evaluation === #
metrics = []

for eps in epsilons:
    print(f"[*] Epsilon = {eps:.2f}")
    X_adv = tf.convert_to_tensor(X_cnn, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(X_adv)
        pred = cnn_model(X_adv)
        loss = loss_fn(y, pred)
    grad = tape.gradient(loss, X_adv)
    signed_grad = tf.sign(grad)
    adv_x = X_adv + eps * signed_grad
    adv_x = tf.clip_by_value(adv_x, 0, 1).numpy()

    # CNN Eval
    pred_cnn = cnn_model.predict(adv_x).flatten()
    pred_cnn_cls = (pred_cnn > 0.5).astype(int)
    acc_cnn = np.mean(pred_cnn_cls == y)
    loss_cnn = loss_fn(y, pred_cnn).numpy()

    # Quantum Prep
    scaler = MinMaxScaler()
    adv_x_scaled = scaler.fit_transform(adv_x)
    quantum_input = adv_x_scaled[:, :64]
    quantum_input = torch.tensor(quantum_input, dtype=torch.float32)

    # Save quantum-compatible data
    pd.DataFrame(quantum_input.numpy()).to_csv(
        os.path.join(preprocessed_quantum_csv_dir, f"adv_quantum_eps_{eps:.2f}.csv"),
        index=False
    )

    # Quantum Eval
    with torch.no_grad():
        q_pred = quantum_model(quantum_input.to(device)).squeeze().cpu().numpy()
    q_pred_cls = (q_pred > 0.5).astype(int)
    acc_quantum = np.mean(q_pred_cls == y)
    loss_quantum = bce_torch(torch.tensor(q_pred, dtype=torch.float32), y_tensor).item()

    metrics.append({
        "epsilon": eps,
        "acc_cnn": acc_cnn,
        "loss_cnn": loss_cnn,
        "acc_quantum": acc_quantum,
        "loss_quantum": loss_quantum
    })

# Save Metrics
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(metrics_save_path, index=False)

# === Plotting === #
# Accuracy Plot
plt.figure()
plt.plot(df_metrics["epsilon"], df_metrics["acc_cnn"], marker="o", label="CNN")
plt.plot(df_metrics["epsilon"], df_metrics["acc_quantum"], marker="o", label="Quantum")
plt.title("Accuracy vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "accuracy_vs_epsilon.png"))
plt.close()

# Loss Plot
plt.figure()
plt.plot(df_metrics["epsilon"], df_metrics["loss_cnn"], marker="o", label="CNN")
plt.plot(df_metrics["epsilon"], df_metrics["loss_quantum"], marker="o", label="Quantum")
plt.title("Adversarial Test Loss vs Epsilon")
plt.xlabel("Epsilon")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_dir, "loss_vs_epsilon.png"))
plt.close()

# === Heatmaps & Confusion Matrices (Final Epsilon) === #
final_preds_cnn = (cnn_model.predict(adv_x) > 0.5).astype(int)
final_preds_quantum = (quantum_model(quantum_input.to(device)).squeeze().detach().cpu().numpy() > 0.5).astype(int)

for name, preds in [("cnn", final_preds_cnn), ("quantum", final_preds_quantum)]:
    cm = confusion_matrix(y, preds)
    df_cm = pd.DataFrame(cm, index=["Normal", "Attack"], columns=["Pred Normal", "Pred Attack"])
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"Confusion Matrix - {name.upper()} @ Eps=1.0")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(heatmaps_dir, f"heatmap_{name}.png"))
    plt.savefig(os.path.join(analysis_dir, f"confusion_matrix_{name}.png"))
    plt.close()
