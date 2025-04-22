import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, log_loss
from model.quantum_model import create_quantum_model
from adversaries.quantum_adversaries.fgsm_quantum import fgsm_attack
import matplotlib.pyplot as plt

# === Configuration ===
epsilons = np.arange(0, 1.05, 0.05)
quantum_clean_path = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train\weights\quantum_model_clean_10layers.pt"
quantum_adv_path   = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train\weights\quantum_model_fgsm_100%_10layers.pt"
cnn_adv_path       = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train-cnn\weights\cnn_model_fgsm_100%_final.h5"

output_metrics = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\Attacks_against_each_other\Quantum_vs_CNN\metrics\metrics_quantum_vs_CNN.csv"

# === Load test data ===
df_test = pd.read_csv(r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\quantum\quantum_test_preprocessed.csv")
print("[DEBUG] Columns in df_test:", df_test.columns.tolist())

# Fix column name from 'label' ➤ 'class'
X_test = torch.tensor(df_test.drop(columns=["class"]).values, dtype=torch.float32)
y_test = torch.tensor(df_test["class"].values, dtype=torch.float32)



# === Load models ===
quantum_clean = create_quantum_model(n_layers=10)
quantum_clean.load_state_dict(torch.load(quantum_clean_path))
quantum_clean.eval()

quantum_adv = create_quantum_model(n_layers=10)
quantum_adv.load_state_dict(torch.load(quantum_adv_path))
quantum_adv.eval()

cnn_adv = load_model(cnn_adv_path)

# === Utility: pad to CNN shape ===
def pad_to_cnn_input(tensor, target_dim=119):
    pad_size = target_dim - tensor.shape[1]
    return torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)

def evaluate_keras_model(model, X, y):
    preds = model.predict(X.numpy(), verbose=0).squeeze()
    preds_class = (preds > 0.5).astype(int)
    return accuracy_score(y, preds_class), log_loss(y, preds)

def evaluate_torch_model(model, X, y):
    with torch.no_grad():
        outputs = model(X).squeeze()
        preds_class = (outputs > 0.5).int()
        return accuracy_score(y, preds_class), log_loss(y, outputs.numpy())

# === Evaluation Loop ===
records = []

for eps in epsilons:
    print(f"[INFO] Evaluating for epsilon={eps:.2f}")
    adv_X = fgsm_attack(quantum_clean, X_test, y_test, eps)
    adv_X_padded = pad_to_cnn_input(adv_X, target_dim=119)

    acc_q, loss_q = evaluate_torch_model(quantum_adv, adv_X, y_test)
    acc_c, loss_c = evaluate_keras_model(cnn_adv, adv_X_padded, y_test)

    records.append({
        "epsilon": eps,
        "accuracy_on_quantum": acc_q,
        "loss_on_quantum": loss_q,
        "accuracy_on_cnn": acc_c,
        "loss_on_cnn": loss_c
    })

# Save metrics
df = pd.DataFrame(records)
df.to_csv(output_metrics, index=False)
print(f"[INFO] Saved metrics to {output_metrics}")
