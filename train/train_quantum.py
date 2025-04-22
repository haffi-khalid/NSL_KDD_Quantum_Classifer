"""
Quantum Classifier Training Script (PyTorch version)

This script trains a quantum variational classifier using amplitude embedding and a single-output qubit.
The model is built using PennyLane's TorchLayer and trained using PyTorch.

Features:
- Clean training or adversarial training using FGSM/BIM
- Adjustable number of variational layers (hardware-efficient ansatz)
- Adjustable adversarial ratio (e.g., 50% clean + 50% adversarial)
- Saves training/testing loss and accuracy
- Saves adversarial evaluation metrics (loss, acc)
- Saves confusion matrix
- Saves model weights

Author: Muhammad Haffi Khalid
Final Year Project: Effects of Quantum Embeddings on the Generalization of Adversarially Trained Quantum Classifiers
"""
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model.quantum_model import create_quantum_model
from adversaries.quantum_adversaries.fgsm_quantum import fgsm_attack
from adversaries.quantum_adversaries.bim_quantum import bim_attack

# === CONFIG ===
ATTACK_TYPE = "bim"       # Options: "clean", "fgsm", "bim"
ADV_RATIO = 1.0            # % of adversarial samples in training
N_LAYERS = 10              # Number of variational layers
EPSILON = 0.1              # Perturbation size
ALPHA = 0.01               # BIM step size
ITERATIONS = 3            # BIM iterations
EPOCHS = 200
BATCH_SIZE = 64
SEED = 42

torch.manual_seed(SEED)

# === PATHS ===
DATA_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\quantum"
SAVE_DIR = rf"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\quantum-gen_data\{ATTACK_TYPE}"
WEIGHT_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train\weights"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)

# === DATA LOADING ===
def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "quantum_train_preprocessed.csv")).sample(frac=1, random_state=SEED)
    test_df = pd.read_csv(os.path.join(DATA_DIR, "quantum_test_preprocessed.csv")).sample(frac=1, random_state=SEED)

    X_train = torch.tensor(train_df.drop(columns="class").values, dtype=torch.float32)
    y_train = torch.tensor(train_df["class"].values, dtype=torch.float32)

    X_test = torch.tensor(test_df.drop(columns="class").values, dtype=torch.float32)
    y_test = torch.tensor(test_df["class"].values, dtype=torch.float32)

    return X_train, y_train, X_test, y_test

# === METRICS SAVING ===
def save_metrics(train_losses, train_accuracies, test_acc, test_loss):
    pd.DataFrame(train_losses).to_csv(os.path.join(SAVE_DIR, "train_loss.csv"), index=False)
    pd.DataFrame(train_accuracies).to_csv(os.path.join(SAVE_DIR, "train_accuracy.csv"), index=False)
    pd.DataFrame([test_loss]).to_csv(os.path.join(SAVE_DIR, "test_loss.csv"), index=False)
    pd.DataFrame([test_acc]).to_csv(os.path.join(SAVE_DIR, "test_accuracy.csv"), index=False)

def save_adv_metrics(y_true, y_probs, adv_loss):
    acc = ((y_probs > 0.5).float().flatten() == y_true).float().mean().item()
    pd.DataFrame([adv_loss]).to_csv(os.path.join(SAVE_DIR, "adv_test_loss.csv"), index=False)
    pd.DataFrame([acc]).to_csv(os.path.join(SAVE_DIR, "adv_test_accuracy.csv"), index=False)
    print(f"[RESULT] Adversarial Test Accuracy: {acc*100:.2f}% | Loss: {adv_loss:.4f}")

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
    plt.close()

# === TRAINING ===
def train():
    print(f"[INFO] Training Mode: {ATTACK_TYPE.upper()} | Layers: {N_LAYERS} | Adversarial Ratio: {ADV_RATIO}")

    X_train, y_train, X_test, y_test = load_data()
    model = create_quantum_model(n_layers=N_LAYERS)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    if ATTACK_TYPE in ["fgsm", "bim"]:
        print(f"[INFO] Generating {ATTACK_TYPE.upper()} adversarial training data...")

        if ATTACK_TYPE == "fgsm":
            adv_X = fgsm_attack(model, X_train, y_train, EPSILON)
        else:
            adv_X = bim_attack(model, X_train, y_train, EPSILON, ALPHA, ITERATIONS)

        adv_n = int(ADV_RATIO * len(X_train))
        clean_n = len(X_train) - adv_n

        X_train = torch.cat([X_train[:clean_n], adv_X[:adv_n]], dim=0)
        y_train = torch.cat([y_train[:clean_n], y_train[:adv_n]], dim=0)

        print(f"[INFO] Training with {clean_n} clean + {adv_n} adversarial examples.")

    train_losses = []
    train_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        epoch_acc = 0

        for i in range(0, X_train.size(0), BATCH_SIZE):
            indices = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = X_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            with torch.no_grad():
                pred_labels = (outputs > 0.5).float()
                acc = (pred_labels == batch_y).float().mean().item()
                epoch_acc += acc

        epoch_loss /= (X_train.size(0) // BATCH_SIZE)
        epoch_acc /= (X_train.size(0) // BATCH_SIZE)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")


    # === Evaluate on Clean Test Set ===
    model.eval()
    with torch.no_grad():
        outputs = model(X_test).squeeze()
        preds = (outputs > 0.5).float()
        test_loss = criterion(outputs, y_test).item()
        test_acc = (preds == y_test).float().mean().item()

        print(f"[RESULT] Final Test Accuracy: {test_acc*100:.2f}% | Loss: {test_loss:.4f}")
        save_metrics(train_losses, train_accuracies, test_acc, test_loss)
        save_confusion_matrix(y_test.numpy(), preds.numpy())

    # === Evaluate on Adversarial Test Set ===
    if ATTACK_TYPE in ["fgsm", "bim"]:
        print("[INFO] Generating adversarial test set...")
        if ATTACK_TYPE == "fgsm":
            adv_X_test = fgsm_attack(model, X_test, y_test, EPSILON)
        else:
            adv_X_test = bim_attack(model, X_test, y_test, EPSILON, ALPHA, ITERATIONS)

        with torch.no_grad():
            adv_probs = model(adv_X_test).squeeze()
            adv_loss = criterion(adv_probs, y_test).item()
            save_adv_metrics(y_test, adv_probs, adv_loss)

    # === Save Weights ===
    tag = f"{ATTACK_TYPE}_{int(ADV_RATIO * 100)}%_{N_LAYERS}layers" if ATTACK_TYPE != "clean" else f"clean_{N_LAYERS}layers"
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, f"quantum_model_{tag}.pt"))
    print(f"[INFO] Saved weights to: {WEIGHT_DIR}\\quantum_model_{tag}.pt")

if __name__ == "__main__":
    train()
