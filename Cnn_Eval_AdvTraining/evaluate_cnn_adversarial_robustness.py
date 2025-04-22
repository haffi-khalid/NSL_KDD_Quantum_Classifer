import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy

# === Setup ===
BASE = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier"
SAVE_DIR = os.path.join(BASE, "cnn_adversarial_evaluation")
os.makedirs(SAVE_DIR, exist_ok=True)

# === Paths ===
paths = {
    "clean": os.path.join(BASE, "train-cnn", "weights", "cnn_model_clean.h5"),
    "fgsm": os.path.join(BASE, "train-cnn", "weights", "cnn_model_fgsm_100%_final.h5"),
    "bim": os.path.join(BASE, "train-cnn", "weights", "cnn_model_bim_100%_final.h5"),
    "test_csv": os.path.join(BASE, "pre-processed_data", "cnn_test_preprocessed.csv"),
}

# === FGSM Attack ===
def fgsm_attack(model, x, y, eps):
    loss_fn = BinaryCrossentropy()
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        prediction = model(x_tensor, training=False)
        loss = loss_fn(y_tensor, prediction)

    gradients = tape.gradient(loss, x_tensor)
    signed_grad = tf.sign(gradients)
    x_adv = x_tensor + eps * signed_grad
    return tf.clip_by_value(x_adv, 0, 1).numpy()

# === Load Models ===
print("[INFO] Loading CNN models...")
cnn_clean = load_model(paths["clean"])
cnn_fgsm = load_model(paths["fgsm"])
cnn_bim = load_model(paths["bim"])

# === Load Data ===
print("[INFO] Loading CNN test data...")
df = pd.read_csv(paths["test_csv"])
X = df.drop(columns=["class"]).values.astype(np.float32)
y = df["class"].values.astype(np.float32)

# === Evaluate ===
results = []
epsilons = np.arange(0, 1.05, 0.05)

for eps in epsilons:
    print(f"[*] Epsilon = {eps:.2f}")
    X_adv = fgsm_attack(cnn_clean, X, y, eps)

    for model_name, model in [("clean", cnn_clean), ("fgsm", cnn_fgsm), ("bim", cnn_bim)]:
        preds = model.predict(X_adv, verbose=0).squeeze()
        acc = np.mean((preds > 0.5).astype(np.int32) == y.astype(np.int32))
        loss = BinaryCrossentropy()(y, preds).numpy()

        results.append({
            "epsilon": eps,
            "model": model_name,
            "accuracy": acc,
            "loss": loss
        })

# === Save Results ===
df_results = pd.DataFrame(results)
csv_path = os.path.join(SAVE_DIR, "cnn_fgsm_evaluation.csv")
df_results.to_csv(csv_path, index=False)
print(f"[INFO] Saved results to {csv_path}")

# === Plot Accuracy and Loss ===
plt.figure(figsize=(10, 6))
for model_name in ["clean", "fgsm", "bim"]:
    subset = df_results[df_results["model"] == model_name]
    plt.plot(subset["epsilon"], subset["accuracy"], label=f"{model_name.upper()}")

plt.title("CNN Accuracy vs Epsilon (FGSM Attack)")
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "cnn_accuracy_vs_epsilon.png"))
plt.close()

plt.figure(figsize=(10, 6))
for model_name in ["clean", "fgsm", "bim"]:
    subset = df_results[df_results["model"] == model_name]
    plt.plot(subset["epsilon"], subset["loss"], label=f"{model_name.upper()}")

plt.title("CNN Adversarial Loss vs Epsilon (FGSM Attack)")
plt.xlabel("Epsilon")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(SAVE_DIR, "cnn_loss_vs_epsilon.png"))
plt.close()

print("[OK] Evaluation complete and plots saved.")
