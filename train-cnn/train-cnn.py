
# train_cnn.py


# Trains a convolutional neural network (CNN) on the NSL-KDD data (preprocessed for CNN) and saves outputs.
# This updated version includes:
#   - Loading preprocessed training and test CSV files.
#   - Building the CNN model (from cnn.py), which now includes BatchNormalization and Dropout.
#   - Training the model while logging per-epoch metrics.
#   - Saving the final model weights to:
#        C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train-cnn\weights\cnn_model.h5
#   - Saving the per-epoch training and testing metrics as four separate CSV files:
#        - train_loss_4layers.csv
#        - train_accuracy_4layers.csv
#        - test_loss_4layers.csv
#        - test_accuracy_4layers.csv
#      in the folder:
#        C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\cnn-gen_data\clean
#   - Saving the model summary (with parameter counts) as a text file for your thesis documentation.


#  Also trains a 1D Convolutional Neural Network (CNN) on the NSL-KDD dataset with optional adversarial training.

#   This version supports:
# - Standard training on clean data
# - Adversarial training using FGSM or BIM attacks (with a tunable ratio)
# - Evaluation of adversarial robustness on test data
# - Saving:
#     - Model weights
#     - Per-epoch training and test metrics
#     - Model architecture summary
#     - Confusion matrix image
#     - Adversarial test accuracy and loss (if FGSM or BIM is used)


# Author: Muhammad Haffi Khalid
# Date: 13/04/2025

import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import csv
import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from model.cnn import create_cnn_model
from adversaries.cnn_adversaries.fgsm_cnn import generate_adversarial_examples as fgsm_attack
from adversaries.cnn_adversaries.bim_cnn import generate_adversarial_examples as bim_attack

# -------------------------------
# 1. Configuration: Parameters & Paths
# -------------------------------
attack_type = "bim"  # Options: "clean", "fgsm", "bim"
adv_ratio = 1.0       # Ratio of adversarial samples for training if attack is enabled
EPOCHS = 200
BATCH_SIZE = 128
EPSILON = 0.1
ALPHA = 0.01
ITERATIONS = 3

TRAIN_CSV = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\cnn_train_preprocessed.csv"
TEST_CSV = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\pre-processed_data\cnn_test_preprocessed.csv"

WEIGHTS_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\train-cnn\weights"
os.makedirs(WEIGHTS_DIR, exist_ok=True)
# Dynamically name the model file based on attack type and adv ratio
if attack_type == "clean":
    weight_filename = "cnn_model_clean2.h5"
else:
    ratio_percent = int(adv_ratio * 100)
    weight_filename = f"cnn_model_{attack_type}_{ratio_percent}%_final.h5"

WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, weight_filename)

# Choose data directory based on attack type
BASE_GEN_DIR = r"C:\Users\haffi\OneDrive - University of Birmingham\Documents\Desktop\NSL-KDD Quantum Classifier\cnn-gen_data"
GEN_DATA_DIR = os.path.join(BASE_GEN_DIR, attack_type)
os.makedirs(GEN_DATA_DIR, exist_ok=True)

# Output files
TRAIN_LOSS_FILE = os.path.join(GEN_DATA_DIR, "train_loss_4layers.csv")
TRAIN_ACC_FILE  = os.path.join(GEN_DATA_DIR, "train_accuracy_4layers.csv")
TEST_LOSS_FILE  = os.path.join(GEN_DATA_DIR, "test_loss_4layers.csv")
TEST_ACC_FILE   = os.path.join(GEN_DATA_DIR, "test_accuracy_4layers.csv")
MODEL_SUMMARY_FILE = os.path.join(GEN_DATA_DIR, "cnn_model_summary.txt")
CONF_MATRIX_PNG = os.path.join(GEN_DATA_DIR, "confusion_matrix.png")
ADV_TEST_ACC_FILE = os.path.join(GEN_DATA_DIR, "adv_test_accuracy.csv")
ADV_TEST_LOSS_FILE = os.path.join(GEN_DATA_DIR, "adv_test_loss.csv")

# -------------------------------
# 2. Utility Functions
# -------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["class"]).to_numpy()
    y = df["class"].to_numpy().astype(np.float32)
    return X, y

def save_model_summary(model, output_file):
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(stream.getvalue())
    print(f"[INFO] Saved model summary to {output_file}")

def save_metrics_separately(history):
    train_losses = history.history.get('loss', [])
    train_accuracies = history.history.get('accuracy', [])
    test_losses = history.history.get('val_loss', [])
    test_accuracies = history.history.get('val_accuracy', [])

    for metric, filename in [
        (train_losses, TRAIN_LOSS_FILE),
        (train_accuracies, TRAIN_ACC_FILE),
        (test_losses, TEST_LOSS_FILE),
        (test_accuracies, TEST_ACC_FILE),
    ]:
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metric)
        print(f"[INFO] Saved metric data to {filename}")

def save_confusion_matrix(y_true, y_pred, file_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(file_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {file_path}")

def save_adv_metrics(loss, acc):
    pd.DataFrame([acc]).to_csv(ADV_TEST_ACC_FILE, index=False)
    pd.DataFrame([loss]).to_csv(ADV_TEST_LOSS_FILE, index=False)
    print(f"[INFO] Saved adversarial test metrics.")

# Load clean model for adversarial generation
CLEAN_MODEL_PATH = os.path.join(WEIGHTS_DIR, "cnn_model_clean.h5")
if os.path.exists(CLEAN_MODEL_PATH):
    clean_model = keras.models.load_model(CLEAN_MODEL_PATH)
    print(f"[INFO] Loaded clean model from {CLEAN_MODEL_PATH} for adversarial sample generation.")
else:
    clean_model = None
    print("[WARN] Clean model not found. FGSM/BIM samples will be generated from untrained model.")


# -------------------------------
# 3. Main Training Process
# -------------------------------
def main():
    print("[INFO] Loading training data...")
    X_train, y_train = load_data(TRAIN_CSV)

    print("[INFO] Loading test data...")
    X_test, y_test = load_data(TEST_CSV)

    input_dim = X_train.shape[1]
    model = create_cnn_model(input_dim)
    save_model_summary(model, MODEL_SUMMARY_FILE)

    # Load clean model for generating adversaries
    clean_model_path = os.path.join(WEIGHTS_DIR, "cnn_model_clean.h5")
    if os.path.exists(clean_model_path):
        clean_model = keras.models.load_model(clean_model_path)
        print("[INFO] Loaded clean model for adversarial generation.")
    else:
        clean_model = model
        print("[WARN] Clean model not found. Using untrained model for adversaries.")

    # Generate adversarial training data
    if attack_type == "fgsm":
        adv_X = fgsm_attack(clean_model, X_train, y_train, EPSILON)
    elif attack_type == "bim":
        adv_X = bim_attack(clean_model, X_train, y_train, EPSILON, ALPHA, ITERATIONS)
    else:
        adv_X = None

    if adv_X is not None:
        num_adv = int(len(X_train) * adv_ratio)
        X_train = np.concatenate([X_train[:len(X_train) - num_adv], adv_X[:num_adv]])
        y_train = np.concatenate([y_train[:len(y_train) - num_adv], y_train[:num_adv]])
        print(f"[INFO] Training with {num_adv} adversarial examples and {len(X_train) - num_adv} clean examples.")

    print("[INFO] Starting CNN training...")
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = dict(enumerate(class_weights))

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
        verbose=2
    )

    model.save(WEIGHTS_FILE)
    save_metrics_separately(history)

    y_pred = (model.predict(X_test) > 0.5).astype(int)
    save_confusion_matrix(y_test, y_pred, CONF_MATRIX_PNG)

    if attack_type in ["fgsm", "bim"]:
        print("[INFO] Generating adversarial test set for evaluation...")
        if attack_type == "fgsm":
            adv_X_test = fgsm_attack(clean_model, X_test, y_test, EPSILON)
        else:
            adv_X_test = bim_attack(clean_model, X_test, y_test, EPSILON, ALPHA, ITERATIONS)
        
        max_diff = np.max(np.abs(X_test - adv_X_test))
        print(f"[DEBUG] Max pixel diff between clean and adversarial test examples: {max_diff:.4f}")


        # 🔍 Debug: print predictions
        preds = (model.predict(adv_X_test[:5]) > 0.5).astype(int).flatten()
        print("[DEBUG] First 5 adversarial predictions vs. true labels:")
        for i in range(5):
            print(f"Sample {i+1}: Prediction = {preds[i]}, True Label = {int(y_test[i])}")

        adv_loss, adv_acc = model.evaluate(adv_X_test, y_test, verbose=0)
        save_adv_metrics(adv_loss, adv_acc)
        print(f"[RESULT] Adversarial Test Accuracy: {adv_acc*100:.2f}% - Loss: {adv_loss:.4f}")
    else:
        print("[INFO] No adversarial evaluation performed for 'clean' training.")

    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Final Training Accuracy: {train_acc*100:.2f}% - Loss: {train_loss:.4f}")
    print(f"[RESULT] Final Testing Accuracy: {test_acc*100:.2f}% - Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
