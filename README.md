# NSL-KDD Quantum & Classical Adversarial Robustness Project

This repository explores **adversarial machine learning** and the **robustness of quantum and classical (CNN) classifiers** using the NSL-KDD dataset — a widely used dataset for intrusion detection. It includes preprocessing pipelines, adversarial attack generation (FGSM, BIM), adversarial training, robustness evaluations, and transferability studies between quantum and classical models.

---

## 📂 Folder Structure

```
NSL_KDD_Quantum_Classifier/
│
├── preprocess_nslkdd_quantum.py       # Quantum preprocessing (MinMax + L2 + Padding)
├── preprocess_nslkdd_cnn.py           # CNN preprocessing (One-hot + MinMax)
│
├── train_quantum.py                   # Quantum model training (clean, FGSM, BIM)
├── train-cnn.py                       # CNN model training (clean, FGSM, BIM)
│
├── cnn.py                             # CNN architecture
├── quantum_model.py                   # Quantum classifier using PennyLane
│
├── fgsm_quantum.py                    # FGSM implementation for quantum model
├── bim_quantum.py                     # BIM implementation for quantum model
├── fgsm_cnn.py                        # FGSM for CNN
├── bim_cnn.py                         # BIM for CNN
│
├── evaluate_cnn_adversarial_robustness.py     # Evaluates clean/FGSM/BIM-trained CNNs against FGSM
├── quantum_adv_training_eval.py                # Evaluates clean/FGSM/BIM-trained Quantum models against FGSM
│
├── evaluate_fgsm_transfer_cnn_vs_quantum.py    # CNN-generated FGSM → evaluate on quantum model
├── visualise_cnn_vs_quantum_fgsm.py            # Plots transferability metrics for CNN→Quantum
│
├── Attacks_against_each_other/         # Transferability experiments: CNN↔Quantum
├── Quantum_vs_cnn_working/             # Reversal experiments from quantum to CNN inputs
├── pre-processed_data/                 # Preprocessed CSVs (CNN & Quantum)
├── train/weights/                      # Quantum model weights (.pt)
├── train-cnn/weights/                  # CNN model weights (.h5)
```

---

## 📌 Highlights of the Project

### ✅ Preprocessing
- **Quantum**: 
  - Min-Max scaling
  - L2 normalization
  - Power-of-2 padding
  - No one-hot encoding
- **CNN**:
  - Categorical → One-hot encoding (`protocol_type`, `service`, `flag`)
  - Min-Max scaling
  - No padding

### ✅ Model Architectures
- **CNN**: `cnn.py` uses a Conv1D architecture tailored for 119 features.
- **Quantum**: `quantum_model.py` builds a variational classifier using PennyLane and PyTorch.

---

## ⚔️ Adversarial Attacks

| Attack | Description | Supported On |
|--------|-------------|--------------|
| FGSM   | Fast Gradient Sign Method | CNN, Quantum |
| BIM    | Basic Iterative Method    | CNN, Quantum |

Scripts:
- `fgsm_cnn.py`, `bim_cnn.py`
- `fgsm_quantum.py`, `bim_quantum.py`

---

## 🧪 Robustness Evaluation

### Quantum Evaluation (`quantum_adv_training_eval.py`)
- Compares **clean**, **FGSM-trained**, and **BIM-trained** quantum models
- Generates:
  - Accuracy vs. Epsilon plot
  - Adversarial test loss vs. Epsilon
  - Saves confusion matrices & ROC curves

### CNN Evaluation (`evaluate_cnn_adversarial_robustness.py`)
- Same structure as quantum evaluation

---

## 🔁 Transferability Experiments

### CNN → Quantum
- `evaluate_fgsm_transfer_cnn_vs_quantum.py`: Generates FGSM attacks from CNN and evaluates on quantum.
- `visualise_cnn_vs_quantum_fgsm.py`: Plots combined performance.

### Quantum → CNN (Option A′)
- `Quantum_vs_cnn_working/quantum_vs_cnn_working.py`:
  - Uses a Nearest Neighbor Reversal to recover original 119 features from padded quantum inputs.
  - Evaluates CNN performance on quantum-generated FGSM data.

---

## 📊 Output Artifacts

Each major script saves:
- `.csv` metrics (accuracy, loss, per-epsilon breakdown)
- `accuracy_vs_epsilon.png`
- `loss_vs_epsilon.png`
- Confusion matrix heatmaps
- Recovered adversarial vectors (in applicable directories)

---

## 📚 References

- Goodfellow et al. (2015) — [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- Chakrabarti et al. (2020) — [Quantum Adversarial Machine Learning](https://arxiv.org/abs/2001.00030)
- NSL-KDD Dataset — [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)

---

## ⚙️ Setup Instructions

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision pennylane tensorflow scikit-learn pandas matplotlib seaborn liac-arff
```

### 2. Dataset

Place the raw NSL-KDD ARFF files here:
```
/archive/
  ├── KDDTrain+_20Percent.arff
  └── KDDTest+.arff
```

### 3. Preprocess

```bash
python preprocess_nslkdd_quantum.py
python preprocess_nslkdd_cnn.py
```

### 4. Train

```bash
python train_quantum.py
python train-cnn.py
```

### 5. Evaluate

```bash
python quantum_adv_training_eval.py
python evaluate_cnn_adversarial_robustness.py
```

---

## 👨‍💻 Author

**Muhammad Haffi Khalid**  
University of Birmingham — Final Year BSc Computer Science  
> Project: *"Effects of Quantum Embeddings on the Generalization of Adversarially Trained Quantum Classifiers"*

---

## 📎 License

This project is open-source for academic and non-commercial use.  
Feel free to cite this repo in your work or explore the robustness of adversarial machine learning!

---
