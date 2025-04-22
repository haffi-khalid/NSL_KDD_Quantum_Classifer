"""
Quantum Variational Classifier using PyTorch (TorchLayer)

This module defines a hardware-efficient quantum classifier architecture:

- Uses amplitude embedding to encode classical input into quantum state space.
- Applies L layers of parameterized RY rotations followed by CNOT entanglement gates.
- Measures a single output qubit (PauliZ expectation).
- Wraps the PennyLane QNode using TorchLayer for native PyTorch integration.
- Configurable: number of qubits, layers, and output qubit.
- Can be used with PyTorch optimizers, loss functions, and training loops.

Author: Muhammad Haffi Khalid
Final Year Project: Effects of Quantum Embeddings on the Generalization of Adversarially Trained Quantum Classifiers
"""

import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn

# === CONFIG ===
n_features = 64                             # Input feature size
n_input_qubits = int(np.log2(n_features))   # Qubits needed for amplitude embedding
n_qubits = n_input_qubits + 1               # Add 1 output qubit
output_wire = n_qubits - 1                  # Last qubit used for measurement

def make_quantum_circuit(n_layers):
    """
    Returns a PennyLane QNode with L variational layers, each with:
    - RY rotations on all qubits
    - CNOT entanglement in a linear chain
    Measures the PauliZ expectation of the output qubit.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.AmplitudeEmbedding(inputs, wires=list(range(n_input_qubits)), normalize=True)

        for l in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[l, i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

        return qml.expval(qml.PauliZ(output_wire))

    return circuit

def create_quantum_model(n_layers=3):
    """
    Builds and returns a PyTorch model wrapped around a quantum layer.
    """
    weight_shapes = {"weights": (n_layers, n_qubits)}
    qnode = make_quantum_circuit(n_layers)
    quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    # Combine into PyTorch Sequential model
    model = nn.Sequential(
        quantum_layer,
        nn.Sigmoid()  # Convert expectation [-1, 1] to probability [0, 1]
    )

    return model
