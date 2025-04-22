"""
FGSM Attack for Quantum Classifier (PyTorch Version)

Generates adversarial examples by computing the sign of the gradient of the loss w.r.t the input.
"""

import torch
import torch.nn.functional as F

def fgsm_attack(model, X, y, epsilon=0.1):
    X_adv = X.clone().detach().requires_grad_(True)
    y = y.clone().detach()

    model.eval()
    outputs = model(X_adv).squeeze()
    loss = F.binary_cross_entropy(outputs, y)

    loss.backward()
    signed_grad = X_adv.grad.sign()
    adv_X = X_adv + epsilon * signed_grad
    adv_X = torch.clamp(adv_X, 0, 1)

    return adv_X.detach()
