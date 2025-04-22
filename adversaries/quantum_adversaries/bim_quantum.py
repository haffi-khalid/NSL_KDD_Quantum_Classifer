"""
BIM Attack (Iterative FGSM) for Quantum Classifier (PyTorch Version)

Performs multiple small FGSM steps to generate stronger adversarial examples.
"""

import torch
import torch.nn.functional as F

def bim_attack(model, X, y, epsilon=0.05, alpha=0.01, iterations=3):
    X_adv = X.clone().detach().requires_grad_(True)
    y = y.clone().detach()
    original = X.clone().detach()

    for _ in range(iterations):
        model.eval()
        outputs = model(X_adv).squeeze()
        loss = F.binary_cross_entropy(outputs, y)
        model.zero_grad()
        loss.backward()

        grad_sign = X_adv.grad.sign()
        X_adv = X_adv + alpha * grad_sign
        X_adv = torch.min(torch.max(X_adv, original - epsilon), original + epsilon)
        X_adv = torch.clamp(X_adv, 0, 1).detach().requires_grad_(True)

    return X_adv.detach()
