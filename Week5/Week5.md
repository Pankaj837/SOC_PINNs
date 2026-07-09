# Week 5 — Spectral Bias Mitigation: Fourier Features & Gradient-Norm Balancing

## Overview

Week 4 exposed a real limitation: the PINN's highest error concentrated exactly at the Burgers' shock (`t=1, x=0`), a symptom of **spectral bias**, networks learn low-frequency content first and struggle with sharp, high-frequency features. Week 5 addresses this directly with two independent fixes, built on the exact same Burgers' setup from Week 4, and compares them head-to-head.

---

## Topics Covered

### Random Fourier Feature Embeddings

Instead of feeding raw `(x, t)` into the network, first project it through fixed random sine/cosine features:

```python
B = torch.randn(d_model, 2) * sigma   # fixed, not trained
proj = xt @ B.T
embedded = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
```

This gives the network direct access to high-frequency basis functions from the start, rather than having to discover them slowly through training. `σ` controls the embedding's frequency range, higher `σ` biases toward sharper, higher-frequency features.

### Gradient-Norm Adaptive Loss Balancing

In a standard PINN loss (`L_pde + λ_ic·L_ic + λ_bc·L_bc`), the PDE term's gradient (involving second derivatives) is typically much larger than the IC/BC terms', so the optimizer effectively ignores IC/BC. Gradient-norm balancing rescales each loss term's weight, at every step, so their gradient contributions stay comparable:

```python
lambda_ic = mean_gradient_norm / (gn_ic + eps)
lambda_bc = mean_gradient_norm / (gn_bc + eps)
```

---

## Assignment

Setup carried over unchanged from Week 4 Part A (10,000 collocation points, 4×100 tanh network, Adam, 10,000 epochs).

### Task 1 — Fourier Feature Embedding
Trained with `σ = 1` and `σ = 10` to compare how the embedding's frequency scale affects convergence.

### Task 2 — Gradient-Norm Loss Balancing
Implemented adaptive reweighting of `L_pde`, `L_ic`, `L_bc` based on their gradient norms.

### Task 3 — Comparative Experiment
Trained three variants under identical conditions: vanilla PINN, +Fourier features (σ=10), +gradient-norm balancing. Compared loss curves and final L² error.

### Task 4 — Analysis
Written reflection on which fix helped most, and why.

---

## Key Results

| Method | L2 Error |
|---|---|
| Vanilla PINN | 0.0289 |
| + Fourier Features (σ=1) | 0.0235 |
| + Fourier Features (σ=10) | 0.0205  |
| + Gradient-Norm Balancing | 0.0277 |

Fourier features gave the largest improvement, cutting L2 error by ~29% relative to vanilla, with `σ=10` outperforming `σ=1` : the higher-frequency embedding matches the shock's sharpness better. Gradient-norm balancing gave a smaller, more marginal improvement.

---

## Analysis

### Which fix helped most?

Fourier features (σ=10) helped most for the shock region. By embedding `(x,t)` into high-frequency sine/cosine features, the network gained access to sharp-gradient basis functions from the start of training, rather than having to discover them slowly. Gradient-norm balancing improved training stability but had a smaller effect on the shock itself, since it doesn't change what frequencies the network can represent, it only rebalances competing loss terms.

### Gradient norm observation

In the vanilla PINN, `L_pde` has the largest gradient norm (second-order derivatives amplify gradients), which drowns out `L_ic` and `L_bc`. This is consistent with why hard BCs (Week 4) helped, they remove `L_bc` from the competition entirely rather than trying to rebalance it.

---

## Conclusion & Next Steps

Week 5 closed the accuracy gap identified in Week 4: Fourier feature embeddings are the more effective fix for shock-region spectral bias, with gradient-norm balancing as a complementary but secondary improvement.

In **Week 6**, the focus shifts from forward problems to **inverse problems**, using the physics constraint to recover an unknown PDE parameter directly from data.
