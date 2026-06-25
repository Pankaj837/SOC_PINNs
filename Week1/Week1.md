# Week 1 — PDEs, Neural Network Refresher & Automatic Differentiation

## Overview

Week 1 focused on building the foundational skills required for Physics-Informed Neural Networks. The core goal was to understand what PDEs are, refresh neural network intuition, and most importantly, learn how to differentiate a neural network with respect to its **inputs** using PyTorch's automatic differentiation. This one idea is what makes PINNs possible.

---

## Topics Covered

### PDE Classification

- Elliptic, parabolic, and hyperbolic PDEs
- Reading standard PDE notation: ∂u/∂t, ∂²u/∂x²
- Understanding what it means to "solve" a PDE (finding a function, not a number)

PDEs are equations involving derivatives of unknown functions. Unlike algebraic equations that solve for a number, PDEs solve for an entire function, e.g., the temperature distribution across a rod over time.

---

### Neural Network Refresher

- MLP architecture: input layer → hidden layers → output layer
- Forward pass: how data flows through the network
- Loss function and gradient descent
- `tanh` activations and why they matter for PINNs (smooth, infinitely differentiable)

---

### Automatic Differentiation

- Forward mode vs reverse mode (backpropagation)
- **Key insight:** `torch.autograd.grad` differentiates the network **output** w.r.t. the **input**, not the weights

This is the crucial distinction:

| Normal Backprop | PINN Autograd |
|---|---|
| d(Loss) / d(weights) | d(output) / d(input) |
| Used to update weights | Used to compute PDE residual |

```python
# This is the core of every PINN:
x = torch.tensor([1.0], requires_grad=True)
u = network(x)
du_dx = torch.autograd.grad(u, x, create_graph=True)[0]
d2u_dx2 = torch.autograd.grad(du_dx, x)[0]
```

---

### PyTorch Mechanics

- `requires_grad=True`: tells PyTorch to track operations on a tensor
- `create_graph=True`: keeps the computation graph alive for higher-order derivatives
- Why `create_graph=True` matters: without it, PyTorch discards the graph after the first derivative, making second derivatives impossible

---

## Assignment

### Task 1 — Build an MLP
Implemented a 3-layer MLP in pure PyTorch (no PINN libraries) that takes scalar `x` as input and returns scalar `u(x)` using `tanh` activations.

### Task 2 — Compute Derivatives using Autograd
Used `torch.autograd.grad` to compute `du/dx` and `d²u/dx²` for the network output.

### Task 3 — Verify Against Analytical Derivatives
Trained the network to approximate `u(x) = sin(x)` and verified:
- `du/dx ≈ cos(x)`
- `d²u/dx² ≈ −sin(x)`

Evaluated at 100 random points in [−π, π].

### Task 4 — Plot
Produced a single figure with 3 subplots comparing autograd vs analytical for `u`, `du/dx`, and `d²u/dx²`.

---

## Key Results

| Derivative | Max Error |
|---|---|
| u(x) | ~0.005 |
| du/dx | ~0.07 |
| d²u/dx² | ~0.35 |

Errors increase for higher-order derivatives, a known effect of **spectral bias** (networks learn low frequencies first).

The plots showed near-perfect agreement for `u` and `du/dx`, with slight deviation at the edges for `d²u/dx²`.

---

## Reflection Questions

**1. What is the difference between differentiating w.r.t. weights vs inputs?**
Backprop differentiates the loss w.r.t. weights to update the network. PINNs differentiate the network output w.r.t. inputs (like `x` or `t`) to compute PDE residuals. These are completely different operations.

**2. Why does `create_graph=True` matter for second-order derivatives?**
After computing the first derivative, PyTorch discards the computation graph by default. `create_graph=True` keeps it alive so we can differentiate again to get the second derivative.

**3. What is spectral bias?**
Neural networks naturally learn low-frequency (slowly changing) patterns before high-frequency ones. This means higher-order derivatives, which amplify high-frequency content, are harder to learn accurately. This will become critical in Week 2's frequency sweep experiment.

---

## Conclusion & Next Steps

Week 1 successfully established the core PINN skill: using `torch.autograd.grad` to compute exact derivatives of a neural network with respect to its inputs. This is the exact tool used in every PINN to compute PDE residuals.

In **Week 2**, these tools will be applied to build a complete PINN on the damped harmonic oscillator.
