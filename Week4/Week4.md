# Week 4 — Burgers' Equation, Boundary Conditions & Classical Solver Comparison

## Overview

Week 4 focused on reproducing the benchmark PINN result from the founding paper (Raissi et al. 2019) using pure PyTorch, no libraries. The Burgers' equation was solved using both **soft** and **hard** boundary condition enforcement, and an honest comparison between PINNs and classical Finite Element Methods (FEM) was written based on Grossmann et al. 2023.

---

## Topics Covered

### Burgers' Equation

The standard PINN benchmark from Raissi et al. 2019:

```
uₜ + u uₓ = (ν/π) uₓₓ,   x ∈ [−1, 1],  t ∈ [0, 1]

Boundary:  u(−1, t) = 0,  u(1, t) = 0
Initial:   u(x, 0) = −sin(πx)
ν = 0.01/π
```

**Why Burgers' equation is the PINN benchmark:**
- It is **nonlinear**, contains the `u × uₓ` term
- It develops a **near-discontinuous shock** at `t=1, x=0`
- It has a known analytical solution (via Hopf-Cole transform)

This makes it a good stress test for PINNs.

---

### Soft vs Hard Boundary Conditions

**Soft BCs (what we used in Weeks 2–3):**
```python
# Add a boundary loss term to the total loss
u_bc = model(x_boundary)
L_bc = ((u_bc - 0)**2).mean()
L_total = L_pde + lambda_bc * L_bc
```
The network is penalized for violating boundaries but not prevented from doing so.

**Hard BCs (trial function approach):**
```python
# Multiply network output by a function that is zero on the boundary
# For u(±1, t) = 0: φ(x) = (1 − x²)
def u_hard(model, x, t):
    raw = model(x, t)
    return (1 - x**2) * raw   # automatically zero at x = ±1
```
The network **cannot** violate the boundary condition, it is mathematically impossible. No boundary loss term is needed.

**Distance Function Approach (Sukumar & Srivastava 2021):**

The trial function `φ(x) = (1 − x²)` is essentially a distance function, it equals zero exactly on the boundary and is positive inside.

For complex 2D/3D domains, distance functions `φ(x, y)` can be constructed from CAD geometry using R-functions. The key formula remains:

```
u(x, t) = φ(x) · NN(x, t)
```

This guarantees exact BC satisfaction without any penalty term, simplifying training and removing the need to tune BC loss weights.

---

## Assignment

### Part A — Burgers' PINN from Scratch (Pure PyTorch)

**Network:** 4-layer MLP, 100 neurons per layer, `tanh` activation  
**Collocation points:** 10,000 interior points (random)  
**Boundary points:** 200  
**Initial condition points:** 100  
**Optimizer:** Adam, 10,000 epochs

**Deliverables:**
1. Plot of `u(x, t)` predicted by the PINN
2. Plot of absolute error `|u_pinn − u_exact|`
3. Training loss curve

---

### Part B — Hard BCs

Modified the Part A implementation to use the trial function `φ(x) = (1 − x²)`.

**Deliverables:**
1. Training loss curves: soft vs hard BCs on the same axes
2. Final L² error for both methods

---

### Part C — Written Analysis

Based on Grossmann et al. 2023 ("Can PINNs beat FEM?"), answered: when would you choose a PINN over a classical FEM solver?

---

## Key Results

### Part A — Burgers' PINN

- PINN captured the main shock structure in Burgers' equation
- Highest error concentrated near `t=1, x=0`, exactly where the shock develops
- This is expected: PINNs struggle with near-discontinuous solutions

### Part B — Soft vs Hard BCs

**L² Error Comparison:**

| Method | L2 Error |
|---|---|
| Soft BCs | 0.0721 |
| Hard BCs | 0.0235 |

**Hard BCs performed better** because the network is mathematically forced to satisfy `u(±1,t)=0`, removing one source of loss conflict and simplifying the optimization landscape.

**However**, on geometries where constructing a closed-form distance function is difficult (e.g. irregular 3D domains), soft BCs are more practical.

---

## Part C — Written Analysis: PINNs vs FEM

Based on Grossmann et al. 2023, PINNs rarely outperform FEM for standard forward problems. FEM solves smooth PDEs in milliseconds with guaranteed convergence. PINNs are slower, less accurate on shocks, and require careful tuning. However, PINNs have genuine advantages in specific scenarios:

**Scenario 1 — Inverse Problems:**
When you have sparse sensor measurements and want to recover an unknown PDE parameter (e.g. viscosity ν in Burgers' equation), PINNs handle this naturally by adding the unknown as a trainable parameter. FEM cannot do this without a separate expensive optimisation loop.

**Scenario 2 — High-Dimensional PDEs:**
FEM requires a mesh. In 10+ dimensions, meshing is computationally impossible (curse of dimensionality). PINNs sample random collocation points and scale to high dimensions. Example: option pricing PDEs in finance (Black-Scholes in 100 dimensions).

**Scenario 3 — Data Assimilation:**
When you have noisy experimental data AND a known PDE, PINNs can fuse both into one training loss. Example: reconstructing a full velocity field from a few PIV sensor readings in fluid mechanics. FEM has no native way to incorporate scattered data.

**Honest Limitations:**
- PINNs struggle with sharp shocks (as seen in this Burgers' result, error peaks at t=1)
- Training is slow, sensitive to hyperparameters, and not guaranteed to converge
- For any smooth forward PDE on a simple domain, FEM is faster and more accurate

## Conclusion & Next Steps

Week 4 established that **the boundary condition enforcement strategy has a measurable effect on accuracy** — hard BCs cut L² error roughly 3x by removing a competing loss term. It also produced the field's standard benchmark result (Burgers' shock capture) as a reference point for everything that follows.

In **Week 5**, the shock-region accuracy problem observed here is addressed directly using Fourier feature embeddings and gradient-norm loss balancing.

---

