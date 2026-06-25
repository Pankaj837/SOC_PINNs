# Week 2 — The PINN: Architecture, Loss Function & Harmonic Oscillator

## Overview

Week 2 focused on building a complete, working Physics-Informed Neural Network from scratch. Using the skills from Week 1 (autograd w.r.t. inputs), a full PINN was constructed to solve the **damped harmonic oscillator** ODE. The week concluded with a frequency sweep experiment that directly demonstrated **spectral bias** in PINNs.

---

## Topics Covered

### PINN Components

A PINN is a neural network trained to satisfy a differential equation. It has three key components:

1. **The network:** A standard MLP that acts as the surrogate solution `u_θ(t)`
2. **Collocation points:** Random time points where the PDE is enforced (replacing the mesh in classical methods)
3. **Loss function:** Sum of physics residual, initial condition, and boundary condition losses

---

### Loss Function Construction

The total loss has three parts:

```
L_total = L_pde + λ_ic × L_ic + λ_bc × L_bc
```

- **L_pde:** How much the network violates the differential equation at collocation points
- **L_ic:** How wrong the network is at the initial condition (t=0)
- **L_bc:** How wrong the network is at the boundaries
- **λ weights:** Used to emphasize certain conditions (e.g., λ_ic = 100)

There are no labels. The physics supervises the training.

---

### Soft Constraint Enforcement

Initial conditions and boundary conditions are enforced as weighted penalty terms in the loss, this is called **soft enforcement**. The network is penalized for violating them but not mathematically prevented.

---

### The Damped Harmonic Oscillator

The equation solved:

```
m ẍ + μ ẋ + kx = 0,   x(0) = 1,  ẋ(0) = 0
m = 1,  μ = 0.1
```

In plain terms: a mass on a spring with friction. It starts stretched (x=1) with zero velocity and oscillates while losing energy over time.

The PINN learns `x(t)`, the position at every time point, without being given labeled position data. Only the equation and initial conditions are used.

---

### Spectral Bias

Neural networks naturally learn **low-frequency (slowly changing) patterns** much faster than **high-frequency (rapidly oscillating) patterns**. This is called spectral bias.

**Why it happens:** When learning high-frequency patterns, gradient updates are chaotic, improving one oscillation disturbs another. The network cannot settle. For slow patterns, gradients are smooth and consistent.

---

## Assignment

### Task 1 — Frequency Sweep

Trained the PINN for each value of ω₀ ∈ {1, 5, 10, 15, 20} and recorded:
- Final training loss
- Final L² error between predicted and exact solution

### Task 2 — Written Reflection

Answered two questions about spectral bias and proposed a hypothetical fix.

---

## Key Results

| ω₀ | L2 Error |
|---|---|
| 1 | 0.027 ✅ |
| 5 | 0.548 ❌ |
| 10 | 0.554 ❌ |
| 15 | 0.554 ❌ |
| 20 | 0.555 ❌ |

The PINN learned `ω₀=1` (slow oscillation) almost perfectly. For `ω₀=5` and above, the error plateaued near 0.55, the network gave up and predicted approximately zero everywhere.

The worst-case plot for `ω₀=20` showed the PINN predicting a flat line while the exact solution oscillated 20 times, a clear failure due to spectral bias.

---

## Written Reflection

### Q1: What is spectral bias and what does the experiment demonstrate?

Spectral bias is the tendency of neural networks to learn low-frequency patterns much faster and more accurately than high-frequency patterns. The frequency sweep experiment demonstrated this clearly, at ω₀=1 the PINN achieved an L2 error of just 0.027, but for ω₀=5 and above the error jumped to ~0.55 and stayed flat. The worst-case plot for ω₀=20 shows the network predicting nearly zero everywhere instead of the rapidly oscillating exact solution.

### Q2: What would you try to fix the high-frequency failure?

One approach is **Fourier feature embeddings**, instead of feeding raw time `t` into the network, first transform it into sine and cosine functions of many different frequencies. This gives the network explicit access to high-frequency information from the start, rather than having to discover it during training. This is planned for Week 5 of the project.

---

## Conclusion & Next Steps

Week 2 demonstrated the full PINN workflow: define the equation, enforce initial conditions via loss, train the network, and evaluate against the exact solution. The frequency sweep gave a concrete, visual demonstration of spectral bias, a real limitation of standard PINNs.

In **Week 3**, the focus shifts from ODEs to PDEs, using the DeepXDE library to solve the heat and wave equations.
