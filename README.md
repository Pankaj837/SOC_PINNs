# SOC-26: Physics-Informed Neural Networks

This repository documents my week-by-week progress on the SOC-26: Physics-Informed Neural Networks project. It contains concise weekly reports and solved notebooks for Weeks 1–4, along with key results and findings.

## Project Goal

Build a solid practical understanding of Physics-Informed Neural Networks (PINNs) — neural networks trained to satisfy differential equations as part of their loss function — progressing from automatic differentiation fundamentals to solving canonical PDEs and the benchmark Burgers' equation.

## Repo Layout

```
SOC-26-PINNs
├── README.md
├── Week1
│   ├── Week1.md
│   └── solutions
├── Week2
│   ├── Week2.md
│   └── solutions
├── Week3
│   ├── Week3.md
│   └── solutions
├── Week4
    ├── Week4.md
    └── solutions
```

Each `WeekX.md` contains:

- Short overview of the week
- Topics covered and why they matter
- Problems/assignments solved
- Key results and figures

The `solutions/` folder holds the Jupyter notebooks implementing the solved problems for that week.

---

## What to Expect

**Week 1 — PDEs, Neural Network Refresher & Automatic Differentiation**

- PDE classification: elliptic, parabolic, hyperbolic
- MLP architecture and forward pass in PyTorch
- Automatic differentiation using `torch.autograd.grad`
- Differentiating network outputs w.r.t. inputs (the key PINN skill)
- Verified derivatives against analytical functions (`sin(x)`, `cos(x)`, `-sin(x)`)

**Week 2 — The PINN: Architecture, Loss Function & Harmonic Oscillator**

- PINN components: network as surrogate solution, collocation points, loss construction
- PDE residual loss, initial condition loss, boundary condition loss
- Solved the damped harmonic oscillator ODE from scratch in PyTorch
- Frequency sweep experiment revealing spectral bias

**Week 3 — Canonical PDEs: Heat Equation & Wave Equation**

- Scaled from ODEs to PDEs (2D: space + time)
- Solved the 1D heat equation and 1D wave equation using DeepXDE
- Collocation sampling experiment: error vs number of collocation points
- Qualitative comparison between heat and wave equation solutions

**Week 4 — Burgers' Equation, Boundary Conditions & Classical Solver Comparison**

- Reproduced the benchmark PINN result from Raissi et al. 2019 (Burgers' equation) in pure PyTorch
- Implemented soft vs hard boundary condition enforcement
- Compared training loss curves and L2 errors for both approaches
- Written analysis: when to choose PINNs over classical FEM solvers

---

## Dependencies

```
torch
numpy
matplotlib
deepxde
scipy
```

---

## How to Review the Work

1. Open `WeekX/WeekX.md` to read the weekly summary and findings.
2. Open `WeekX/solutions/` for the Jupyter notebook implementing the solved problems.

---

## Status

This repository contains completed work for Weeks 1, 2, 3, and 4 of the SOC-26 project.
Submitted as Midterm Report.
