# SOC-26: Physics-Informed Neural Networks
A from-scratch implementation of Physics-Informed Neural Networks in PyTorch, progressing from automatic differentiation fundamentals to solving canonical PDEs, mitigating spectral bias, and finally recovering unknown physical parameters from sparse, noisy sensor data on a nonlinear PDE.


## Project Goal
Build a solid practical understanding of Physics-Informed Neural Networks (PINNs), neural networks trained to satisfy differential equations as part of their loss function, progressing from automatic differentiation fundamentals to solving canonical PDEs, the benchmark Burgers' equation, and inverse problems where an unknown physical parameter is recovered purely from sparse sensor data.

## Repo Layout

```
SOC-26-PINNs
├── README.md
├── Week1
│   ├── Week1.md
│   └── Solutions
├── Week2
│   ├── Week2.md
│   └── solutions
├── Week3
│   ├── Week3.md
│   └── solutions
├── Week4
│   ├── Week4.md
│   └── solutions
├── Week5
│   ├── Week5.md
│   └── solutions
├── Week6
│   ├── Week6.md
│   └── solutions
└── Week7&8
    ├── Week7&8.md
    └── solutions
```

Each `WeekX.md` contains:

- Short overview of the week
- Topics covered and why they matter
- Problems/assignments solved
- Key results and figures

The `solutions/` folder holds the Jupyter notebooks implementing the solved problems for that week. ( Link if uploaded file doesn't work : https://drive.google.com/drive/folders/10SzPzSxJl2s6ZP1kIzczt-fP4zIfiqas?usp=sharing ) 

Each `WeekX.md` documents that module: the problem, the approach, the results, and what it feeds into next. The `solutions/` folder holds the runnable notebook(s).

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

**Week 5 — Spectral Bias Mitigation: Fourier Features & Gradient-Norm Balancing**

- Random Fourier feature embeddings (σ = 1 and σ = 10) to give the network direct access to high-frequency content
- Gradient-norm adaptive loss balancing across PDE/IC/BC terms
- Comparative experiment: vanilla vs Fourier vs gradient-balanced, on the Week 4 Burgers' setup

**Week 6 — Inverse Problems: Parameter Recovery in a Diffusion-Reaction Equation**

- Treated an unknown reaction rate `k` as a trainable network parameter
- Recovered `k` from 100 sparse, noisy observations, jointly with the forward solution
- Noise sensitivity sweep (σ ∈ {0%, 1%, 5%})
- Track proposal — selected Track A: Inverse Problems & Parameter Identification

**Week 7-8 — Track A: Burgers' Viscosity Recovery**

- Extended the inverse-problem approach to the nonlinear Burgers' equation
- Recovered the unknown viscosity `ν` from 200 sparse, noisy observations
- Noise sensitivity sweep (σ ∈ {0%, 1%, 5%, 10%})
- Benchmarked against a classical scipy curve-fitting baseline

## Why this progression

Each module builds a capability the next one needs:

1. **Differentiate a network w.r.t. its inputs** (Week 1) → the one trick that makes PINNs possible
2. **Assemble that into a full physics-constrained loss** (Week 2) → and immediately hit spectral bias, a real limitation
3. **Scale to full PDEs in space+time** (Week 3)
4. **Handle nonlinearity and boundary conditions properly** (Week 4) → using the field's standard benchmark, Burgers' equation
5. **Fix spectral bias** (Week 5) → so the network can actually resolve sharp features
6. **Use the physics constraint to recover unknowns, not just solve forward problems** (Week 6) → this is where PINNs do something classical solvers can't
7. **Push that to a genuinely hard nonlinear inverse problem** (Week 7-8) → and prove it against a classical baseline


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
2. Open `WeekX/solutions/` for the Jupyter notebook implementing the solved problems ( Every notebook is self-contained, open it in Colab/Jupyter and run all cells top to bottom, no shared state between modules )

---

## Status

This repository contains completed work for Weeks 1-8 complete including the Track A specialization (Inverse Problems & Parameter Identification) of the SOC-26 project.
Submitted as Endterm Report.
