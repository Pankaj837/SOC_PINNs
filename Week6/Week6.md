# Week 6 — Inverse Problems: Parameter Recovery in a Diffusion-Reaction Equation

## Overview

Weeks 1–5 all solved **forward** problems: given a known PDE and its parameters, find the solution `u(x,t)`. Week 6 flips this around, the PDE's form is known, but one physical parameter is not. This is an **inverse problem**, and it's where PINNs do something classical solvers genuinely can't: treat the unknown parameter as just another trainable variable, optimized jointly with the network weights, using nothing but sparse, noisy observations plus the physics itself.

---

## Topics Covered

### The Diffusion-Reaction Equation

- uₜ = D · uₓₓ − k · u,   x ∈ [0,1],  t ∈ [0,1]
- D = 0.001 (known)
- k = unknown (true value: 0.01)

**Physics:** a quantity diffusing through space while simultaneously decaying at rate `k`. `D` is known; `k` is what we're trying to recover.

### Treating a Physical Parameter as Trainable

```python
k = torch.nn.Parameter(torch.tensor([0.5]))  # initial guess, wrong on purpose
optimizer = torch.optim.Adam(list(model.parameters()) + [k], lr=1e-3)
```

`k` sits in the same optimizer as the network weights. The PDE residual loss `(uₜ − D·uₓₓ + k·u)²` only goes to zero if `k` is correct **and** the network correctly represents the solution, so minimizing the total loss (physics + sparse data + IC + BC) simultaneously fits the solution and recovers the unknown.

### Why This Requires Both Physics and Data

Physics alone (`L_pde`) has infinitely many `(u, k)` pairs that satisfy it. Data alone (100 sparse points) is nowhere near enough to pin down a full 2D solution. Together, they constrain each other: the data anchors the solution, the physics propagates that information everywhere, including back onto `k`.

---

## Assignment

### Part A — Recover k from Sparse Observations

**Ground truth:** generated via `scipy.solve_ivp`, then 100 points sampled at random `(x, t)` locations with `u` values as the only observed data.
**Network:** 4×64 tanh PINN, `k` initialized far from the truth (0.5 vs true 0.01).
**Loss:** `L_pde + 10·L_data + 10·L_ic + 10·L_bc`, 10,000 Adam epochs.

**Deliverables:**
1. Loss curve + `k` convergence plot
2. Noise sensitivity sweep at σ ∈ {0%, 1%, 5%}

### Part B — Track Proposal

Selected a specialization track for Weeks 7-8, with reading plan and project description.

---

## Key Results

**Recovered k (clean data):** 0.010988 vs true 0.01 → **9.88% error**

### Noise Sensitivity

| Noise σ | Recovered k | Rel. Error |
|---|---|---|
| 0.00 | 0.010203 | 2.03% |
| 0.01 | 0.011546 | 15.46% |
| 0.05 | 0.006041 | 39.59% |

Error grows with noise, as expected, noisier observations pull the data-fitting term toward incorrect values, and with only 100 sparse points there's limited redundancy to average that noise out. Recovery stays in a reasonable, physically plausible range even at 5% noise rather than diverging wildly, since the PDE residual continues to regularize the fit.

---

## Track Proposal — Track A: Inverse Problems & Parameter Identification

**Project:** extend this Week 6 approach to Burgers' equation, recovering the unknown viscosity `ν` from 200 sparse, noisy observations of a *nonlinear* PDE, with a noise sensitivity sweep (σ ∈ {0%, 1%, 5%, 10%}) and comparison against a classical scipy curve-fitting baseline.

**Reading plan:**
1. Raissi, Perdikaris, Karniadakis Part II (2017) — arXiv:1711.10566 — Section 4
2. Yang, Meng, Karniadakis — B-PINNs (2021) — arXiv:2003.06097 — Full
3. Yazdani et al. (2020) — Systems biology informed deep learning — Sections 1–2

---

## Conclusion & Next Steps

Week 6 demonstrated the core idea behind PINN-based inverse problems: an unknown physical parameter can be recovered purely by making it trainable and letting the physics + data constrain it jointly, no separate optimization loop needed the way classical methods require.

In **Week 7-8 (Track A)**, this same approach is pushed onto a genuinely harder, nonlinear problem: recovering Burgers' equation's viscosity from sparse, noisy data.
