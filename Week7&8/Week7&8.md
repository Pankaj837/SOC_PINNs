# Week 7-8 — Track A: Inverse Problems & Parameter Identification (Burgers' Viscosity Recovery)

## Overview

Week 6 established the core idea: an unknown PDE parameter can be recovered by making it trainable alongside the network, using sparse data + the physics constraint jointly. Weeks 7-8 push that same idea onto a genuinely hard, nonlinear problem, recovering the viscosity `ν` in Burgers' equation from 200 sparse, noisy observations, and benchmark it against a classical curve-fitting baseline.

This module also documents a real debugging story: the first working version converged to the wrong answer, and finding out *why* mattered more than the fix itself.

---

## The Problem
- uₜ + u·uₓ = ν·uₓₓ,   x ∈ [−1, 1],  t ∈ [0, 1]
- u(x, 0) = −sin(πx),   u(±1, t) = 0
- ν_true = 0.01/π ≈ 0.003183   (unknown to the model)

200 noisy point observations of `u(x,t)` are given. The task: recover `ν` purely from those observations plus the known PDE structure.

---

## Debugging Journey: Why the First Attempt Failed

The first working implementation, same recipe as Week 6, ported to Burgers' equation, converged, but to the **wrong** ν: relative error stayed in the 75–215% range no matter how much training or optimizer tuning was applied (Adam, then Adam + L-BFGS refinement).

**Root cause:** ν this small (0.003) produces a viscous shock only ~0.006 wide, thinner than the spacing between the grid points the 200 observations were being sampled from (~0.03). Almost no observation ever landed inside the region where ν actually has a visible effect. Away from the shock, Burgers' solutions for very different ν values look nearly identical, so the training data genuinely couldn't distinguish the correct ν from a wrong one. No amount of extra optimization fixes a parameter that the data can't identify.

**The actual fix, three parts:**
1. **Continuous, shock-aware sampling** — observations sampled directly at continuous `(x,t)` locations instead of snapped to a coarse grid, with a portion deliberately placed inside the shock region (`x` near 0, `t` in [0.6, 1.0]), a normal experimental design choice (sensor placement), not a change to the true ν or the physics.
2. **Shock-concentrated collocation points** — extra PDE-residual evaluation points in the same region, so autograd's computed `u_xx` is accurate exactly where ν's effect on the residual is largest.
3. **Log-space parametrization** — training `log(ν)` instead of `ν` directly, avoiding a clamp-based positivity constraint that fights the optimizer near its boundary, and putting the search on the parameter's natural multiplicative scale.

This is the version documented below.

---

## Assignment

### Week 7 Checkpoint (`week7_progress.ipynb`)
Working PINN recovering ν from **clean** observations only: exact-solution setup, shock-aware data + collocation sampling, PINN with log-ν parametrization, Adam (15,000 epochs, LR decay) followed by L-BFGS refinement.

### Week 8 — Final (`final_notebook.ipynb`)
Everything above, plus:
- Noise sensitivity sweep, σ ∈ {0%, 1%, 5%, 10%}
- Classical baseline: scipy `curve_fit` on an assumed linear-diffusion model
- Full results summary and discussion

---

## Key Results

### Clean-data recovery

| Stage | Recovered ν | Relative Error |
|---|---|---|
| After Adam (15,000 epochs) | 0.003191 | 0.23% |
| After L-BFGS refinement | 0.003163 | 0.64% |

(True ν = 0.003183)

### Noise Sensitivity

| Noise σ | Recovered ν | Rel. Error |
|---|---|---|
| 0.00 | 0.003172 | 0.36% |
| 0.01 | 0.003199 | 0.49% |
| 0.05 | 0.003166 | 0.55% |
| 0.10 | 0.003303 | 3.76% |

Error stays under 4% even at 10% observation noise, the PDE residual term acts as a regularizer, preventing the fit from chasing noise the way a purely data-driven method would.

### PINN vs. Classical Baseline

| Method | Recovered ν | Rel. Error |
|---|---|---|
| PINN (this work) | 0.003163 | 0.64% |
| Classical baseline (scipy curve_fit) | 0.040888 | 1184.55% |

The baseline assumes `u(x,t) ≈ A·sin(πx)·exp(−k·t)` — a linear diffusion model. It fails completely on Burgers' equation because that assumption breaks down under the nonlinear `u·uₓ` term. The PINN makes no such assumption; it learns the full nonlinear solution and recovers ν purely from the physics constraint.

---

## Discussion

**Where PINNs win here:** classical parameter-fitting methods need a closed-form model of the solution's shape. For a nonlinear PDE like Burgers', no simple closed form exists, the baseline's 1184% error is exactly this assumption failing. A PINN needs no such assumption; it discovers the shape and the parameter simultaneously.

**Where identifiability, not optimization, was the real constraint:** the debugging journey above is the main finding of this module. A converged, low-loss training run is not sufficient evidence that a recovered parameter is correct — if the data can't distinguish between candidate values of ν, the optimizer will happily converge to a self-consistent but wrong answer. Sensor placement (or sample coverage) matters as much as model capacity or training budget.

---

## Conclusion

Track A demonstrates the full arc of this project: from differentiating a network's output (Week 1) to using that mechanism to recover an unknown physical constant from sparse, noisy data on a nonlinear PDE (Week 7-8), while being honest about the one place a naive implementation silently fails, and why.
