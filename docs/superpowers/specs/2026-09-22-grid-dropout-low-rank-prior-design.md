# Grid Dropout-EAKI with a Degenerate Low-Rank Prior

## Goal

Create a new, concise implementation of grid-valued dropout-EAKI for the
barotropic inverse problem without editing the current implementations.  The
new implementation must use the actual grid prior induced by the coefficient
prior, retain the scalable ensemble-space calculations from `InflatedEKI.jl`,
and retain the stability and diagnostic features added to `InflationEKI.jl`.

The deliverable also produces the three requested diagnostics:

1. relative initial-vorticity error and data misfit versus iteration;
2. truth and recovered initial-vorticity fields;
3. the final pointwise ensemble standard-deviation field.

## New Files

Implementation will be confined to new files:

- `Inversion/InflatedEKI_LowRankPrior.jl`
- `Climate/Barotropic/GridDropoutEAKI_Barotropic_LowRankPrior.jl`
- `Climate/Barotropic/Section52LowRankPlots.jl`
- `Climate/Barotropic/run_grid_dropout_eaki_lowrank.jl`
- `Climate/Barotropic/GridDropoutEAKI_Barotropic_LowRankPrior_smoke.jl`

Existing source, run, plotting, and smoke-test files remain unchanged.  The new
barotropic implementation may include and call existing physical-model code
from `Barotropic.jl`, but it must not edit it.

## Mathematical Problem

Let `tau` be the truncated spectral-coefficient vector and let

```
tau ~ N(m_tau, C_tau),       C_tau = L_tau * L_tau'
theta = m_grid + B * (tau - m_tau).
```

Define the thin grid-prior factor

```
U = B * L_tau.
```

The induced covariance is

```
C_grid = U * U',
```

which is singular in the full grid space.  The inverse problem is therefore
the constrained problem

```
minimize over theta in m_grid + range(U):

    0.5 * ||Sigma_nu^(-1/2) * (y_obs - G(theta))||^2
  + 0.5 * ||U^+ * (theta - m_grid)||^2.
```

Equivalently, with `theta = m_grid + U * xi`, the prior term is
`0.5 * ||xi||^2`.  The implementation must not replace this prior by an
isotropic covariance in the ambient grid space.

## Low-Rank Prior Representation

The implementation will factor the thin matrix `U` once using a thin QR or
rank-revealing SVD.  It must expose the following operations without forming
`U*U'`, `U^+`, or an ambient-space projector:

- `prior_sample(rng, n_ens)`: return `m_grid + U * Xi`;
- `prior_project(z)`: return the orthogonal projection of `z` onto `range(U)`;
- `prior_whiten(z)`: return the unique white coordinates associated with a
  vector in `range(U)`;
- `prior_penalty(theta)`: return
  `0.5 * ||prior_whiten(theta - m_grid)||^2`.

Rank deficiency in `U` beyond the intended coefficient truncation is an error
and must be reported explicitly.

Both supported coefficient priors use the same machinery:

- isotropic coefficient prior `C_tau = sigma^2 I`;
- degree-dependent power-law coefficient prior constructed by
  `barotropic_power_law_prior_cov`.

## EKI Core

`InflatedEKI_LowRankPrior.jl` will be a standalone implementation.  It will
not include both existing EKI files because they define conflicting global
symbols.

It retains these properties from `InflatedEKI.jl`:

- Kalman actions are evaluated through an `N_ens x N_ens` Gram matrix;
- no `N_theta x N_y` Kalman-gain matrix is formed;
- each stochastic-EKI member receives independent observation noise;
- mean predictions are stored as `G(mean(theta_n))`.

It incorporates these properties from the updated `InflationEKI.jl`:

- optional adaptive joint-dropout weight;
- optional Armijo line search for the dropout-EAKI mean;
- explicit mean/anomaly state when line search is active;
- iteration callback and line-search diagnostics;
- finite-value checks and safe numerical fallbacks.

The implementation will avoid copying all eight filter branches unless they
are needed by the new runner.  The required branch is `dropout-EAKI`; small
shared helpers may also support `EAKI` if doing so reduces duplication rather
than increasing it.

## Prior-Augmented Ensemble-Space Update

For observation anomalies `Y` and grid anomalies `Z`, define

```
Yw = Sigma_nu^(-1/2) * Y
Zw = prior_whiten(Z).
```

The prior-augmented Gram matrix is assembled without augmented observations:

```
Gram = I + Yw' * Yw + Zw' * Zw,
```

with the appropriate artificial-time scaling applied to both the data and
prior blocks.  Mean-update right-hand sides similarly combine the whitened
data residual with the whitened prior residual.  The same augmented inner
product is used in the sequential or joint dropout correction.

Grid dropout first masks grid components and then projects the masked
direction back to the prior support:

```
Z_drop = prior_project(mask .* Z).
```

Every accepted mean and anomaly is projected back to the affine support
`m_grid + range(U)`.  Consequently, the algorithm cannot acquire unpenalized
components in `range(U)^perp`.

## Observation Noise

Observation noise `Sigma_nu` is distinct from the degenerate grid prior.  The
new implementation accepts `Diagonal` or dense positive-definite observation
covariance matrices and caches a square-root factor.  The barotropic runner
uses the existing heteroscedastic diagonal observation-noise construction.

The data misfit is always

```
0.5 * ||Sigma_nu^(-1/2) * (y_obs - G(mean(theta_n)))||^2.
```

No ensemble-size normalization and no prior penalty are included in this
diagnostic.

## Barotropic Grid Layer

The new Grid file owns only problem-specific responsibilities:

- construct the Section 5.2 barotropic problem and noisy observations;
- convert between grid parameters and physical vorticity fields;
- project physical fields into the T85 model space when requested;
- construct the coefficient-to-grid map `B` and thin prior factor `U`;
- invoke the generic low-rank-prior dropout-EAKI core;
- calculate and serialize diagnostics.

It will not contain a second copy of the EKI update equations.

The public entry points are:

- `run_section52_grid_dropout_eaki_lowrank`
- `run_section52_grid_dropout_eaki_lowrank_smoke_test`

The result stores at least:

- problem configuration and prior metadata;
- `theta_est` and `grid_vor_est`;
- `mean_predictions`;
- `vorticity_errors`;
- `data_misfits`;
- `optimization_errors` (data plus true low-rank prior penalty);
- `grid_vor_std`;
- output and plot paths;
- the EKI object only when requested.

## Diagnostics and Plots

`Section52LowRankPlots.jl` will contain only the plotting functions needed by
the new runner and will reuse existing field-plot helpers where safe.

The convergence figure has two panels:

- left: `||omega(mean_n) - omega_0||_2 / ||omega_0||_2`;
- right: the pure data misfit defined above.

The recovered-field figure displays the truth and final recovered initial
vorticity on a common color scale.

For the uncertainty figure, each final ensemble member is first reconstructed
and projected using the same path as the reported estimate.  The corrected
sample standard deviation is then computed pointwise across members.  The
standard-deviation field itself is not spectrally projected after the
nonlinear variance calculation.

## Run Script

`run_grid_dropout_eaki_lowrank.jl` will be a short declarative experiment
configuration.  It will select either the isotropic or power-law coefficient
prior, call the new runner once, and print the principal output paths and final
errors.  Algorithm implementation will not be duplicated in the run script.

## Verification

The new smoke test will verify:

1. low-rank prior sampling, projection, and whitening;
2. isotropic and power-law coefficient priors;
3. equivalence of the compact update and an explicit small augmented problem;
4. every ensemble mean and anomaly remains in `range(U)`;
5. `data_misfits[n]` equals the requested formula evaluated at
   `G(mean(theta_n))`;
6. `optimization_errors[n]` equals data misfit plus the true low-rank prior
   penalty;
7. recovered and standard-deviation fields have the model-grid shape and the
   standard deviation is nonnegative;
8. all three requested plot files are created by the small barotropic smoke
   configuration.

The existing InflationEKI tests will also be run where they are applicable,
but existing files will not be changed to accommodate the new implementation.

## Compatibility and Non-Goals

- Existing files and entry points remain untouched.
- The new implementation does not pretend that the low-rank grid prior has an
  ordinary inverse on the ambient grid space.
- It does not build a dense grid covariance or an explicitly augmented grid
  observation covariance.
- It does not change the T85 barotropic dynamics.
- It does not refactor unrelated optimizers or parameter-space experiments.
