# Thermodynamic Qubit Collector–Modulator Reconstruction (TQ-CMR)

Status: private research prototype and candidate architecture. “Novel” is a
hypothesis requiring a systematic prior-art review, not a claim made here.

## 1. Image-to-spin map

Each grayscale pixel is represented by a qubit magnetization:

```text
x_hat_i = (1 + <Z_i>) / 2.
```

Observed pixels produce longitudinal fields `h_i = a(2*x_i-1)`. Withheld
pixels have no direct image field.

## 2. Collector

For each overlapping local window, prepare the same transverse-field Ising
system at two temperatures:

```text
H_C = -J_C sum_<i,j> Z_i Z_j - sum_i h_i Z_i - Gamma_C sum_i X_i.
```

Collect:

- low- and high-temperature magnetizations;
- thermal response `chi_i = |m_i(T_low)-m_i(T_high)| / |T_high-T_low|`;
- local confidence `c_i = |mean_T m_i(T)|`.

The collector is not the final reconstruction. It estimates local spin
orientation, uncertainty, and temperature sensitivity.

## 3. Modulator

The collector reshapes a second Hamiltonian:

```text
H_M =
  -sum_<i,j> J_ij^M Z_i Z_j
  -sum_i h_i^M Z_i
  -sum_i Gamma_i^M X_i.
```

An optional missing-pixel feedback channel is:

```text
h_i^M = lambda * mean_T(m_i^C).
```

Revision 1 showed that positive `lambda` self-reinforced collector errors and
collapsed grayscale detail. Revision 2 therefore freezes `lambda=0`: the
collector modulates interactions and transverse uncertainty, not a
target-shaped longitudinal field.

Edge-aware couplings reduce smoothing across collector-detected boundaries:

```text
J_ij^M = J_M [floor + (1-floor) exp(-kappa |m_i^C-m_j^C|)].
```

Quantum exploration is concentrated at uncertain sites:

```text
Gamma_i^M = Gamma_M [floor + (1-floor)(1-c_i)].
```

The final pixels are measured from the Gibbs state of `H_M`.

## 4. What is and is not quantum

Both collector and modulator are quantum Gibbs models with non-commuting `X`
and `Z` terms. The present exact simulation is classical. Moreover, measuring
collector observables and using them to program the modulator is classical
feedback, so the end-to-end prototype is a hybrid quantum-control protocol.
It has no neural decoder, learned convolution, MPS encoding, or image-stitching
model.

A future coherent version could replace measured feedback with ancilla-mediated
controlled interactions, but it cannot simply copy unknown collector states;
that design must respect the no-cloning theorem.

## 5. Falsifiable evaluation

The collector–modulator system must be compared with:

1. the frozen one-pass transverse-field model;
2. a `Gamma=0` collector–modulator control;
3. Gaussian interpolation;
4. new held-out masks after all modulation parameters are frozen.

Improvement over the one-pass quantum model tests whether modulation helps.
Improvement over the `Gamma=0` modulator isolates the transverse contribution.
Improvement over Gaussian interpolation is required before claiming practical
image-reconstruction value.

## 6. Version 3: finite-reservoir heat-current dynamics

Version 3 adopts the thermodynamic-neuron current law from the separate
thermodynamic-gates project. Each pixel is a finite output reservoir:

```text
g(beta) = 1 / (1 + exp(beta*epsilon))
J_C = mu*epsilon*(g(beta_C)-g(beta_z))
J_M = mu_prime*epsilon*(g(beta_M)-g(beta_z))
d beta_z/dt = -(beta_z^2/C)*(J_C+J_M).
```

Here currents are defined as positive *into* the finite output reservoir. The
reference implementation uses the opposite population ordering while retaining
the same minus sign in the beta ODE; used literally, that makes the
zero-current fixed point unstable. Version 3 states and tests the stable sign
convention explicitly.

The quantum Ising system supplies the collector occupation. A spatial thermal
reservoir, constructed only from observed pixels, supplies the modulator
occupation. Pixel intensity is the evolved output occupation `g(beta_z(t))`.
This replaces the earlier direct observable-to-Hamiltonian feedback with an
explicit finite-reservoir heat-flow mechanism.
