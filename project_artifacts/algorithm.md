# HVK — Algorithm & Implementation Reference

**Project:** Hamiltonian Vision Kernel (HVK) — a hybrid quantum–classical image
autoencoder. Treats positionally-encoded image patches as quantum spins in an
external field and uses a Heisenberg-type Hamiltonian as a physics-guided
regularizer on top of a reconstruction loss.

This document describes the **shared pipeline** and every **algorithm variant**
(HVK1D, HVK2D, Symmetric HVK1D, and the Qiskit/IBM hardware probe), grounded in
the actual source files. It is a companion to `report.md` (experiment spec),
`report_2.md` (critical review), and `results.md` (results summary).

---

## 1. Shared pipeline (all variants)

```
Grayscale image
  → split into patches (with positional coordinates)
  → [PATH A] MPS/tensor-network features per patch  (quimb)
  → [PATH B] sinusoidal positional encoding per patch
  → project both into per-qubit angles (2 × nn.Linear)
  → Variational Quantum Circuit (PennyLane, default.qubit)
        · AngleEmbedding(MPS features)
        · RY(positional angles) per qubit
        · entangling layer(s) + parameterized rotations
        · measure Pauli observables  (Z, X, ZZ [, XX, YY])
        · Heisenberg energy from learnable couplings J·⟨PauliPair⟩
  → PatchDecoder MLP: [observables ⊕ positions] → … → Sigmoid → patch pixels
  → stitch patches (overlap/seam blending) → reconstructed image
```

**Loss:** `MSE(reconstruction, original) + λ · mean(Hamiltonian energy)`,
with `λ = 0.01`. Optimizer Adam, `lr = 0.003`.

**Key design intent:** the quantum observables act as a per-patch latent code;
the Hamiltonian energy term is the physics motivation connecting the VQC to a
spin model. (Whether these components are actually load-bearing is the subject
of `report_2.md` — see §6 there. This file documents the *design*, not the claim.)

**Core files**

| Role | Path |
|---|---|
| 1D VQC + observables | `Main/src/quantum/circuit.py` |
| 1D quantum model (`nn.Module`) | `Main/src/quantum/quantum_model.py` |
| 2D grid model | `python_library/src/hvk/hvk2d/model.py` |
| Symmetric (U(1)) model | `Main/src/quantum/symmetric_model.py` |
| MPS features | `Main/src/tensornetworks/mps_features.py` |
| Positional encoding | `Main/src/preprocessing/positional_encoding.py` |
| Decoder MLP | `Main/src/decoder/patch_decoder.py` |
| Training loop | `Main/src/training/training.py` |
| IBM/Qiskit probe | `IBM_Cloud/run_ibm_hvk_probe.py` |
| CIFAR benchmark runners | `Baselines/cifar10_comparisons/{hvk1d,hvk2d,symmetric_hvk1d}/` |
| newHVK diagnostic suite | `main2/newHVK/run_newhvk_suite.py` |

---

## 2. Variant A — HVK1D (chain topology) — the reference model

**File:** `Main/src/quantum/circuit.py`, `Main/src/quantum/quantum_model.py`

- **Qubits:** 6, arranged as a 1D **chain**; **bonds** = 5 nearest-neighbor pairs
  `(i, i+1)`.
- **Circuit:** `AngleEmbedding(features)` → `RY(positional_angles)` per qubit →
  `StronglyEntanglingLayers(weights)` with `n_layers = 2` (this is the entangling
  block — a ring of CNOTs + parameterized rotations).
- **Observables (27-dim, "full" set):**
  `Z_i (6) + X_i (6) + ZZ_{i,i+1} (5) + XX (5) + YY (5)`.
- **Hamiltonian energy:** `E = Σ Jz·⟨ZZ⟩ + Σ Jx·⟨XX⟩ + Σ Jy·⟨YY⟩`,
  with learnable per-bond couplings `Jx, Jy, Jz` (init `0.1·randn`).
- **Weights:** `nn.Parameter(rand(2, 6, 3)·π)` → 36 rotation params; +15 couplings = **51 quantum params**.
- **Training-time observable noise:** `+0.01·randn` added to observables when
  `training=True` (a robustness/regularization knob; ablatable via `observable_noise=False`).

**Configurable ablation modes** (all in the same `QuantumModel`):

| Mode flag | Effect |
|---|---|
| `vqc_mode="standard"` | full entangling circuit (default) |
| `vqc_mode="no-entanglement"` | replaces `StronglyEntanglingLayers` with per-qubit `Rot` only — same 36 params, **no two-qubit gates** |
| `vqc_mode="random"` | observables = fixed `randn` (no circuit) — noise control |
| `use_classical_replacement=True` | VQC → `Linear(n_qubits → obs_dim) + tanh` |
| `use_parameter_matched_classical=True` | rank-1 `tanh` bottleneck (parameter-matched classical control) |
| `observable_set="zz-only"` | drops XX/YY → `Z+X+ZZ` observables only |
| `qubit_count ∈ {4,6,8}` | qubit-count sweep |

This single class is what powers Exps 3–9 in `report.md`.

---

## 3. Variant B — HVK2D (grid topology)

**File:** `python_library/src/hvk/hvk2d/model.py`

- **Qubits:** 6, arranged as a **2×3 grid lattice**.
- **Bonds:** explicit horizontal + vertical edges
  `EDGES_H = [(0,1),(1,2),(3,4),(4,5)]`, `EDGES_V = [(0,3),(1,4),(2,5)]` → 7 edges.
- **Circuit:** per layer, apply **CNOTs along all grid edges** (horizontal then
  vertical), then per-qubit `Rot`. This hand-builds a 2D entanglement pattern
  instead of the 1D `StronglyEntanglingLayers` ring.
- **Observables (19-dim):** `Z_i (6) + X_i (6) + ZZ over 7 grid edges`.
- **Hamiltonian energy:** `E = Σ j_2d · ⟨ZZ_edge⟩` over the 7 grid edges
  (single coupling vector `j_2d`, init `0.1·randn`).
- **Decoder:** `PatchDecoder2D` — `Linear(obs+pos → 128) → ReLU → 256 → ReLU →
  patch² → Sigmoid`.

**Why it exists:** tests whether a 2D nearest-neighbor entanglement topology
(matching image locality) beats the 1D chain. On CIFAR32 it is the **best HVK
variant** (see `results.md`).

---

## 4. Variant C — Symmetric HVK1D (U(1)-symmetric)

**File:** `Main/src/quantum/symmetric_model.py`;
runner `Baselines/cifar10_comparisons/symmetric_hvk1d/run_symmetric_hvk1d_cifar32.py`

- Same 1D chain, but the Hamiltonian is restricted to a **U(1)-symmetric
  (particle-number-conserving) form:** `E = J·⟨ZZ⟩ + K·(⟨XX⟩ + ⟨YY⟩)` with
  learnable per-bond `J, K` (the `XX+YY` "hopping" term is symmetric, unlike the
  general `Jx·XX + Jy·YY`).
- Motivation: a physically-motivated symmetry constraint (XXZ-type model) that
  reduces the coupling degrees of freedom and connects to a conserved quantity.

---

## 5. Variant D — Qiskit / IBM hardware probe

**File:** `IBM_Cloud/run_ibm_hvk_probe.py`

This is **not** a full reconstruction model — it is a small **hardware-execution
probe** that runs tiny HVK-style circuits on real IBM Quantum backends (or a
dry-run for depth/gate accounting).

- **Encoding:** each patch vector → per-qubit `(mean, std)·π` → `RY, RZ` per qubit.
- **Entanglement:** per edge, `CX → RY(0.15) → CX` (a fixed-angle entangling
  block), over **chain edges** (`hvk1d`) or **grid edges** (`hvk2d`, computed from
  a √n column layout).
- **Measurement:** computational-basis sampling; from counts it estimates
  `mean_abs_order_parameter`, `mean_zz_correlation`, and a `hardware_proxy_loss
  = (1 − |order|) + 0.5·(1 − zz_corr)`.
- **Guards:** conservative IBM free-plan limits (≤2 circuits, ≤100 shots) unless
  `--allow-large-free-plan-job`; token via `IBM_QUANTUM_TOKEN`, never hard-coded.
- **Measured circuit cost** (6 qubits, transpiled): hvk1d depth 18 / 10 CX;
  hvk2d depth 18 / 14 CX (grid uses more two-qubit gates).

**Role:** demonstrates the circuits are hardware-runnable at small scale and
records depth / CX / order-parameter feasibility — a NISQ feasibility check, not
an advantage claim.

---

## 6. newHVK diagnostic suite (second generation)

**File:** `main2/newHVK/run_newhvk_suite.py`

A rebuilt, statistics-first workspace (multi-seed, real held-out CIFAR-10,
same-width classical controls, shot-noise sweep, IBM probe). It reuses the same
observable philosophy but abstracts the circuit into **feature maps**:

- `entangling_features(x)` = single-site features **plus explicit pair-product
  terms** (the "entanglement-sensitive" channel).
- `no_entanglement_features(x)` = single-site only.
- Plus `parameter-matched-classical`, `raw-linear`, `random-vqc`, `freeze-*`
  controls, all at matched feature width (32) and readout params (2112).

⚠️ **Caveat (see `report_2.md` §9):** the `cifar_nonlocal_advantage` diagnostic
constructs its regression **target** (`run_newhvk_suite.py:1374-1385`) **from the
same pair-product terms** it feeds only to the entangling-feature model
(`run_newhvk_suite.py:1396-1413`), so its
R²≈1.0 result is circular (label leakage), not a genuine advantage. The
**held-out real-CIFAR** result in the same suite is the trustworthy one.

---

## 7. Variant comparison at a glance

| Variant | Topology | Entangler | Observables | Energy term | Runs on HW? |
|---|---|---|---|---|---|
| **HVK1D** | 1D chain (6q, 5 bonds) | `StronglyEntanglingLayers` | Z,X,ZZ,XX,YY (27) | `Jx·XX+Jy·YY+Jz·ZZ` | via probe |
| **HVK2D** | 2×3 grid (6q, 7 edges) | manual CNOT grid | Z,X,ZZ (19) | `j_2d·ZZ` | via probe |
| **Sym-HVK1D** | 1D chain | StronglyEntangling | Z,X,ZZ,XX,YY | `J·ZZ+K·(XX+YY)` (U(1)) | — |
| **IBM probe** | chain or grid | `CX-RY-CX` per edge | basis sampling | order-param proxy | **yes (IBM)** |
| **newHVK** | feature-map abstraction | pair-product channel | configurable | n/a (regression) | proxy |

---

## 8. Reproduction entry points

```bash
# 1D reference reconstruction (Mona Lisa 256×256)
python Main/main.py

# CIFAR-10 32×32 benchmark (all variants + classical baselines)
python Baselines/cifar10_comparisons/hvk1d/run_hvk1d_cifar32.py
python Baselines/cifar10_comparisons/hvk2d/run_hvk2d_cifar32.py
python Baselines/cifar10_comparisons/symmetric_hvk1d/run_symmetric_hvk1d_cifar32.py

# newHVK second-generation suite (multi-seed, held-out CIFAR, shot noise)
./main2/newHVK/scripts/run_all.sh

# IBM hardware probe (dry-run = build + depth/gate report, no token needed)
python IBM_Cloud/run_ibm_hvk_probe.py --variant both --dry-run
```
