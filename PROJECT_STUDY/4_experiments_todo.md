# Experiments TODO

**Central question:** Are the classical parameters (projections + decoder MLP) doing all the learning, or does the quantum circuit contribute meaningfully?

---

## Experiment 1 — Shuffled Observables at Eval (zero retraining)

**What:** Train the full model normally. At eval, shuffle which observable vector goes to which patch (permute the 16 patch indices), but keep positions correctly assigned.

**Code change:** Add ~3 lines to the eval block in `training.py` after the main forward pass:
```python
perm = torch.randperm(16)
shuffled_pred = decoder(observables[perm], positions)
# save shuffled_pred as an additional output panel
```

**Interpret:**
- Shuffled ≈ normal reconstruction → decoder ignores observables, uses position only → classical MLP is doing the work
- Shuffled reconstruction degrades → observables carry patch-specific content → quantum features are load-bearing

**Cost:** No retraining. One eval run.

---

## Experiment 2 — Zero Quantum Features, Real Positions (flag already exists)

**What:** At eval, pass `zeros` for observables but real positions to the decoder.

**Code change:** None. Flag already exists:
```bash
python Main/main.py --zero-latent-uses-positions
```

**Interpret:**
- Good reconstruction → decoder memorized patch content from position alone during training
- Poor reconstruction → quantum observables are needed to distinguish patches

**Cost:** Just rerun with the flag.

---

## Experiment 3 — Freeze Classical, Train Only Quantum (51 params)

**What:** Freeze `feature_projection`, `position_projection`, and entire `PatchDecoder` at random initialization. Train only `weights [2,6,3]` (36 params) and `Jx/Jy/Jz` (15 params).

**Code change:** In `training.py :: train()`, before the optimizer is constructed, add:
```python
for param in decoder.parameters():
    param.requires_grad_(False)
model.feature_projection.requires_grad_(False)
model.position_projection.requires_grad_(False)
```

**Interpret:**
- Good reconstruction with only 51 quantum params → quantum circuit is genuinely learning a useful representation
- Poor reconstruction → quantum params alone are insufficient; classical layers are essential

**Cost:** Small code change, one full training run.

---

## Experiment 4 — Freeze Quantum, Train Only Classical (1.09M params)

**What:** Freeze all quantum weights at random initialization. Train only projections + decoder.

**Code change:** In `training.py :: train()`, before the optimizer:
```python
model.weights.requires_grad_(False)
model.Jx.requires_grad_(False)
model.Jy.requires_grad_(False)
model.Jz.requires_grad_(False)
```

**Interpret:**
- Good reconstruction → 1.09M classical params can learn with VQC as a fixed random nonlinearity → quantum training is not needed
- Poor reconstruction → trained quantum weights matter

**Cost:** Small code change, one full training run.

---

## Experiment 5 — Replace VQC Output with Random Noise

**What:** Skip the VQC entirely. Replace `VQC(...)` call with `torch.randn(27)` in `QuantumModel.forward()`. Train projections + decoder as normal.

**Code change:** In `quantum_model.py :: QuantumModel.forward()`, replace:
```python
output = torch.stack(VQC(feature_vector, position_vector, self.weights))
```
with:
```python
output = torch.randn(27, device=features.device)
```

**Interpret:**
- Good reconstruction → the decoder can recover the image from pure noise + position. The circuit is irrelevant.
- Poor reconstruction → the structured output of the VQC (not just its dimensionality) matters.

**Cost:** One line change, one training run.

---

## Summary Table

| # | Experiment | Code change | Retraining? | Tests |
|---|-----------|-------------|-------------|-------|
| 1 | Shuffled observables at eval | ~3 lines in training.py | No | Do observables carry content? |
| 2 | `--zero-latent-uses-positions` | None (flag exists) | No | Does position alone reconstruct? |
| 3 | Freeze classical, train quantum only | ~3 lines in training.py | Yes | Can 51 quantum params learn? |
| 4 | Freeze quantum, train classical only | ~4 lines in training.py | Yes | Can 1.09M classical params learn with fixed VQC? |
| 5 | Replace VQC with random noise | 1 line in quantum_model.py | Yes | Is structured VQC output needed at all? |

---

## Recommended Order

1. **Exp 2** first — free, just a flag, answers position-memorization question immediately
2. **Exp 1** next — no retraining, directly tests whether observables matter
3. **Exp 4** — if Exp 1 shows observables matter, check if *trained* quantum weights matter
4. **Exp 3** — if Exp 4 is poor, check if quantum-only training recovers it
5. **Exp 5** — last resort: does structure of VQC output matter vs pure noise

## Decision Tree

```
Exp 2: decoder(zeros, positions) reconstructs well?
├── YES → position memorization confirmed → quantum features not needed
│         Run Exp 1 to confirm, then rethink architecture
└── NO  → quantum features genuinely needed
          Exp 1: shuffled observables degrades?
          ├── YES → observables carry content
          │         Exp 4: frozen quantum still works?
          │         ├── YES → classical params do the work, quantum training not needed
          │         └── NO  → trained quantum weights matter → quantum is contributing
          └── NO  → decoder ignores observable values entirely → rethink
```

---

## Extended Profiling — Complete Quantum Contribution Analysis

> These experiments require a new git branch. Student should branch from `main` and implement one experiment per sub-branch, keeping each change isolated and reviewable.

```bash
git checkout -b ablation/quantum-profiling
```

---

### Exp 6 — No Entanglement (Single-Qubit Rotations Only)

**What:** Remove all CNOT gates from the circuit. Replace `StronglyEntanglingLayers` with only per-qubit `Rot` gates — no entanglement, no qubit-qubit interaction.

**Code change:** In `circuit.py`, replace:
```python
qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
```
with:
```python
for i in range(n_qubits):
    qml.Rot(weights[0, i, 0], weights[0, i, 1], weights[0, i, 2], wires=i)
    qml.Rot(weights[1, i, 0], weights[1, i, 1], weights[1, i, 2], wires=i)
```

**Interpret:**
- Quality unchanged → entanglement contributes nothing; single-qubit rotations suffice
- Quality drops → entanglement (qubit-qubit correlations) is genuinely needed

**Branch:** `ablation/no-entanglement`

---

### Exp 7 — Replace VQC with Classical Equivalent (tanh MLP)

**What:** Replace the entire VQC call with a small classical MLP of identical input/output shape: `Linear(6→27)` + tanh. Same 6-dim input, same 27-dim output. This is the cleanest quantum-vs-classical comparison.

**Code change:** In `quantum_model.py`, add `self.classical_replacement = nn.Linear(6, 27)` in `__init__`, then in `forward()` replace:
```python
output = torch.stack(VQC(feature_vector, position_vector, self.weights))
```
with:
```python
combined = projected_features[i] + projected_positions[i]
output = torch.tanh(self.classical_replacement(combined))
```

**Interpret:**
- Classical MLP matches or beats VQC → quantum circuit is not adding computational advantage
- VQC clearly better → quantum nonlinearity / superposition provides something classical tanh cannot

**Branch:** `ablation/classical-replacement`

---

### Exp 8 — Replace MPS Features with Simple Patch Statistics

**What:** Replace the 46-dim MPS feature vector with simple classical patch statistics: mean, std, min, max, and histogram bins — no tensor network, no `quimb`. Tests whether the MPS compression is necessary or if basic stats suffice.

**Code change:** In `mps_features.py`, add an alternative function:
```python
def extract_patch_statistics(patch, n_bins=38):
    flat = patch.flatten()
    hist, _ = np.histogram(flat, bins=n_bins, range=(0,1))
    return np.concatenate([[flat.mean(), flat.std(), flat.min(), flat.max()],
                           hist / hist.sum()])  # → 42-dim, adjust Linear(42→6)
```
Also update `feature_projection` to `Linear(42→6)` (or match to chosen dim).

**Interpret:**
- Similar quality → MPS features are not providing unique information; simple stats are enough
- Quality drops → MPS tensor network compression captures structure that statistics miss

**Branch:** `ablation/no-mps`

---

### Exp 9 — Vary Projection Bottleneck Width (46→N)

**What:** The `feature_projection` compresses 46 MPS features down to 6 (one per qubit). Test N=3, 6, 12, 27 to see if the bottleneck is too tight or just right.

**Code change:** In `quantum_model.py`, make `n_qubits` the bottleneck dim. Run separate training jobs with different values. Note: changing N also changes `AngleEmbedding` and `RY` gate count, so `n_qubits` in `circuit.py` must match.

**Interpret:**
- Quality improves with wider projection → information is being lost at the 46→6 bottleneck
- Quality flat → 6 dimensions is sufficient to capture the relevant MPS features

**Branch:** `ablation/projection-width`

---

### Exp 10 — Vary Number of Qubits (4, 6, 8)

**What:** Test `n_qubits = 4` and `n_qubits = 8`. This changes circuit depth, observable count, and projection width simultaneously.

**Observable counts:**
- 4 qubits → 4+4+3+3+3 = 17 observables
- 6 qubits → 27 observables (current)
- 8 qubits → 8+8+7+7+7 = 37 observables

**Code change:** In `circuit.py`, change `n_qubits`. Update `feature_projection` and `position_projection` input sizes accordingly in `quantum_model.py`. The `HVKRunConfig.validate()` in `python_library/` hardcodes `n_qubits=6` — bypass it or update it.

**Interpret:**
- More qubits → better reconstruction → observable dimensionality matters (richer latent)
- Fewer qubits sufficient → 6 is over-engineered

**Branch:** `ablation/qubit-count`

---

### Exp 11 — Remove Energy Regularizer from Loss

**What:** Train without the Heisenberg energy term in the loss. Set coefficient to 0:
```
loss = reconstruction_loss   # instead of reconstruction_loss + 0.01 * energy_loss
```

**Code change:** In `training.py`, comment out or zero the energy term:
```python
loss = reconstruction_loss  # + 0.01 * energy_loss
```

**Interpret:**
- No quality change → the energy regularizer is cosmetic; Jx/Jy/Jz couplings don't help training
- Quality drops → energy term is guiding the VQC toward physically meaningful states

**Branch:** `ablation/no-energy-loss`

---

### Exp 12 — Remove Training Noise on Observables

**What:** Remove the `0.01 * randn` noise added to observables during training. Tests if this noise regularization is actually needed.

**Code change:** In `quantum_model.py`, remove or comment:
```python
# if self.training:
#     observables = observables + 0.01 * torch.randn_like(observables)
```

**Interpret:**
- No quality change → noise regularization is unnecessary
- Overfitting worsens → noise was preventing decoder from memorizing exact observable values

**Branch:** `ablation/no-obs-noise`

---

### Exp 13 — Vary MPS Bond Dimension (χ = 1, 2, 4, 8)

**What:** Test different bond dimensions for MPS compression. χ=1 is a product state (no entanglement in the MPS itself). χ=4 is current. χ=8 is higher fidelity.

**Code change:** In `mps_features.py`, find the `quimb` MPS compression call and vary the `max_bond` parameter. Feature vector size changes with χ, so update `feature_projection` input dim accordingly.

**Interpret:**
- χ=1 (no MPS entanglement) works fine → tensor network correlations in features are unnecessary
- Quality scales with χ → MPS compression quality directly impacts reconstruction

**Branch:** `ablation/bond-dim`

---

### Exp 14 — Reduced Observable Set (ZZ only, like HVK2D)

**What:** Drop XX and YY measurements. Use only Z + X + ZZ (19 observables, same as HVK2D). Tests whether the full Heisenberg observable set is needed or ZZ-only suffices.

**Code change:** In `circuit.py`, remove XX and YY from the return:
```python
return Z + X + ZZ  # 19 values instead of 27
```
Update `observable_dim = n_qubits + n_qubits + n_bonds` and decoder input dim.

**Interpret:**
- ZZ-only matches full → XX and YY measurements add no useful information to the latent
- Quality drops → the full Heisenberg observable set (XX+YY capturing transverse correlations) is needed

**Branch:** `ablation/zz-only`

---

## Full Summary Table

| # | Experiment | Files to change | Branch name | Tests |
|---|-----------|----------------|-------------|-------|
| 1 | Shuffled observables at eval | training.py | — (no branch needed) | Do observables carry content? |
| 2 | `--zero-latent-uses-positions` | — (flag exists) | — (no branch needed) | Does position memorize? |
| 3 | Freeze classical, train quantum only | training.py | `ablation/freeze-classical` | Can 51 quantum params learn? |
| 4 | Freeze quantum, train classical only | training.py | `ablation/freeze-quantum` | Can classical learn with fixed VQC? |
| 5 | Replace VQC with random noise | quantum_model.py | `ablation/random-vqc` | Is VQC structure needed? |
| 6 | No entanglement (Rot only, no CNOT) | circuit.py | `ablation/no-entanglement` | Does entanglement help? |
| 7 | Replace VQC with classical tanh MLP | quantum_model.py | `ablation/classical-replacement` | Quantum vs classical nonlinearity |
| 8 | Replace MPS with patch statistics | mps_features.py | `ablation/no-mps` | Is tensor network needed? |
| 9 | Vary projection width (46→3/12/27) | quantum_model.py, circuit.py | `ablation/projection-width` | Is 46→6 bottleneck too tight? |
| 10 | Vary qubit count (4, 6, 8) | circuit.py, quantum_model.py | `ablation/qubit-count` | Is 6 qubits optimal? |
| 11 | Remove energy regularizer | training.py | `ablation/no-energy-loss` | Does Heisenberg loss help? |
| 12 | Remove training noise on observables | quantum_model.py | `ablation/no-obs-noise` | Does noise regularization matter? |
| 13 | Vary MPS bond dim (χ=1,2,4,8) | mps_features.py | `ablation/bond-dim` | Does MPS quality matter? |
| 14 | ZZ-only observables (drop XX, YY) | circuit.py | `ablation/zz-only` | Is full Heisenberg set needed? |

---

## Git Workflow for Student

```bash
# Start from clean main
git checkout main
git pull

# For each experiment, create a fresh branch from main
git checkout -b ablation/freeze-classical
# ... make changes, run, save outputs to outputs/ablation/freeze-classical/ ...
git add -A && git commit -m "ablation: freeze classical params, train quantum only"

# Back to main for next experiment
git checkout main
git checkout -b ablation/no-entanglement
# ... and so on
```

Each branch should save its outputs to a clearly named subfolder under `outputs/ablation/<branch-name>/` so results can be compared side by side.

**Metric to report for each:** MSE, SSIM, PSNR on the final reconstruction vs original — same metrics used in `Baselines/cifar10_comparisons/common.py`.

*Last updated: 2026-07-02*
