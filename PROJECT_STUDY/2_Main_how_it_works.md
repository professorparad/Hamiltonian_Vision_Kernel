# Main/ — How It Works

## Entry Point

```
python Main/main.py [--model-variant standard|symmetric|both] [--steps 120] [--image-path ...]
```

Default runs **both** variants sequentially (standard HVK1D then symmetric HVK1D).

---

## Step-by-Step Pipeline

### Step 1 — Load Image
`src/preprocessing/image_loader.py`
- Loads image as grayscale, resizes to 256×256
- Output: `image` numpy array [256, 256], values in [0,1]

### Step 2 — Patch Extraction
`src/preprocessing/patching.py :: extract_patches()`
- Splits 256×256 image into non-overlapping 64×64 patches
- 256/64 = 4 rows × 4 cols = **16 patches total**
- Also returns **normalized positions** in [0,1): `positions[i] = [row/n_rows, col/n_cols]`
- Output: `patches` [16, 64, 64], `positions` [16, 2]

### Step 3 — MPS Feature Extraction
`src/tensornetworks/mps_features.py :: extract_mps_features()`
- Each 64×64 patch is flattened to 4096 values → reshaped to [2]^12 (12-site spin-½ tensor)
- Compressed into MPS with bond dimension χ=4 using `quimb`
- Extracts 3 types of features per patch:
  - 24 local: `<Z_i>` and `<X_i>` for each of 12 sites
  - 11 bond: `<Z_i Z_{i+1}>` two-site correlations
  - 11 entropy: von Neumann entanglement entropy at each bond
- **Total: 46-dim feature vector per patch**
- Feature matrix normalized: zero mean, unit std across the 16 patches

### Step 4 — Positional Encoding
`src/preprocessing/positional_encoding.py :: sinusoidal_positional_encoding()`
- Encodes 2D position [row, col] ∈ [0,1)² as sinusoidal features (like ViT)
- d_model=8: 2 frequency bands × sin+cos × x+y = 8 values per patch
- Output: `positions` tensor [16, 8]

### Step 5 — Quantum Model Forward Pass
`src/quantum/quantum_model.py :: QuantumModel.forward()`  
`src/quantum/circuit.py :: VQC()` (PennyLane QNode)

#### 5a — Classical projections (before the circuit)

The circuit has exactly **6 qubits**. Both the 46-dim MPS features and 8-dim positional encoding must be compressed to 6 values each before entering. This is done by two learned classical linear layers inside `QuantumModel.__init__()`:

```
features    [16, 46]  →  feature_projection  (Linear 46→6)  →  [16, 6]
positions   [16,  8]  →  position_projection (Linear  8→6)  →  [16, 6]
```

These are standard `nn.Linear` layers — their weights are trained by backprop alongside the quantum circuit weights. The projection is **not** fixed; it learns which combinations of MPS features and which combinations of position coordinates are most informative for the circuit.

After projection, the loop runs **once per patch** (16 iterations), feeding a single [6] vector for features and a single [6] vector for position into the circuit.

#### 5b — Inside the VQC (circuit.py)

The circuit has 3 stages executed in sequence on 6 qubits:

**Stage 1 — AngleEmbedding (feature encoding)**
```python
qml.AngleEmbedding(inputs, wires=range(6))
```
- `inputs` = the projected feature vector [6], one value per qubit
- Applies `RX(inputs[i])` rotation on qubit `i` for i=0..5
- This encodes MPS-derived patch content into the initial qubit state
- All 6 qubits start at |0⟩, then get rotated by their respective feature angle

**Stage 2 — Positional modulation (RY gates)**
```python
for qubit in range(6):
    qml.RY(positional_angles[qubit], wires=qubit)
```
- `positional_angles` = the projected position vector [6], one angle per qubit
- Applies `RY(θ_i)` on each qubit **on top of** the AngleEmbedding state
- This shifts the qubit state depending on where in the image the patch sits
- Patches at different grid positions get different RY rotations → circuit "feels" location
- Combined effect of Stage 1+2: qubit `i` has had `RX(feature_i)` then `RY(position_i)`

**Stage 3 — StronglyEntanglingLayers (entanglement + trainable)**
```python
qml.StronglyEntanglingLayers(weights, wires=range(6))
```
- `weights` shape: `[n_layers=2, n_qubits=6, 3]` — an `nn.Parameter` initialized randomly in [0, π]
- Each layer applies: parametrized single-qubit rotations (Rot gate = RZ·RY·RZ) on every qubit, followed by CNOT entangling gates between qubits (in a pattern that shifts per layer)
- 2 layers × 6 qubits × 3 parameters = **36 trainable quantum weights**
- This is where the circuit learns to create quantum correlations between qubits

#### 5c — Measurement (27 observables)

After the circuit runs, 27 Pauli expectation values are measured:

```
output[0:6]   = Z  = [⟨Z₀⟩, ⟨Z₁⟩, ⟨Z₂⟩, ⟨Z₃⟩, ⟨Z₄⟩, ⟨Z₅⟩]          # local spin-z
output[6:12]  = X  = [⟨X₀⟩, ⟨X₁⟩, ⟨X₂⟩, ⟨X₃⟩, ⟨X₄⟩, ⟨X₅⟩]          # local spin-x
output[12:17] = ZZ = [⟨Z₀Z₁⟩, ⟨Z₁Z₂⟩, ⟨Z₂Z₃⟩, ⟨Z₃Z₄⟩, ⟨Z₄Z₅⟩]     # nearest-neighbor ZZ
output[17:22] = XX = [⟨X₀X₁⟩, ⟨X₁X₂⟩, ⟨X₂X₃⟩, ⟨X₃X₄⟩, ⟨X₄X₅⟩]     # nearest-neighbor XX
output[22:27] = YY = [⟨Y₀Y₁⟩, ⟨Y₁Y₂⟩, ⟨Y₂Y₃⟩, ⟨Y₃Y₄⟩, ⟨Y₄Y₅⟩]     # nearest-neighbor YY
```

All values are in [-1, +1] (expectation values of Pauli operators).  
This 27-dim vector is the **quantum latent representation** of the patch.

#### 5d — Heisenberg energy (from the same observables)

Back in `QuantumModel.forward()`, the Hamiltonian energy is computed from the measured ZZ/XX/YY values:

```python
energy = sum(Jz * ZZ) + sum(Jx * XX) + sum(Jy * YY)
```

- `Jz`, `Jx`, `Jy` are `nn.Parameter` vectors of shape [5], one coupling per bond
- Initialized as `0.1 * randn` — small random values, learned during training
- Energy is a scalar per patch; stacked to `energies` [16]
- This energy goes into the loss as `0.01 * mean(energies)` — it regularizes the couplings toward physically meaningful values

#### 5e — Training noise

During training only (not eval):
```python
observables = observables + 0.01 * torch.randn_like(observables)
```
Small Gaussian noise is added to observables to act as a regularizer and prevent the decoder from overfitting to exact observable values.

#### Summary of all trainable parameters in Step 5

| Component | Shape | Count |
|-----------|-------|-------|
| `feature_projection` (Linear 46→6) | [6, 46] + bias [6] | 282 |
| `position_projection` (Linear 8→6) | [6, 8] + bias [6] | 54 |
| `weights` (VQC StronglyEntangling) | [2, 6, 3] | 36 |
| `Jx`, `Jy`, `Jz` (Heisenberg couplings) | 3 × [5] | 15 |
| **Total Step 5** | | **387** |

For **SymmetricQuantumModel** (U(1) symmetric):
- Same circuit, projections, weights — identical forward pass
- Only the Hamiltonian changes: `H = Σ J·ZZ + Σ K·(XX+YY)` with J [5] and K [5]
- Enforces axial (U(1)) symmetry: XX and YY couplings are tied together (same K)
- 10 coupling parameters instead of 15, total Step 5 = 382

### Step 6 — Classical Decoder
`src/decoder/patch_decoder.py :: PatchDecoder.forward()`
- Input: concatenate `observables [16,27]` + `positions [16,8]` → [16, 35]
- MLP: 35 → 256 → 1024 → patch_size² → reshape to [16, 1, 64, 64]
- Sigmoid activation at output → pixel values in [0,1]

### Step 7 — Loss and Backprop
In `training.py :: train()`, each step:
```
loss = reconstruction_loss + 0.01 * energy_loss
     = MSE(decoder_output, target_patches) + 0.01 * mean(energies)
```
- Adam optimizer updates ALL parameters: Jx/Jy/Jz, VQC weights, MLP decoder weights
- MPS compression is NOT trained (fixed feature extractor)
- Training runs for `steps=120` iterations over all 16 patches simultaneously (batch = all patches)

### Step 8 — Reconstruction
After training, for evaluation:
- `pred` = decoder(final observables, positions) → 16 patches [16,1,64,64]
- `stitch_patches()` → assembles back into 256×256 image
- `blend_seams()` → smooths patch boundaries

Also generates 3 comparison reconstructions:
- **MPS baseline**: reconstruct patch from compressed MPS directly (no VQC/decoder)
- **Zero latent**: decoder(zeros, zeros) — what the decoder outputs with no information
- **Random latent**: decoder(randn, positions) — decoder with random quantum features

### Step 9 — Order Parameter Tracking (per epoch)
`src/training/order_parameters.py`
- At each training step, records `mean_order_parameter = mean(<Z_i>)` across all qubits and patches
- Susceptibility = |Δ(mean_order)| between consecutive steps
- `detect_phase_transition()`: finds the epoch where susceptibility peaks sharply
  - Threshold = median + 2×std of susceptibility signal
  - Returns: `critical_epoch`, `max_susceptibility`, `detected` bool

---

## Output Files Saved

All saved under `outputs/training_analysis/` (or per-variant subfolder if `model_variant="both"`):

| File | What it is |
|------|-----------|
| `reconstructions.png` | 5-panel: Original / Quantum Recon / MPS Baseline / Random / Zero latent |
| `training_curves.png` | Total loss, reconstruction loss, energy loss vs step |
| `observables.png` | Heatmaps of 5 observable groups (Z, X, ZZ, XX, YY) across 16 patches |
| `entropy_map.png` | 4×4 grid of mean entanglement entropy per patch position |
| `metrics.json` | Final losses, mean/std energy, phase transition summary |
| `quantum_reconstruction.npy` | Final reconstructed image array [256,256] |
| `mps_baseline.npy` | MPS-only reconstruction [256,256] |
| `observables.npy` | Raw observable matrix [16,27] |
| `hvk_epoch_reconstruction_table.csv` | Per-epoch metrics: loss, order parameter, susceptibility |
| `hvk_epoch_correlation_table.csv` | Per-epoch per-patch Pauli correlations (ZZ, XX, YY per bond) |
| `hvk_order_parameter_curve.png` | Order parameter vs training step |
| `phase_transition_epoch_vs_order_parameter.gif` | Animated: susceptibility + order parameter evolving |
| `phase_transition_order_parameter_reconstruction.gif` | 3-panel GIF: reconstruction + order param + susceptibility per epoch |
| `hvk_order_parameter_phase_transition.gif` | Order parameter signal animation |
| `hvk_reconstruction_phase_transition.gif` | Reconstruction-only animation across epochs |
| `phase_transition_frames/` | Individual PNG frames (one per tracked epoch) |

If `model_variant="both"`, also saves at the base output dir:
| File | What it is |
|------|-----------|
| `hvk1d_standard_vs_symmetric.png` | 4-panel: Original / Standard Recon / Symmetric Recon / Abs Difference |
| `hvk1d_standard_vs_symmetric_metrics.json` | Final losses for both variants + phase transition for both |

---

## Default Hyperparameters

| Parameter | Default | CLI flag |
|-----------|---------|---------|
| image_size | 256 | --image-size |
| patch_size | 64 | --patch-size |
| steps | 120 | --steps |
| lr | 0.003 | --lr |
| positional_dim | 8 | --positional-dim |
| model_variant | "both" | --model-variant |
| device | "auto" | --device |
| epoch_frame_interval | 1 (every step) | --epoch-frame-interval |

Config can also be set via JSON file: `Main/src/config/training_config.json`

---

## Data Flow Summary

```
monalisa.jpg [256×256]
    │ extract_patches()
    ▼
16 patches [64×64] + positions [16×2]
    │ extract_mps_features()          (fixed, not trained)
    ▼
MPS features [16×46]  →  normalize  →  [16×46]
    │                                        │
    │ sinusoidal_positional_encoding()       │
    ▼                                        │
positions [16×8] ─────────────────────────► │
    │                                        │
    ▼                                        ▼
QuantumModel (VQC + Heisenberg H)  ← features [16×46]
    │
    ▼
observables [16×27] + energies [16]
    │
    │ concat with positions [16×8]
    ▼
PatchDecoder MLP → patches [16,1,64,64]
    │
    ▼
stitch + blend_seams → reconstructed image [256×256]
```

*Last updated: 2026-07-02*
