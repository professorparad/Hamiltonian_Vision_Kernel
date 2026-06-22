# Summary_Main — Algorithm Map of `Main/`

The **Hamiltonian Vision Kernel (HVK)** pipeline in `Main/` is a hybrid
**MPS (Matrix Product State) + Variational Quantum Circuit (VQC)** image
autoencoder. A grayscale image is split into patches; each patch is compressed
into a tensor-network feature vector; a quantum circuit maps those features
(plus a positional encoding) to a vector of Pauli expectation values and a spin
energy; a classical decoder reconstructs each patch from the observables; the
patches are stitched and seam-blended back into a full image. Throughout
training, per-patch observables are reduced to **order parameters** whose
**susceptibility** is monitored to detect a **phase transition** in the model's
internal "spin" representation, and animated GIFs are produced.

Entry point: [`Main/main.py`](Main/main.py) → `run_hvk_analysis()` →
[`train()`](Main/src/training/training.py).

---

## 1. Top-level data-flow graph

```
                         Main/main.py
                  (build_run_config + run_hvk_analysis)
                              │  config dict
                              ▼
              src/training/training.py :: train()
                              │
        ┌─────────────────────┴──────────────────────────────────────┐
        ▼                                                            (loop)
  build_dataset()                                                      │
        │                                                             │
        ├─ image_loader.load_image_grayscale ── image (HxW, [0,1])    │
        ├─ patching.extract_patches ─────────── patches, positions    │
        ├─ mps_features.extract_mps_features ── features (per patch)  │
        │        └─ standardized (z-score over patches)               │
        ├─ positional_encoding.sinusoidal ───── positions → angles    │
        └─ targets = patches tensor                                   │
                              │                                        │
       features, positions, targets                                   │
                              ▼                                        │
        ┌──────────── TRAINING STEP (for step in range(steps)) ───────┘
        │   QuantumModel(features, positions) → observables, energies
        │        └─ circuit.VQC  (PennyLane QNode, 6 qubits, 2 layers)
        │   PatchDecoder(observables, positions) → predicted patches
        │   loss = MSE(pred, targets) + 0.01 * mean(energies)
        │   loss.backward(); optimizer.step()
        │
        │   every `epoch_frame_interval` steps (eval mode):
        │     ├─ reconstruct full image:
        │     │     stitch_patches → blend_seams
        │     ├─ TrainingDataGenerator.record(...)
        │     │     └─ order_parameters.compute_order_parameters(...)
        │     └─ phase_media.save_epoch_frame(...)  → frame PNG
        └─────────────────────────────────────────────────────────────
                              │ (after loop, eval mode)
                              ▼
        Final reconstructions:
          ├─ quantum_reconstruction (decoder on trained observables)
          ├─ mps_baseline           (mps_reconstruction.mps_reconstruct)
          ├─ random_latent          (decoder on randn observables)
          └─ zero_latent            (decoder on zeros)
                              │
                              ▼
        order_parameters.detect_phase_transition(epoch_rows)
                              │
                              ▼
        save_analysis_outputs(...)  → PNGs, NPYs, CSVs, GIFs, metrics.json
                              │
                              ▼
        outputs dict  ──►  main.py prints JSON summary
```

**Module roles by stage**

| Stage | Module(s) |
|-------|-----------|
| Config / CLI | `main.py`, `training.py` (`load_config`, `resolve_path`) |
| Preprocessing | `preprocessing/image_loader.py`, `preprocessing/patching.py`, `preprocessing/positional_encoding.py` |
| Tensor-network features | `tensornetworks/mps_features.py` |
| Quantum core | `quantum/circuit.py`, `quantum/quantum_model.py` |
| Decoder | `decoder/patch_decoder.py` |
| Reconstruction | `reconstruction/patch_stitching.py`, `reconstruction/seam_bleading.py`, `tensornetworks/mps_reconstruction.py` |
| Physics analysis | `training/order_parameters.py`, `training/data_generator.py` |
| Media / GIFs | `training/phase_media.py` |
| Static plots (optional) | `visualization/*` |

---

## 2. Module-by-module detail

### 2.1 `main.py` — orchestration & CLI
- `build_run_config()`: starts from a hard-coded default config (image size 256,
  patch 64, positional dim 8, 120 steps, lr 0.003, etc.), overlays a JSON config
  file via `load_config`, then applies CLI overrides. Resolves `image_path` and
  `output_dir` to absolute paths.
- `run_hvk_analysis()`: builds the config and calls `train(**config)`.
- `main()`: parses args, runs the analysis, then prints a JSON summary
  (final total/reconstruction/energy loss, phase-transition record, GIF paths,
  output dir).
- **Data in:** CLI args + JSON config. **Data out:** `(model, decoder, outputs, config)`.

### 2.2 `src/training/training.py` — the pipeline driver
This is the spine that wires every submodule together.

- **`resolve_device`**: `"auto"` → CUDA if available else CPU.
- **`build_dataset`** (sub-stage data flow):
  1. `load_image_grayscale` → `image` (HxW float in [0,1]).
  2. `extract_patches` → `patches` (N,patch,patch), `positions` (N,2) normalized
     row/col fractions; `raw_positions` kept for CSV labeling.
  3. For each patch, `extract_mps_features` → feature vector; stacked into
     `features` (N, F).
  4. **Standardization:** `features = (features - mean) / (std + 1e-8)` over the
     patch axis (zero-mean/unit-variance per feature).
  5. `sinusoidal_positional_encoding(positions, d_model=positional_dim)` →
     `positions` tensor (N, positional_dim).
  6. `targets` = patches as a (N,1,patch,patch) float tensor.
  Returns image, patches, raw_positions, features, positions, targets (last three on device).
- **`train`** (core loop):
  - Builds `QuantumModel` (input dims from `features`/`positions` shapes) and
    `PatchDecoder` (`observable_dim` from circuit, positional dim, patch size).
  - One Adam optimizer over **both** model and decoder parameters.
  - Optionally constructs a `TrainingDataGenerator` and a `phase_transition_frames`
    directory (only when `save_outputs && output_dir && track_order_parameters`).
  - **Per step:** forward `model(features, positions) → (observables, energies)`;
    `decoder(observables, positions) → output`; `loss = MSE(output, targets) + 0.01*mean(energies)`;
    backprop; Adam step; record the three loss histories; log every 20 steps.
  - **Per tracked step** (`step % epoch_frame_interval == 0` or last): switch to
    eval, recompute observables/energies, decode, stitch + seam-blend a full
    reconstruction, call `data_generator.record(...)` to compute order parameters,
    then `save_epoch_frame(...)` for the GIF, and stash the frame path back into
    the last epoch row.
  - **After loop:** eval mode produces four reconstructions — quantum (trained
    observables), `mps_baseline` (pure MPS reconstruction per patch, no learning),
    `random_latent` (decoder on `randn` observables), `zero_latent` (decoder on
    zeros; positions zeroed unless `zero_latent_uses_positions`). All four are
    stitched + blended.
  - Assembles `outputs` dict (images, observables, energies, positions, loss
    history, epoch rows, and `detect_phase_transition(...)`).
  - If saving, calls `save_analysis_outputs`. If `show_plots`, calls the
    `visualization/*` functions.
- **`save_analysis_outputs`**: writes `reconstructions.png`, `training_curves.png`,
  `observables.png`, `entropy_map.png`; if a data generator exists, writes the two
  CSVs and the order-parameter curve + (optionally) four GIFs; saves
  `quantum_reconstruction.npy`, `mps_baseline.npy`, `observables.npy`; writes
  `metrics.json` (final losses, mean/std energy, phase-transition record, media paths).
- Helper plotters (`save_reconstruction_plot`, `save_training_curve`,
  `save_observable_plot`, `save_entropy_plot`) are the file-saving twins of the
  interactive `visualization/*` functions. `get_entropy_features` slices feature
  columns ≥ 35 (the bond-entropy block). `load_config`/`resolve_path` handle config.

### 2.3 Preprocessing

- **`image_loader.load_image_grayscale`**: OpenCV reads the image as grayscale,
  resizes to `size`, casts float32, divides by 255 → values in [0,1]. Raises if
  the file is missing.
- **`patching.extract_patches`**: validates the image is 2D and divisible by
  `patch_size`; tiles it into non-overlapping `patch_size × patch_size` blocks in
  row-major order; records each patch's normalized top-left position
  `[i/height, j/width]`. Returns `(patches, positions)`.
- **`positional_encoding`**: `sinusoidal_positional_encoding` delegates to
  `sinusodial_encoding`. Requires `d_model % 4 == 0`. For `n_bands = d_model//4`
  log-spaced frequencies, each 2D position `(x,y)` is encoded as
  `[sin(2π·x·f), cos(2π·x·f), sin(2π·y·f), cos(2π·y·f)]` per band → a
  (N, d_model) tensor. This is a Fourier-feature encoding of patch location.

### 2.4 Tensor-network feature extraction — `tensornetworks/mps_features.py`
Per patch (`n_sites=12`, `bond_dim=4`):
1. Flatten the patch and **L2-normalize** it into a quantum state vector
   (requires `patch_size² == 2**n_sites`, i.e. 64×64 = 4096 = 2¹²).
2. Reshape to a rank-12 tensor and build an exact MPS via
   `quimb`'s `MatrixProductState.from_dense`.
3. `compress(max_bond=bond_dim)` truncates to bond dimension 4 (lossy
   compression that retains dominant entanglement), then `normalize()`.
4. **Features extracted:**
   - Local `⟨Z⟩` and `⟨X⟩` at each of the 12 sites (via `local_expectation_values`).
   - Nearest-neighbor `⟨Z_i Z_{i+1}⟩` for 11 bonds (via `two_site_expectation`).
   - **Entanglement entropy** at each of the 11 internal bonds: from the Schmidt
     values, `p = λ²` (normalized), `S = -Σ p log p`.
   Concatenated into one float32 vector per patch (the entropy block is the
   trailing columns that `get_entropy_features` later slices for the entropy map).

`local_expectation_values` / `two_site_expectation` copy the MPS, apply Pauli
gate(s) at the target site(s), and take `⟨ψ|gated ψ⟩` (real part).

### 2.5 Quantum core

- **`quantum/circuit.py`** — defines the VQC as a PennyLane QNode.
  - Globals: `n_qubits=6`, `n_layers=2`, `n_bonds=5`,
    `observable_dim = n_qubits + n_qubits + 3*n_bonds = 6+6+15 = 27`.
  - `VQC(inputs, positional_angles, weights)`:
    1. `AngleEmbedding(inputs)` — encodes the 6 projected features as rotation angles.
    2. Per-qubit `RY(positional_angles[q])` — injects positional information.
    3. `StronglyEntanglingLayers(weights)` — the trainable entangling ansatz.
    4. Returns 27 expectation values: 6×⟨Z⟩, 6×⟨X⟩, 5×⟨ZZ⟩, 5×⟨XX⟩, 5×⟨YY⟩
       (a 1D chain of nearest-neighbor correlators).
- **`quantum/quantum_model.py`** — `QuantumModel(nn.Module)`:
  - `feature_projection`: Linear `feature_dim → 6` (compresses MPS features to qubit count).
  - `position_projection`: Linear `positional_dim → 6` (positional encoding → 6 RY angles).
  - Trainable parameters: `weights` (2,6,3) for the ansatz; coupling vectors
    `Jx, Jy, Jz` (5 each) defining a **Heisenberg-like spin Hamiltonian** over the bonds.
  - `forward`: projects features and positions, then loops over patches. For each,
    runs `VQC` to get the 27 observables, slices out ZZ/XX/YY, and computes
    `energy = Σ Jz·ZZ + Σ Jx·XX + Σ Jy·YY` (the expectation of the learned spin
    Hamiltonian). Stacks observables (N,27) and energies (N,). In **training mode**
    adds Gaussian noise (`0.01·randn`) to observables — a regularizer / exploration term.

### 2.6 Decoder — `decoder/patch_decoder.py`
`PatchDecoder(nn.Module)`: input dim = `observable_dim + positional_dim` (27 + 8 = 35
by default). MLP: `Linear(→128) → ReLU → Linear(→256) → ReLU → Linear(→patch²) → Sigmoid`.
`forward` concatenates observables and positional encoding, runs the MLP, reshapes
to (N,1,patch,patch) in [0,1]. Maps the quantum "latent" + position back to pixels.

### 2.7 Reconstruction

- **`reconstruction/patch_stitching.stictch_patches`**: writes each predicted patch
  back into its row-major tile of a blank `image_size × image_size` canvas.
- **`reconstruction/seam_bleading.blend_seams`**: builds a per-pixel weight map that
  dips toward 0 within `blend_width` of every patch seam, computes a separable
  `box_blur` of the image, and blends original/blurred by the weight to hide seam
  discontinuities; clips to [0,1].
- **`tensornetworks/mps_reconstruction.mps_reconstruct`**: the **classical TN
  baseline** — same MPS compression as feature extraction, but reconstructs the
  patch by `mps.to_dense()` rescaled by the original norm. No learning; isolates
  how much information bond-dim-4 MPS alone preserves.

### 2.8 Physics analysis

- **`training/order_parameters.py`**:
  - `observable_slices`: splits the 27-vector into z, x, zz, xx, yy blocks.
  - `compute_order_parameters`: per patch, `order = mean(Z)` (magnetization-like
    **order parameter**), `transverse_order = mean(X)`, `lattice_correlation =
    mean(ZZ+XX+YY)/3`. Returns means across patches plus
    `order_parameter_susceptibility = |mean_order − previous_mean_order|`
    (discrete time-derivative of the order parameter).
  - `detect_phase_transition`: over all epoch rows, finds the susceptibility peak,
    sets `threshold = median + 2·std`; a transition is **detected** if the peak
    exceeds threshold and is > 0. Reports `critical_epoch`, the order-parameter
    jump there, and a human-readable proof string.
- **`training/data_generator.py`** — `TrainingDataGenerator`:
  - Defines the schemas for two CSVs: an **epoch table** (per-epoch order
    parameters / losses) and a **correlation table** (per-patch ZZ/XX/YY plus
    derived quantities).
  - `record(...)`: computes order parameters for the epoch (tracking the previous
    mean order for susceptibility), appends an epoch row, and expands per-patch
    `build_correlation_rows`.
  - `write_csvs`: emits `hvk_epoch_reconstruction_table.csv` and
    `hvk_epoch_correlation_table.csv`.

### 2.9 Media — `training/phase_media.py`
All functions are matplotlib→PIL GIF/PNG builders (no-ops if PIL missing or rows empty):
- `save_epoch_frame`: side-by-side original vs reconstruction PNG titled with loss
  and order parameter (these become the reconstruction-GIF frames).
- `save_order_parameter_plot`: static two-panel order/energy + susceptibility figure.
- `save_order_parameter_gif`: animated two-panel order + susceptibility, drawing a
  critical-epoch line once reached.
- `save_phase_transition_order_parameter_gif`: animated single-panel order-vs-epoch
  with the critical-epoch marker.
- `save_merged_phase_transition_gif`: the headline animation — reconstruction frame
  plus growing order and susceptibility curves, a detection-threshold line, and a
  proof caption.
- `save_frames_as_gif`: assembles existing reconstruction frames into a GIF.

### 2.10 Visualization (interactive, only when `--show-plots`)
`visualization/reconstruction_plots.py`, `entropy_maps.py`, `observable_plots.py`,
`training_curve.py` are `plt.show()` versions of the panels that
`save_analysis_outputs` writes to disk: the 5-panel reconstruction comparison, the
patch entropy heatmap, the five observable maps, and the loss curves.

---

## 3. Final results produced
Written under `output_dir` (default `outputs/training_analysis/`):
`reconstructions.png`, `training_curves.png`, `observables.png`, `entropy_map.png`,
`hvk_order_parameter_curve.png`, up to four GIFs (reconstruction, order-vs-epoch,
order+susceptibility, merged proof), `*.npy` arrays, the two CSV tables, and
`metrics.json`. `main.py` prints final losses, the phase-transition record, and GIF paths.
