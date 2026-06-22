# Summary_Main2 — Algorithm Map of `Main2/`

`Main2/` is a **modular refactor and 2D-grid variant** of the Hamiltonian Vision
Kernel. It keeps the same overall recipe — patch the image, extract MPS features,
run a VQC per patch, decode patches with an MLP, stitch + seam-blend, then track
order parameters and detect a phase transition — but changes two things:

1. **Quantum topology.** Where `Main/` uses a 1D qubit chain with `StronglyEntanglingLayers`
   and a Heisenberg (Jx/Jy/Jz) energy, `Main2/` arranges the **6 qubits as a 2×3
   grid** with explicit horizontal/vertical CNOT edges and a purely **ZZ
   Ising-like energy** over those edges.
2. **Code organization.** The monolithic `training.py` of `Main/` is split into
   focused modules: `config`, `dataset`, `model`, `analysis`, `outputs`, `pathing`,
   `training`. **Preprocessing, MPS features, stitching, seam-blending, and the
   phase-media GIF builders are reused directly from `Main/`** (imported via
   `pathing.add_main_package_to_path`).

Entry point: [`Main2/main.py`](Main2/main.py) → [`run_main2(config)`](Main2/src/training.py).

---

## 1. Top-level data-flow graph

```
                         Main2/main.py
              (parse args → Main2Config → run_main2)
                              │  Main2Config
                              ▼
                  src/training.py :: run_main2()
                              │
        ┌─────────────────────┴───────────────────────────────────────┐
        ▼                                                            (loop)
  dataset.build_main2_dataset()                                        │
        │   (REUSES Main/ preprocessing + MPS features)                │
        ├─ image_loader.load_image_grayscale ── image (HxW, [0,1])     │
        ├─ patching.extract_patches ─────────── patches, raw_positions │
        ├─ mps_features.extract_mps_features ── features → z-scored     │
        ├─ positional_encoding.sinusoidal ───── positions → angles     │
        └─ targets = patches tensor                                    │
                              │                                         │
       features, positions, targets, image, raw_positions             │
                              ▼                                         │
        ┌──────────── TRAINING STEP (for step in range(steps+1)) ──────┘
        │   Quantum2DGridModel(features, positions) → observables, energies
        │        └─ model.quantum_grid_circuit (2×3 grid, CNOT edges, Rot)
        │   PatchDecoder(observables, positions) → predicted patches
        │   loss = MSE(pred, targets) + 0.01 * mean(energies)
        │   loss.backward(); optimizer.step()
        │
        │   every `frame_interval` steps (eval mode):
        │     ├─ reconstruct: stitch_patches → blend_seams   (REUSED Main/)
        │     ├─ outputs.save_reconstruction_frame(...)  → frame PNG
        │     ├─ analysis.main2_order_summary(...)       → epoch row
        │     └─ analysis.main2_correlation_rows(...)    → per-patch rows
        └──────────────────────────────────────────────────────────────
                              │ (after loop)
                              ▼
        analysis.detect_phase_transition(epoch_rows)
                              │
                              ▼
        outputs.write_csv ×2, outputs.save_order_curve
        outputs.save_gif (reconstruction)
        phase_media.save_phase_transition_order_parameter_gif   (REUSED Main/)
        phase_media.save_merged_phase_transition_gif            (REUSED Main/)
                              │
                              ▼
        result dict ──► main.py prints JSON summary
```

**Module roles (★ = reused from `Main/`)**

| Stage | Module(s) |
|-------|-----------|
| Config / CLI | `main.py`, `src/config.py` (`Main2Config`) |
| Path bootstrap | `src/pathing.py` |
| Preprocessing | ★ `Main/...image_loader`, `...patching`, `...positional_encoding` (via `src/dataset.py`) |
| Tensor-network features | ★ `Main/...mps_features` (via `src/dataset.py`) |
| Quantum core (2D grid) | `src/model.py` (`quantum_grid_circuit`, `Quantum2DGridModel`) |
| Decoder | `src/model.py` (`PatchDecoder`) |
| Reconstruction | ★ `Main/...patch_stitching`, `...seam_bleading` |
| Physics analysis | `src/analysis.py` |
| Outputs (CSV/PNG/GIF) | `src/outputs.py` |
| Phase-transition GIFs | ★ `Main/...phase_media` |

---

## 2. Module-by-module detail

### 2.1 `main.py` — CLI & orchestration
- Inserts `REPO_ROOT` and `Main/` onto `sys.path` so both `Main2.src.*` and the
  reused `src.*` (Main) packages import cleanly.
- `parse_args()`: builds defaults from a fresh `Main2Config()` and exposes
  `--image-path/--output-dir/--image-size/--patch-size/--positional-dim/--steps/
  --lr/--device/--frame-interval/--no-gif`.
- `main()`: assembles a `Main2Config` from args, calls `run_main2(config)`, then
  prints a JSON summary (output dir, epochs recorded, the three GIF paths,
  phase-transition record).

### 2.2 `src/config.py` — `Main2Config` dataclass
Centralizes all run parameters. Defaults: image `Main/data/monalisa.jpg`, output
`Main2/outputs/training_analysis`, image 256, patch 64, positional dim 8,
**steps 200, lr 0.004** (note: higher than `Main/`'s 120 / 0.003), device `auto`,
`frame_interval 1`, `save_gif True`. `MAIN2_ROOT`/`REPO_ROOT` are derived from the
file location so paths are portable.

### 2.3 `src/pathing.py` — cross-package import bootstrap
`add_main_package_to_path()` puts `<repo>/Main` at the **front** of `sys.path`
(removing any stale copy first) so that `import src.preprocessing...` etc. resolve
to **`Main/`'s** `src` package. This is what lets `Main2` reuse `Main`'s
preprocessing, MPS, reconstruction, and phase-media code without duplication. It is
called at import time in `dataset.py`, `analysis.py` (indirectly via model import),
and `training.py`.

### 2.4 `src/dataset.py` — `build_main2_dataset` (reuses Main/ preprocessing)
Identical preprocessing path to `Main/`'s `build_dataset`:
1. `load_image_grayscale` → image in [0,1].
2. `extract_patches` → `patches`, `raw_positions` (normalized top-left fractions).
3. `extract_mps_features` per patch → feature matrix, then **z-scored** per feature
   (`(x-mean)/(std+1e-8)`).
4. `sinusoidal_positional_encoding(raw_positions, d_model=positional_dim)`.
5. `targets` = (N,1,patch,patch) tensor.
Returns a dict `{image, patches, raw_positions, features, positions, targets}`
with the tensors moved to `device`. (See Summary_Main §2.3–2.4 for the internals of
these reused functions.)

### 2.5 `src/model.py` — the 2D-grid quantum core, energy, and decoder
This is the substantive departure from `Main/`.

- **Globals:** `N_QUBITS=6`, `N_LAYERS=2`. Qubits are laid out as a **2×3 grid**:
  - `EDGES_H = [(0,1),(1,2),(3,4),(4,5)]` (horizontal neighbors),
  - `EDGES_V = [(0,3),(1,4),(2,5)]` (vertical neighbors),
  - `ALL_EDGES = EDGES_H + EDGES_V` (7 edges),
  - `OBS_DIM = 6 + 6 + 7 = 19` (vs 27 in `Main/`).
- **`quantum_grid_circuit(inputs, positional_angles, weights)`** (PennyLane QNode):
  1. `AngleEmbedding(inputs)` over the 6 qubits.
  2. Per-qubit `RY(positional_angles[q])` positional injection.
  3. For each layer: apply `CNOT` along **all horizontal then vertical edges**
     (the 2D entangling structure), then a general `Rot(α,β,γ)` per qubit from
     `weights[layer, qubit, :]`.
  4. Returns 19 expectations: 6×⟨Z⟩, 6×⟨X⟩, and **7×⟨Z_u Z_v⟩ over `ALL_EDGES`**
     (2D nearest-neighbor ZZ correlators — no XX/YY here, unlike `Main/`).
- **`Quantum2DGridModel(nn.Module)`**:
  - `feature_projection`: Linear `feature_dim → 6`; `position_projection`: Linear
    `positional_dim → 6`.
  - Trainable: `weights` (2,6,3) and `j_2d` (length-7 edge couplings).
  - `forward`: loops over patches, runs the grid circuit, takes the trailing 7
    ZZ values, and computes a **2D Ising energy** `energy = Σ j_2d · zz_2d`. Stacks
    observables (N,19) and energies (N,). Adds `0.01·randn` noise to observables in
    **training mode** (same regularizer trick as `Main/`).
- **`PatchDecoder(nn.Module)`**: MLP `Linear(OBS_DIM+positional_dim → 128) → ReLU →
  Linear(→256) → ReLU → Linear(→patch²) → Sigmoid`; concatenates observables +
  positions, outputs (N,1,patch,patch) in [0,1]. Functionally the same as `Main/`'s
  decoder but sized to `OBS_DIM=19`.

### 2.6 `src/analysis.py` — order parameters & phase detection
2D-aware versions of `Main/`'s order-parameter logic.
- **`main2_order_summary(observables, energies, previous_order)`**: slices the
  19-vector into Z (0:6), X (6:12), ZZ (12:). `order = mean(Z)` per patch;
  returns `mean_energy`, `mean_order_parameter`, `order_parameter_susceptibility`
  (`|mean_order − previous_order|`), `mean_abs_order_parameter`,
  `mean_transverse_order_parameter` (mean X), and
  `mean_total_lattice_correlation` (mean ZZ).
- **`main2_correlation_rows(...)`**: per-patch rows carrying energy, order /
  transverse / abs order, total lattice correlation, and one column per edge
  `corr_ZZ_{u}_{v}` (keyed by the actual 2D edge, e.g. `corr_ZZ_0_3`) — richer
  topology labeling than `Main/`'s linear `corr_ZZ_i`.
- **`detect_phase_transition(epoch_rows)`**: identical statistics to `Main/` —
  susceptibility peak, `threshold = median + 2·std`, `detected` if peak exceeds
  threshold and > 0, plus `critical_epoch`, order-parameter jump, and a proof string.

### 2.7 `src/outputs.py` — CSV / PNG / GIF writers
- `write_csv(path, rows)`: dict-rows → CSV (header inferred from the first row;
  no fixed schema, unlike `Main/`'s `TrainingDataGenerator`).
- `save_reconstruction_frame(original, reconstruction, epoch, dir)`: side-by-side
  original vs reconstruction PNG (the reconstruction-GIF frames).
- `save_order_curve(epoch_rows, path)`: two-panel order/energy + susceptibility plot.
- `save_gif(frame_paths, path)`: assembles reconstruction frames into a GIF (PIL).

### 2.8 `src/training.py` — `run_main2` (driver)
- Resolves device (reuses `Main/`'s `resolve_device`), builds the dataset, then the
  `Quantum2DGridModel` and `PatchDecoder` (input dims taken from the dataset shapes),
  and a single Adam optimizer over both.
- **Loop `for step in range(steps+1)`** (note: inclusive of `steps`):
  - Train forward/backward exactly as in the graph; log every 20 steps.
  - When `step % frame_interval == 0` or `step == steps`: eval-mode recompute,
    decode, **stitch + seam-blend (reused `Main/` reconstruction)**, save a
    reconstruction frame, then append one epoch row (`main2_order_summary`, carrying
    `previous_order` forward) and extend the correlation rows (`main2_correlation_rows`).
- **After loop:** `detect_phase_transition(epoch_rows)`; write
  `hvk_epoch_reconstruction_table.csv` and `hvk_epoch_correlation_table.csv`;
  `save_order_curve`. If `save_gif`: build the reconstruction GIF (`outputs.save_gif`)
  plus the two **reused `Main/` phase-media** animations
  (`save_phase_transition_order_parameter_gif`, `save_merged_phase_transition_gif`).
- Returns `{model, decoder, epoch_rows, correlation_rows, frame_paths, gif_path,
  order_gif_path, merged_gif_path, phase_transition}`.

---

## 3. Key differences vs `Main/` (at a glance)

| Aspect | `Main/` | `Main2/` |
|--------|---------|----------|
| Qubit topology | 1D chain | 2×3 grid (H+V CNOT edges) |
| Ansatz | `StronglyEntanglingLayers` | per-edge CNOTs + per-qubit `Rot` |
| Observables | 27 (Z,X,ZZ,XX,YY) | 19 (Z,X, ZZ over 7 edges) |
| Energy | Heisenberg `Jx·XX+Jy·YY+Jz·ZZ` | 2D Ising `Σ j_2d·ZZ` |
| Default steps / lr | 120 / 0.003 | 200 / 0.004 |
| Code shape | one large `training.py` | split modules + reuse of `Main/` |
| Extra reconstructions | mps/random/zero baselines | reconstruction frames only |

---

## 4. Final results produced
Under `Main2/outputs/training_analysis/` (default): the two CSV tables
(`hvk_epoch_reconstruction_table.csv`, `hvk_epoch_correlation_table.csv`),
`hvk_order_parameter_curve.png`, and (unless `--no-gif`) three GIFs —
`hvk_reconstruction_phase_transition.gif`,
`phase_transition_epoch_vs_order_parameter.gif`,
`phase_transition_order_parameter_reconstruction.gif`. `main.py` prints a JSON
summary with the output dir, epoch count, GIF paths, and the phase-transition record.
