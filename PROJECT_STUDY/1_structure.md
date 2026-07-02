# Project Structure

## Sub-Projects (5 total)

| Folder | What it is | Quantum? |
|--------|-----------|----------|
| `Main/` | Core HVK study — 1D qubit chain, Heisenberg H (Jx·XX+Jy·YY+Jz·ZZ), trains on full image | Simulator |
| `Main2/` | HVK2D study — 2D 2×3 qubit grid, ZZ only, CNOT+Rot gates | Simulator |
| `IBM_Cloud/` | Hardware validation on IBM Quantum (ibm_heron); runs circuits on real device | Real HW |
| `Baselines/` | Classical comparison suite (autoencoder, MLP, CNN, GAN, PHL) — no quantum | No |
| `python_library/` | Shared utilities imported by Main/ and tests | No |

Other folders: `tests/` (unit tests), `docs/` (GitHub Pages), `latex_outputs/` (paper), `PROJECT_STUDY/` (this knowledge base).

---

## Main/ — File Map

```
Main/
  main.py                          # Entry point; run_hvk_analysis(); supports model_variant="both"
  src/
    preprocessing/
      image_loader.py              # Load + grayscale image
      patching.py                  # extract_patches() → patches, normalized positions
      positional_encoding.py       # sinusoidal_positional_encoding() → torch.Tensor [N, 8]
    tensornetworks/
      mps_features.py              # MPS compression → 46-dim feature vector per patch
      mps_reconstruction.py        # Reconstruct patch from MPS (unused in main training loop)
    quantum/
      circuit.py                   # PennyLane QNode; 6 qubits, 2 layers; measures 27 Paulis
      quantum_model.py             # QuantumModel: Jx,Jy,Jz per bond; full Heisenberg H
      symmetric_model.py           # SymmetricQuantumModel: J(ZZ) + K(XX+YY); U(1) symmetry
    decoder/
      patch_decoder.py             # PatchDecoder MLP: (observable_dim + positional_dim) → patch pixels
    reconstruction/
      patch_stitching.py           # stitch_patches() → full image from patch grid
      seam_bleading.py             # seam blending for patch boundaries
    training/
      training.py                  # train() — main training loop; save_analysis_outputs()
      data_generator.py            # TrainingDataGenerator; writes epoch + correlation CSVs
      order_parameters.py          # compute_order_parameters(); detect_phase_transition()
      phase_media.py               # GIF generation for epoch animations
    visualization/
      entropy_maps.py              # Plot entanglement entropy maps
      observable_plots.py          # Plot Pauli observable distributions
      reconstruction_plots.py      # Plot original vs reconstructed image
      training_curve.py            # Plot loss vs epoch
```

---

## Main2/ — File Map

```
Main2/
  main.py          # Entry point
  src/
    config.py      # HVK2DConfig dataclass (steps, patch_size, bond_dim, etc.)
    dataset.py     # Load image, extract patches, positional encoding
    model.py       # HVK2D: 2×3 grid edges, CNOT+Rot circuit, OBS_DIM=19
    training.py    # Training loop — BUG: range(config.steps+1) runs 201 steps
    analysis.py    # Per-patch analysis rows — BUG: int() on normalized positions → always 0
    outputs.py     # Save PNGs, CSVs, JSONs
    pathing.py     # Path resolution helpers
```

---

## IBM_Cloud/ — File Map

```
IBM_Cloud/
  prepare_ibm_dataset.py           # Prepare Mona Lisa patches as .npz for IBM jobs
  run_ibm_hvk_probe.py             # Core IBM experiment: submit circuits, collect results
  run_ibm_epoch_probe.py           # Probe reconstruction error at different training epochs
  run_cross_quantum_validation.py  # Compare IBM HW vs statevector simulator latent features
  run_hardware_latent_validation.py# Validate latent space consistency on hardware
  plot_hardware_patch_map.py       # Plot which patches ran on which hardware qubits
  plot_reconstruction_error_vs_epoch.py  # Plot error vs epoch from probe results
  datasets/                        # monalisa_patches.npz
  outputs/                         # JSON/CSV/PNG results from hardware runs
```

---

## Baselines/ — File Map

```
Baselines/
  plot_mps_bond_dim_scaling.py     # Standalone: plot MPS bond dim vs reconstruction error
  cifar10_comparisons/
    main.py                        # Orchestrator: runs all methods via subprocess
    common.py                      # Shared metric helpers (MSE, SSIM, PSNR, LPIPS)
    download_cifar32.py            # Download + preprocess CIFAR-10 to 32×32 grayscale
    run_comprehensive_benchmark.py # Run all methods + aggregate
    plot_all_architecture_cifar_benchmark.py  # Final bar chart figure
    smoke_test.py                  # Quick sanity check
    autoencoder/run_autoencoder_cifar32.py
    cnn/run_cnn_cifar32.py
    gan/run_gan_cifar32.py
    mlp/run_mlp_cifar32.py
    phl/run_phl_cifar32.py         # PHL = Patch Hamiltonian Learning (custom method)
    hvk1d/run_hvk1d_cifar32.py    # HVK1D on CIFAR (quantum baseline)
    hvk2d/run_hvk2d_cifar32.py    # HVK2D on CIFAR
    symmetric_hvk1d/run_symmetric_hvk1d_cifar32.py
  monalisa_comparisons/
    main.py                        # Run same methods on Mona Lisa; aggregate metrics
```

---

## tests/ — File Map

```
tests/
  conftest.py
  test_preprocessing.py            # Tests: extract_patches, sinusoidal_positional_encoding
  test_decoder.py                  # Tests: PatchDecoder forward pass
  test_training_smoke.py           # Smoke test: 2-step training run end-to-end
  test_cifar_benchmark_helpers.py  # Tests: common.py metric functions
  utils/
    pathing.py                     # add_main_to_path() — adds Main/src to sys.path for tests
    test_mps_helpers.py            # MPS test utilities
```

---

## Key Numbers to Remember

| Parameter | Value |
|-----------|-------|
| Image | Mona Lisa 256×256 grayscale |
| Patch size (Main) | 64×64 |
| n_sites (MPS) | 12 (flattened 4096 → [2]^12) |
| bond_dim χ | 4 |
| MPS feature vector | 46-dim (24 local + 11 ZZ + 11 entropy) |
| n_qubits (VQC) | 6 |
| n_layers | 2 StronglyEntanglingLayers |
| Observables (HVK1D) | 27 (6Z + 6X + 5ZZ + 5XX + 5YY) |
| Observables (HVK2D) | 19 (6Z + 6X + 7ZZ) |
| Positional encoding dim | 8 |
| Decoder input | observable_dim + positional_dim |
| CIFAR patch size | 8×8, n_sites=6 |

---

*Last updated: 2026-07-02*
