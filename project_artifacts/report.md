# Quantum Contribution Ablation — Experiment Report

**Date:** 2026-07-07
**Branch:** `experiments_ablation_study`
**Central question:** Are the classical projections and decoder doing all the learning, or does the quantum circuit contribute meaningful patch-specific structure?

---

## Status Overview

| # | Experiment | Status | Result |
|---|---|---|---|
| 1 | Shuffle observables at eval | Done | Observables are load-bearing |
| 2 | Zero latent + real positions | Done | Position memorization ruled out |
| 3 | Freeze classical, train quantum only | Not run — code not written | — |
| 4 | Freeze quantum, train classical only | Not run — code not written | — |
| 5 | Replace VQC with classical tanh | Not run — code not written | — |

---

## Exp 1 — Shuffle Observables at Eval

**Question:** Does the decoder use observable values, or does it reconstruct from position alone?

**How it works:** Train normally. At eval, randomly permute the 16 observable vectors across patches while keeping positional encodings fixed. If the decoder ignores observables, reconstruction quality should not change.

### Results

| Reconstruction | MSE | PSNR |
|---|---:|---:|
| Normal HVK | 0.000602 | 32.20 dB |
| Shuffled observables | 0.010727 | 19.70 dB |
| Degradation | ×17.8 worse | −12.5 dB |

### Interpretation

A −12.5 dB PSNR drop is a large, clean signal. The decoder cannot reconstruct the correct patch content when the wrong observable vector is fed for a given position. Observables carry patch-specific information; the decoder is using them.

### Conclusion

**Observables are load-bearing.** Proceed to Exp 2.

### What to improve

- The shuffle is a single random permutation over 16 elements. With 16 patches there is a small probability some patches land in their original position by chance. Report the number of fixed points in the permutation, or average results over 5–10 different permutations and report mean ± std PSNR. This strengthens the claim statistically.
- SSIM was not reported. Add SSIM alongside MSE and PSNR for all eval experiments — image quality metrics that agree across multiple measures are more convincing in a paper.

---

## Exp 2 — Zero Latent with Real Positions

**Question:** Can the decoder reconstruct the image from positional encoding alone, with no observable input?

**How it works:** Run the trained model normally. At eval, replace all observable vectors with zeros but keep the real positional encodings. If the decoder memorized "patch at position X looks like Y" during training, it will still reconstruct well.

### Results

| Reconstruction | MSE | PSNR |
|---|---:|---:|
| Normal HVK | 0.000621 | 32.07 dB |
| Zero observables + real positions | 0.029463 | 15.31 dB |
| Random latent + real positions | 0.071579 | 11.45 dB |
| Degradation (zero) | ×47.4 worse | −16.8 dB |

### Interpretation

A −16.8 dB drop when observables are zeroed rules out position memorization. The decoder did not learn to map position → pixel content independently of the quantum features. The random latent result (11.45 dB) serves as a noise floor — it confirms the decoder is not producing meaningful output without real observables.

### Conclusion

**Position memorization is not occurring.** Both Exp 1 and Exp 2 together establish that the quantum observable vectors are carrying real patch-specific content into the decoder. This is the minimum sanity check the paper needs before any stronger claim can be made.

### What to improve

- The `run_eval_controls.sh` script at `experiments/quantum_contribution/run_eval_controls.sh` only invokes Exp 2. It does not run Exp 1. Fix: add the `--shuffle-observables-at-eval` invocation to the script so both eval controls run together.
- Report SSIM.
- The two "normal HVK" rows in Exp 1 and Exp 2 come from separate runs and give slightly different PSNR values (32.20 vs 32.07). Both experiments should use the same baseline run so comparisons are on identical footing. Run a single baseline once, save its outputs, then run each eval variant against the same trained model checkpoint.

---

## Exp 3 — Freeze Classical, Train Quantum Only

**Question:** Can 51 quantum parameters (36 VQC weights + 15 Jx/Jy/Jz) learn a useful reconstruction when the projections and decoder are frozen at random initialization?

**Branch:** `ablation/freeze-classical`
**Status:** Spec written, no code implemented, not run.

### What the code change must do

In `Main/src/training/training.py`, before the optimizer is constructed (line 142), add:

```python
for param in decoder.parameters():
    param.requires_grad_(False)
model.feature_projection.requires_grad_(False)
model.position_projection.requires_grad_(False)
```

Then change the optimizer line to exclude frozen parameters:

```python
optimizer = optim.Adam(
    [p for p in list(model.parameters()) + list(decoder.parameters()) if p.requires_grad],
    lr=lr,
)
```

**Critical:** `requires_grad_(False)` alone is not enough. If the optimizer is constructed before freezing, or if it is constructed over all parameters without filtering, Adam will hold moment estimates for every parameter and may still apply updates due to numerical state. The optimizer must be built over only the trainable parameters.

### What to look for

- If reconstruction quality is comparable to the full model → quantum parameters alone are sufficient → strong claim supported.
- If reconstruction quality is poor (PSNR << 32 dB) → 51 quantum params cannot carry the full learning burden without classical support.

Expected outcome given architecture: likely poor, because the decoder is a 1.09M-parameter MLP initialized randomly — it cannot produce meaningful images without training. This experiment tests an extreme case. The scientifically interesting result is the degree of degradation, not a binary pass/fail.

---

## Exp 4 — Freeze Quantum, Train Classical Only

**Question:** Can the classical projections and decoder learn a good reconstruction while the VQC weights are fixed at random initialization?

**Branch:** `ablation/freeze-quantum`
**Status:** Spec written, no code implemented, not run.

### What the code change must do

In `Main/src/training/training.py`, before optimizer construction (line 142), add:

```python
model.weights.requires_grad_(False)
model.Jx.requires_grad_(False)
model.Jy.requires_grad_(False)
model.Jz.requires_grad_(False)
```

Then filter the optimizer identically to Exp 3:

```python
optimizer = optim.Adam(
    [p for p in list(model.parameters()) + list(decoder.parameters()) if p.requires_grad],
    lr=lr,
)
```

### What to look for

- If performance matches the full trained model → trained quantum weights add nothing; the classical decoder + random fixed VQC is sufficient → weak claim fails. This would be a serious problem for the paper.
- If performance is meaningfully worse → trained quantum weights matter → supports the weak claim (quantum + classical > classical alone).

This is the most important of the three remaining experiments. Run it first among Exp 3–5.

---

## Exp 5 — Replace VQC with Classical tanh MLP

**Question:** Does the quantum circuit provide something a matched classical layer cannot? This is the cleanest quantum-vs-classical comparison.

**Branch:** `ablation/classical-replacement`
**Status:** Spec written, no code implemented, not run.

### What the code change must do

In `Main/src/quantum/quantum_model.py`:

**`__init__`:** Add `self.classical_map = nn.Linear(6, 27)` and remove `self.weights`, `self.Jx`, `self.Jy`, `self.Jz`.

**`forward`:** Replace the VQC call and energy computation:

```python
# Replace lines 27–41 of quantum_model.py with:
for feature_vector, position_vector in zip(projected_features, projected_positions):
    combined = feature_vector + position_vector
    output = torch.tanh(self.classical_map(combined))
    observables.append(output)
    energies.append(torch.tensor(0.0, device=output.device))
```

**Critical:** The current `forward()` computes Heisenberg energy using `self.Jx`, `self.Jy`, `self.Jz` and the ZZ/XX/YY slices of the VQC output (lines 28–40 of `quantum_model.py`). When the VQC is removed, those parameters and slices no longer exist. The energy term must either be zeroed out (as shown above) or removed entirely. Failing to handle this will cause a `NameError` or index error at runtime.

Also check `training.py` line 172: `loss = reconstruction_loss + 0.01 * energy_loss`. With a zeroed energy tensor this line is safe but contributes nothing — you may want to note in the branch commit that the energy term is inactive.

### What to look for

- Full model (VQC) PSNR > classical tanh PSNR → quantum nonlinearity or entanglement provides representational advantage → strong claim supported.
- Classical tanh matches or beats VQC → the quantum circuit is not contributing computational value beyond what a linear map + nonlinearity can provide.

Note that `Linear(6→27) + tanh` has 27×6 + 27 = 189 parameters, versus 36 VQC weights. The classical replacement has more parameters. For a fair comparison, also consider reporting results with the classical map's bias removed (189 → 162 params) or with an explicitly parameter-matched classical layer (36 weights only).

---

---

## Next 5 Experiments — Beyond the Core

These address questions the first five leave entirely unanswered. They are ordered by scientific priority.

---

### Exp 6 — Generalization to a Second Image

**Question:** Does the model learn anything transferable, or has it memorized one image?

**Why it matters:** Every experiment so far trains and evaluates on the same single image. The VQC weights, projections, and decoder are all optimized for 16 patches of that one image. If the trained model produces noise-level output on a different image, the paper's claim shifts from "a model that learns image structure via quantum observables" to "a per-image optimization procedure." That is a fundamentally different — and much weaker — contribution. This must be resolved before submission.

**Branch:** `ablation/generalization`

**What to do:** Take the trained model checkpoint from the baseline run. Without any further training, run inference on a second image of the same size. Measure MSE, PSNR, SSIM against that image.

**Code change:** Minimal. In `main.py` or `training.py`, add a flag `--eval-only-image <path>` that skips the training loop and runs only the eval block on a new image using a loaded checkpoint. Alternatively, write a short standalone script:

```python
# load checkpoint
model.load_state_dict(torch.load("checkpoint.pt"))
decoder.load_state_dict(torch.load("decoder.pt"))
# build_dataset with new image_path, same patch_size/positional_dim
# run eval block, measure metrics
```

**What to look for:**
- PSNR on new image close to baseline (32 dB) → model has learned a generalizable representation → strong result.
- PSNR near random-latent floor (11 dB) → model memorized one image → paper framing must change.

**Note:** Checkpoint saving must be added to `training.py` if not already present. Check whether `save_analysis_outputs` saves model weights.

---

### Exp 7 — Energy Loss Ablation

**Question:** Does the Heisenberg energy regularizer actually improve reconstruction, or is it inert?

**Why it matters:** The energy term `0.01 * energy_loss` is the physics motivation of the paper — it connects the VQC to a physical Hamiltonian (Jx XX + Jy YY + Jz ZZ). If removing it leaves PSNR unchanged, the energy term is not doing anything for reconstruction quality. That means the Hamiltonian physics is decorative, which undermines the paper's narrative. Conversely, if removing it hurts reconstruction, the energy term is genuine regularization and the physics connection is real.

**Branch:** `ablation/no-energy-loss`

**Code change:** In `training.py` line 172, replace:

```python
loss = reconstruction_loss + 0.01 * energy_loss
```

with:

```python
loss = reconstruction_loss
```

**What to look for:**
- PSNR without energy loss ≈ baseline → energy term contributes nothing to reconstruction → physics motivation needs reframing.
- PSNR without energy loss < baseline → energy regularization helps → Hamiltonian physics is load-bearing → this is the result the paper needs.

**Also check:** Whether Jx/Jy/Jz converge to meaningful coupling values in the baseline run. If they stay near their `0.1 * randn` initialization throughout training, the energy loss is not shaping them.

---

### Exp 8 — MPS Features vs Simple Patch Statistics

**Question:** Does the tensor-network MPS compression contribute anything, or would a simpler feature extractor work just as well?

**Why it matters:** The pipeline is patch → MPS → feature vector → VQC → observables → decoder. All five core experiments test the VQC and decoder. None test the MPS encoder. If replacing MPS with mean + std of pixel values (2 numbers), or a small flattened downsampled version of the patch, gives the same PSNR, then the tensor-network step is not contributing. This would be a significant weakness: MPS is computationally expensive and is the primary motivation for using tensor networks at all.

**Branch:** `ablation/no-mps`

**Code change:** In `Main/src/tensornetworks/mps_features.py`, create an alternative feature extractor that computes simple statistics from the raw patch instead of MPS singular values. The output must have the same dimension as the current MPS feature vector so that the rest of the pipeline is unchanged.

Example: if the current MPS feature vector has dimension 46, replace it with `[mean, std, min, max, 10th pct, 25th pct, 75th pct, 90th pct, ...]` computed directly from pixel values, padded to 46 dimensions.

**What to look for:**
- Simple stats ≈ MPS features in PSNR → MPS compression is not adding information the VQC couldn't get from raw statistics → tensor network step is unjustified.
- MPS features > simple stats → MPS is capturing structure (entanglement entropy, bond correlations) that simple statistics miss → tensor network is genuinely useful.

---

### Exp 9 — Remove Entanglement Only (No CNOT Ring)

**Question:** Does qubit-qubit entanglement specifically contribute, or do per-qubit rotations alone explain the performance?

**Why it matters:** Exp 5 compares the full VQC against a classical linear layer — but those differ in many ways simultaneously (entanglement, quantum measurement, nonlinearity, parameter count). The most scientifically precise claim a quantum ML paper can make is that entanglement specifically provides an advantage. This experiment isolates that variable: keep everything else identical but remove only the CNOT entangling gates.

**Branch:** `ablation/no-entanglement`

**Code change:** In `Main/src/quantum/circuit.py`, replace `qml.StronglyEntanglingLayers` with per-qubit rotations only:

```python
# Replace:
qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

# With:
for layer in range(n_layers):
    for qubit in range(n_qubits):
        qml.Rot(weights[layer, qubit, 0],
                weights[layer, qubit, 1],
                weights[layer, qubit, 2],
                wires=qubit)
```

This keeps the same 36 weight parameters and the same circuit depth, but removes all two-qubit gates. The qubits evolve independently — no entanglement.

**What to look for:**
- No-entanglement VQC ≈ full VQC → entangling gates contribute nothing; the paper cannot claim an entanglement advantage.
- No-entanglement VQC < full VQC → entanglement is specifically helpful → this is the sharpest quantum advantage result possible in this paper.

**Note:** This is a stronger and more targeted test than Exp 5. Even if Exp 5 shows VQC > classical tanh, a reviewer will ask whether entanglement is the reason or just the rotation expressiveness. This experiment answers that follow-up directly.

---

### Exp 10 — Training Step Count Sweep

**Question:** Is 120 steps a well-chosen operating point, or is the model underfit or overfit?

**Why it matters:** The current setup trains for 120 steps on 16 patches with a 1.09M-parameter decoder. The shape of the loss curve determines whether all the experiments above are being run in a reasonable regime. If the model plateaus at step 40 and all experiments are compared at step 120, there is wasted compute and potentially misleading comparisons if different model variants plateau at different rates. If the model is still improving at step 120, the comparison is underfit. This also affects interpretation of the freeze experiments: if training quantum-only (Exp 3) needs more steps to converge than training the full model, a 120-step comparison is unfair.

**Branch:** None needed — run on main with different `--steps` values.

**Code change:** None. Run:

```bash
python Main/main.py --steps 30  --output-dir results/steps/030
python Main/main.py --steps 60  --output-dir results/steps/060
python Main/main.py --steps 120 --output-dir results/steps/120  # baseline
python Main/main.py --steps 240 --output-dir results/steps/240
python Main/main.py --steps 500 --output-dir results/steps/500
```

**What to look for:**
- Loss curve and PSNR should be reported at each checkpoint.
- Identify the step at which reconstruction loss plateaus — that is the true convergence point.
- If PSNR at step 240+ is meaningfully higher than at step 120, the baseline was underfit and all five core experiments should be re-run at the higher step count.
- If PSNR at step 120 ≈ step 240, the current 120-step setup is justified.

---

## Cross-Cutting Issues

### 1. No shared baseline checkpoint

Exp 1 and Exp 2 were run as separate training runs and show slightly different "normal HVK" PSNR values (32.20 vs 32.07). For Exp 3–5 the comparison will be even more fragile if each training run starts from a different random seed. 

**Fix:** Train the full model once, save the checkpoint, then run all eval variants and ablation training from the same starting point. Add `--seed 42` (or equivalent) to every run command and record the seed in each result folder.

### 2. SSIM missing from all results

Only MSE and PSNR are reported. SSIM is a perceptual metric that measures structural similarity and is standard in image reconstruction papers. All result JSON files should include it.

### 3. Optimizer filter is absent from both freeze-experiment specs

Both Exp 3 and Exp 4 spec files describe which parameters to freeze but do not include the optimizer filter. A student implementing from the spec alone will build the optimizer over all parameters before freezing, which means Adam accumulates moment state for frozen params and the training loop is technically wrong. The fix is one line but must be explicit in the spec.

### 4. Classical replacement breaks energy loss

Exp 5 removes `Jx`/`Jy`/`Jz` but the spec does not mention that the energy computation in `forward()` and the `0.01 * energy_loss` term in `training.py` both reference those parameters. The spec must be updated to address this before the student attempts the implementation.

---

## Recommended Next Steps

**Immediate (no retraining required):**
1. Fix `run_eval_controls.sh` to run both Exp 1 and Exp 2.
2. Re-run both eval experiments from a single saved baseline with a fixed seed; recompute with SSIM included.
3. Run Exp 10 (step count sweep) — no code change, just different `--steps` values. Establishes whether 120 steps is the right operating point before any further training experiments.

**Before running Exp 3 and 4:**
4. Update both freeze-experiment specs to include the optimizer filter line.
5. Run Exp 4 (freeze quantum) first — most direct test of the weak claim.
6. Run Exp 3 (freeze classical) second.

**Before running Exp 5:**
7. Update the classical replacement spec to address the energy loss path.
8. Run Exp 5 (classical tanh replacement).

**In parallel with Exp 3–5:**
9. Run Exp 7 (energy loss ablation) — one-line code change, high scientific value for the paper's physics narrative.
10. Run Exp 9 (no entanglement) — isolates entanglement contribution; a reviewer will ask for this regardless.

**After core results are stable:**
11. Run Exp 8 (MPS vs simple statistics) — tests the encoder tier; needed to justify the tensor network component.
12. Run Exp 6 (generalization to second image) — must be done before submission; add checkpoint saving to `training.py` first.

**Final:**
13. Compile a single comparison table with MSE, PSNR, SSIM for all ten experiments plus the shared baseline.
14. Based on results, consult the decision tree in `PROJECT_STUDY/4_experiments_todo.md` to determine whether the weak or strong claim can be made.
