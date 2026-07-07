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

**Before running Exp 3 and 4:**
3. Update both freeze-experiment specs to include the optimizer filter line.
4. Run Exp 4 (freeze quantum) first — it is the most direct test of the weak claim and requires the smallest code change.
5. Run Exp 3 (freeze classical) second.

**Before running Exp 5:**
6. Update the classical replacement spec to address the energy loss path.
7. Run Exp 5 last — it requires the most structural change to the model.

**After all five are run:**
8. Compile a single comparison table with MSE, PSNR, SSIM for all five experiments plus the shared baseline.
9. Based on results, consult the decision tree in `PROJECT_STUDY/4_experiments_todo.md` to determine whether the weak or strong claim can be made.
