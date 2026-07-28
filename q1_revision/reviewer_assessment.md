# External Reviewer Assessment (2026-07-27)

Overall Q1 readiness: 5.5/10.

| Criterion | Score | Assessment |
|---|---|---|
| Novelty | 7/10 | Interesting combination of MPS features, VQC observables, Hamiltonian diagnostics, positional encoding, symmetry pooling |
| Technical correctness | 6.5/10 | Generally careful, scoped claims, useful mathematical definitions |
| Experimental rigor | 6/10 | Good audits and controls, but sample sizes and several evidence blocks remain limited |
| Practical significance | 4.5/10 | The quantum representation does not improve ordinary held-out reconstruction |
| Quantum contribution | 5/10 | Six-qubit circuits and a deliberately constructed nonlocal task do not establish practical quantum benefit |
| Writing and presentation | 7/10 | Professional and visually sound, but extremely dense and somewhat overextended |
| Reproducibility | 7.5/10 | Artifact mapping, hardware job IDs, seeds, exclusions, provenance discussion are strong |

Estimated editorial outcome at a demanding Q1 venue: desk rejection ~30-45%, rejection
after review ~40-55%, acceptance in current form ~5-15%, acceptance after substantial
revision ~25-40% depending on journal fit.

## Strongest aspects (keep doing these)

- Commendably honest about negative results.
- D4 pooling argument is mathematically clean, numerical equivariance near machine precision.
- Real IBM hardware execution is more valuable than simulator-only validation.
- Supplement reports resource matching, leakage auditing, seed-level inference, CIs, provenance problems.
- Correctly avoids claiming quantum advantage.
- Hardware methodology and checkpoint replay are described better than in many hybrid-quantum papers.

## Main Q1-level problems

1. **Held-out task loses to simpler controls.** HVK2D ~18.12 dB vs 18.80 dB for raw-linear/local-observable controls — removes the most natural argument for publishing a new vision architecture.
2. **"Autoencoder" framing is vulnerable.** Per-image fitting demonstrates memorization, not generalization; reviewers may argue the quantum pathway isn't essential.
3. **D4 not integrated into the principal model** — it's a feature-grid diagnostic, so "with Provable Symmetry" in the title overstates the system-level result.
4. **Hamiltonian component doesn't help** — removing the energy loss improves reconstruction, weakening "physics-informed" as a performance motivation.
5. **Entanglement-sensitive result is too constructed** — deliberately designed so pair observables succeed under linear readout; may read as a feature-engineering sanity check rather than evidence of practical benefit.
6. **Hardware validation is a feasibility demonstration only** — 5 images, 1 trial, 256 shots, no mitigation, no statistical hardware benchmark.
7. **Evidence hierarchy is complicated** — combines per-image reconstruction, synthetic diagnostics, simulator experiments, surrogate feature maps, small real-circuit comparisons, hardware replay, symmetry proofs, and phase-transition diagnostics; broad but not deep in any one direction.
8. **Phase-transition/effective-temperature terminology is risky** — small systems, short traces, mixed detection rates, fixed entropy denominator; reviewers may see this as physical overinterpretation unless called an optimization diagnostic.

## Highest-impact revisions (source list for this tracker's subfolders)

1. Reframe the central claim as a controlled boundary study, not a superior architecture.
2. Replace per-image headline results with a genuine dataset-level experiment (hundreds-thousands of images, fixed shared model, val/test splits, multiple seeds, unseen-class/distribution-shift eval).
3. Integrate D4 pooling into the complete trainable model (output-level equivariance, sample efficiency, robustness, pooled classical baseline).
4. Redesign the Hamiltonian objective to measurably help, or reframe energy strictly as an interpretability diagnostic.
5. Strengthen the hardware study (repeated jobs, multiple calibration periods, error mitigation, uncertainty intervals, hardware-aware training, matched noisy-simulator controls).
6. Add stronger classical comparisons (conv autoencoder, coordinate MLP, MPS-only, Fourier-feature MLP, classical polynomial features, parameter-matched tensor network).
7. Reduce or move the phase-transition section unless proper finite-size scaling and preregistered detection criteria are provided.
8. Shorten the title, e.g. "Hamiltonian Vision Kernel: Symmetry-Pooled Quantum Correlator Features with Hardware Validation."

## Bottom line

The most credible story going forward: a rigorous study showing what a hybrid quantum
vision representation can guarantee, where it helps, and — equally importantly — where
it does not. Recommend major revision and aggressive reframing, not immediate submission.
