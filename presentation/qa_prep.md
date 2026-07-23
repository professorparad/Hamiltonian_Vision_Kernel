# HVK Ablation Seminar Q&A Prep

## Statistical Rigor and Evidentiary Weight

Q: Five seeds and 20 held-out CIFAR-10 reconstructions sound small. Why should we trust the negative result?

A: I would not claim this is a full CIFAR-10 benchmark. The paper explicitly calls the held-out CIFAR-10 validation "intentionally lightweight": five random splits and 20 held-out reconstructions. The reason I still trust the direction of the negative result is that it is paired, resource-matched, and consistent. On those 20 paired held-out images, raw-linear and local-observable controls reach `18.80 +/- 1.42` dB while HVK2D reaches `18.12 +/- 1.54` dB, with a mean paired difference of `-0.68` dB, bootstrap 95% CI `[-0.93, -0.46]`, and Wilcoxon `p = 9.54e-6`. So the claim is not "we have exhausted CIFAR-10"; it is "under this controlled protocol, we find no evidence that the quantum component is load-bearing, and the evidence we do have consistently favors the matched classical controls."

Q: Why these specific sample sizes: five seeds, 20 held-out CIFAR images, and 400 cached images in the multi-dataset suite?

A: They reflect a tiered validation design rather than a final benchmark scale. The expensive trainable component ablations are five-seed, 240-step matched runs on Monalisa. The held-out CIFAR layer uses five random splits and 20 held-out reconstructions so paired image-level tests can be computed without changing feature width or readout capacity. The broader suite uses 400 cached images per dataset with stratified splits as a fixed-feature sanity check across CIFAR-10, MNIST, Fashion-MNIST, PathMNIST, BloodMNIST, and PneumoniaMNIST. The paper is clear that a full class-balanced CIFAR-10 test-set protocol and end-to-end retraining per dataset remain future work.

Q: Are the 20 held-out CIFAR pairs independent enough for a Wilcoxon test?

A: The paper treats the image-level held-out reconstructions as paired observations within five random splits. That is a reasonable first-pass paired design, but I would not oversell independence as if this were thousands of IID test samples. The defensible point is that every model comparison is made on the same held-out images with the same 32-D feature width and 2112 readout parameters. The Wilcoxon result supports a consistent paired direction; it does not replace a larger class-balanced test-set study.

Q: Why use a Wilcoxon signed-rank test rather than a t-test?

A: Because the sample is small and paired. The Wilcoxon signed-rank test does not require assuming Gaussian paired differences. The paper also reports bootstrap 95% confidence intervals, which helps show the size and stability of the paired gap. For raw-linear versus HVK2D, the paired PSNR difference is `-0.68` dB with CI `[-0.93, -0.46]`, so both the sign and the scale are visible.

Q: Are you claiming statistical significance for all tables?

A: No. The paper separates evidentiary tiers. The core component ablation table is five-seed and matched at 240 steps. The held-out CIFAR comparison has paired statistics. The restricted pair-correlation diagnostic has five seeds. But the Hamiltonian controls, capacity sweeps, second-image generalization control, and the Monalisa 1D versus 2D topology comparison are explicitly single-seed or directional. I would not cite small differences in those tables as statistically resolved.

Q: The held-out CIFAR gap is only 0.68 dB. Is that practically meaningful?

A: It is small in absolute PSNR terms. The point is not that the classical control is dramatically better; the point is that the quantum map does not win even under a strict same-width, same-readout comparison. A small but consistent gap against the quantum model is enough to reject the stronger claim that the trained entangling HVK2D map is adding load-bearing value on this task.

Q: Does the multi-dataset suite really prove generalization across real images?

A: It supports the direction, but it is still a lightweight fixed-feature suite. The reconstruction table reports local/raw controls winning or tying across CIFAR-10 native, MNIST, Fashion-MNIST, PathMNIST, BloodMNIST, and PneumoniaMNIST. For example, CIFAR-10 native is `21.17 +/- 1.98` dB for the best local/raw control versus `19.99 +/- 2.02` dB for the HVK2D pair map. But the paper is explicit that full end-to-end retraining per dataset is future work.

## Leakage and Trustworthiness

Q: You found one leakage bug in your own earlier benchmark. Why should we trust that there is not another undetected leakage problem elsewhere?

A: That is a fair concern. The honest answer is that finding the bug reduces trust in the withdrawn diagnostic, not in the result after it was withdrawn. The paper does two things to address this: first, it removes the circular CIFAR-derived nonlocal diagnostic from every quantitative claim; second, it turns the failure into an explicit audit rule: trace target-construction and feature-construction code for shared closed-form subexpressions and test whether the target is linearly recoverable from a candidate's own input features. The remaining entanglement-necessity diagnostic is stated to have been independently audited against that exact pattern.

Q: What exactly was leaked in the withdrawn diagnostic?

A: The target was built from six product terms of left and right patch features. The supposed entangling feature map then included the same six product formulas in its own `products` block, concatenated them into its inputs, and the fixed-width truncation kept those columns. With ridge regularization `1e-6`, the readout essentially selected those columns and reached near-machine precision. The `120` dB PSNR was also a metric-floor artifact: the MSE was clamped at `1e-12`, and `20 log10(1/sqrt(1e-12)) = 120`.

Q: Does the leakage story undermine the legitimate entanglement-necessity result?

A: It limits how confidently we should treat constructed diagnostics in general, but the surviving diagnostic is intentionally different. In the withdrawn case, the model was handed the same formulas used to build the target. In the restricted pair-correlation diagnostic, every control receives the same raw inputs, and the synthetic target is not concatenated into any candidate's feature block. Under that setup, HVK2D entangling observables reach mean `R^2 = 0.9735`, while no-entanglement is `0.0191`, raw-linear is `0.0133`, parameter-matched classical is `-0.0297`, and random VQC is `-0.1732`. The claim is narrow: entangling pair observables are necessary for that deliberately favorable target under a fixed linear readout.

Q: If the withdrawn diagnostic had `R^2 = 1.0`, how did it get into the project at all?

A: Because it looked like a strong nonlocal-correlation result before the target and feature code were traced together. That is exactly the methodological lesson. Dissimilar names or preprocessing are not enough; one has to inspect whether a candidate feature block contains a quantity from which the target is linearly recoverable. The paper reports the failure rather than deleting it because that audit is probably more transferable than the failed number.

Q: Are the same-set MLP results over 100 dB also leakage?

A: The paper treats those as memorization or same-set capacity, not leakage evidence and not generalization evidence. On five native CIFAR-10 images, the MLP best-case PSNR exceeds `100` dB. That is explicitly called a red flag for memorization, which is why same-set CIFAR and Monalisa are not used as the basis for the central representation-learning claim.

## Framing and Scientific Scope

Q: Is calling this a "negative result" just a way to avoid doing the harder work of explaining why the quantum part fails?

A: No, but it does narrow the paper's claim. The paper is not presenting a complete theory of failure or an optimized replacement architecture. It asks a component-wise question: under resource-matched, freeze-based, leakage-audited tests, does any quantum component measurably help? The answer is no for held-out natural-image reconstruction at this scale. The discussion gives a plausible reason: these image patches are locally structured and low-entanglement enough that raw or local statistics with a linear or shallow readout are already close to sufficient. A deeper architecture search is future work, but it should start from this negative baseline rather than from an untested advantage claim.

Q: Does the title "Do the Quantum Parts Help?" overstate the scope?

A: The answer should be scoped every time: in this HVK architecture, on the tested natural-image reconstruction tasks, under the tested resource-matched protocol, no quantum component is shown to be load-bearing. The paper does not claim that quantum models never help vision, or that entanglement is useless in all representation learning. In fact, the restricted pair-correlation diagnostic is a counterexample where entangling observables do help.

Q: Are you abandoning quantum vision, or just this architecture?

A: Neither as a blanket statement. The result says ordinary natural-image reconstruction at this patch scale is not a good place to claim HVK quantum advantage. The continuing direction would be to search for tasks where the target really requires nonlocal products or Hamiltonian structure after strict classical controls. The paper's own positive diagnostic suggests that such task structure matters. What should be abandoned is the weaker claim that adding a VQC, CNOTs, MPS features, or a Hamiltonian regularizer automatically improves image reconstruction.

Q: If the classical decoder does the work, why keep the quantum machinery at all?

A: For this reconstruction task, there is no performance argument for keeping it. The only defensible reasons to keep exploring it are scientific: to identify task regimes where entangling observables create features a matched classical linear/local map cannot recover, to test symmetry-enforced feature maps such as the D4-pooled variant end to end, and to use the architecture as a controlled testbed for evaluation methodology. As an image autoencoder for these datasets, the paper's result points away from HVK's quantum machinery.

Q: Could the whole quantum feature pipeline be replaced by equivalently sized classical random features?

A: Random classical features are not enough in the held-out CIFAR table: strict classical random features score `15.85 +/- 1.39` dB, below HVK2D's `18.12 +/- 1.54` dB. But simple raw/local classical features are enough, and they score `18.80 +/- 1.42` dB. So the right conclusion is not "any random classical map works"; it is "the useful information is already present in raw/local statistics under the same width and readout budget."

## Hamiltonian and Architecture Questions

Q: Why does removing the Hamiltonian energy term improve performance? Doesn't that undercut the physics-informed motivation?

A: It undercuts the claim that this particular energy regularizer improves reconstruction. In the five-seed Monalisa component table, removing energy reaches `33.40 +/- 0.06` dB versus the shared VQC baseline at `32.97 +/- 0.22` dB. In the single-seed Hamiltonian controls, signed energy gives `32.24` dB and no-energy gives `33.30` dB. The paper's explanation is that the legacy signed mean-energy objective can be reduced by driving learned energy negative without improving pixels, so it adds gradient noise or a competing objective. The physics idea is not disproved universally, but in this setup the pixel loss is already informative enough that the regularizer is not helpful.

Q: Why did the contrastive Hamiltonian objective not solve the problem?

A: It helped relative to the signed-energy baseline but still did not beat removing energy. The single-seed table reports contrastive Hamiltonian core at `32.84` dB versus signed-energy baseline `32.24` dB, but contrastive with no energy loss reaches `33.33` dB. So contrastive energy is diagnostically better than the legacy signed energy, but it still does not provide a reconstruction-quality advantage here.

Q: If no-entanglement ties the baseline on Monalisa and beats HVK2D on held-out CIFAR, are CNOT gates useless?

A: Useless for this natural-image reconstruction task under this protocol, yes. On Monalisa, no-entanglement is `32.98 +/- 0.15` dB versus baseline `32.97 +/- 0.22` dB. On held-out CIFAR, no-entanglement is `18.46 +/- 1.32` dB versus HVK2D `18.12 +/- 1.54` dB, with paired mean difference `-0.34` dB and Wilcoxon `p = 0.021`. But the restricted pair-correlation diagnostic shows CNOT-enabled pair observables can matter when the target explicitly requires distant products.

Q: If no-MPS is statistically tied with baseline, why use MPS features?

A: For this reconstruction benchmark, the MPS stage is not isolated as necessary. The five-seed Monalisa result is `33.04 +/- 0.15` dB for no-MPS versus `32.97 +/- 0.22` dB baseline. The architecture motivation remains that MPS features are compact physics-inspired summaries, but the ablation says the decoder can also use simpler patch statistics at this scale.

Q: The random VQC is much worse, so doesn't that prove the trained quantum circuit matters?

A: It proves that arbitrary random latents are bad, not that trained quantum parameters are load-bearing. Random VQC output gives `27.96 +/- 0.28` dB on Monalisa and `11.15 +/- 1.19` dB on held-out CIFAR, so structure matters. But freeze-quantum, no-entanglement, no-MPS, and matched classical replacements all tie or exceed the trained VQC baseline. The useful structure does not require the trained entangling quantum circuit.

Q: Are the classical controls truly resource matched?

A: In the held-out CIFAR validation layer, yes by the paper's stated design: every row uses 32-D features and the same 2112-parameter linear readout. The table lists HVK2D, ZZ-only, no-entanglement, strict classical random features, and raw/local controls all at feature dimension 32 and readout parameters 2112. That is why the held-out comparison is more important than looser same-set baselines.

## D4 Symmetry and Topology

Q: The D4 pooling result is provable by construction. Is it doing real scientific work, or is it just a corollary dressed up as a contribution?

A: It is a construction and verification result, not an empirical reconstruction result. The scientific value is modest but real: the unpooled HVK2D map is only geometry-motivated and has equivariance error around `0.815 +/- 0.129`, similar to local/raw at `0.837 +/- 0.131`. The D4-pooled map enforces the symmetry and verifies at `9.57e-17 +/- 1.26e-17` over 7000 patch transforms. But I would not claim it improves reconstruction, because the paper explicitly says it has not been integrated into an end-to-end retrained pipeline.

Q: If D4 equivariance is exact by construction, why verify it numerically?

A: To confirm the implementation, not to prove the theorem. The proof is the group average; the numerical check catches coding or indexing mistakes in the observable-grid transform. The important empirical comparison is that the unpooled positional map is not accidentally equivariant: its error is around `0.815`, not near floating-point precision.

Q: The Monalisa 2D topology advantage is `+5.98` dB. Why not make that a major positive result?

A: Because it is a single-image, single-seed same-set result. The paper says HVK2D gets `40.70` dB versus `34.72` dB for the 1D counterpart on Monalisa, which is consistent with topology alignment helping. But it has not been separated from decoder capacity, optimization budget, or dataset-specific structure. It can motivate future architecture work; it cannot bear a general claim.

Q: Does the D4 result contradict the negative result?

A: No. D4 pooling gives a symmetry guarantee for a feature map. The negative result is about end-to-end held-out reconstruction performance of the tested HVK maps against matched classical controls. A representation can have a desirable symmetry and still not improve this reconstruction task. The paper is explicit that the D4-pooled map is not yet part of a retrained reconstruction pipeline.

## Hardware and Quantum Advantage

Q: You did not run on real quantum hardware. Does this paper say anything about quantum advantage on hardware?

A: No. The IBM result is only a compiled-circuit feasibility check. Both HVK1D and HVK2D compile to six measured qubits and depth 18 on a Heron-class basis; HVK1D uses 10 CNOTs and HVK2D uses 14 CNOTs. There are no finite-shot hardware reconstructions and no hardware image-quality claims. The paper is about component attribution under simulation and fixed-feature validation, not quantum advantage on hardware.

Q: If the circuit is only six qubits and depth 18, is this too small to say anything about quantum machine learning?

A: It is too small for broad quantum advantage claims. It is large enough to audit this specific HVK architecture and ask whether its quantum components help under matched controls. The paper's contribution is methodological: resource matching, freeze isolation, held-out evaluation, and leakage auditing. Scaling the circuit is a separate future-work question, and the current result warns that scaling should be tested against strong classical controls from the beginning.

Q: What about shot noise? Would finite shots change the conclusion?

A: The paper includes a shot-noise robustness diagnostic from 128 to 8192 shots, but the headline comparison is not a hardware-shot experiment. Since HVK2D already fails to beat raw/local controls in the noiseless or fixed-feature setting, adding finite-shot noise would not be expected to rescue a quantum advantage. At best it would be another robustness condition to test.

## Single-Image and Same-Set Results

Q: How much weight should the Monalisa results carry?

A: They are useful for component isolation because the runs are controlled, matched, and five-seed for the core ablations. But Monalisa is a single-image optimization task, not a representation-learning benchmark. The paper makes that explicit with a second-image zero-shot control: a model trained on Monalisa reaches only `7.78` dB and `SSIM = 0.0196` on a structurally different image, while training with the second image included reaches `28.31` dB and `SSIM = 0.9862`.

Q: Why present same-set CIFAR results at all if they are not generalization evidence?

A: They show capacity and positioning relative to familiar baselines, but they are explicitly labeled as same-set diagnostics. HVK2D reaches `34.50` dB mean PSNR on five native CIFAR-10 images, but MLP and CNN baselines are stronger in pure same-set pixel accuracy, with MLP best-case PSNR above `100` dB. The paper uses those results to motivate caution, not to claim representation quality.

Q: Could the decoder simply be memorizing patch coordinates?

A: In same-set settings, yes, that is a real risk and the paper treats them as capacity diagnostics. That is why the held-out CIFAR layer is the evidentiary core: the shared linear readout is trained on a disjoint image subset and evaluated on held-out reconstructions. The second-image zero-shot failure also shows that single-image training does not learn a broadly transferable image prior.

## Future Work and Defensible Next Steps

Q: What exactly would you need to do before making a stronger claim?

A: Three things. First, run a full class-balanced CIFAR-10 test protocol rather than 20 held-out reconstructions. Second, retrain the full pipeline end to end per dataset instead of relying only on lightweight fixed-feature readouts in the multi-dataset suite. Third, integrate the D4-pooled map into a trainable reconstruction pipeline and compare it against the same 32-D, 2112-readout-parameter classical controls.

Q: What would count as evidence that quantum structure really helps?

A: A held-out task where the trained entangling map beats raw/local, no-entanglement, random-feature, and parameter-matched classical controls under the same feature width and readout budget, with multiple seeds and a leakage audit. The restricted pair-correlation diagnostic shows one narrow version of that: `R^2 = 0.9735` for HVK2D entangling observables versus `R^2 <= 0.02` for non-entangling controls. The missing step is finding a real data task where the same kind of advantage survives.

Q: If you had to summarize the thesis defense in one sentence, what would it be?

A: The strongest result is negative but useful: in this HVK image autoencoder, the classical decoder is load-bearing, while the trained VQC, CNOT entanglement, MPS features, and Hamiltonian energy term do not improve held-out natural-image reconstruction over strict matched classical controls.

## Restored Demo Slides and Transition Questions

Q: Why does the quantitative-results slide show about 27 dB while later slides reference about 33 dB for the same model? Isn't that inconsistent or suspicious?

A: It is not the same training run. The demo slide reports one specific Monalisa demo run: MSE `1.953763e-3`, PSNR `27.09128` dB, 387 model parameters, and about `1.090304e6` decoder parameters. The later ablation table reports a more careful matched protocol: five seeds, 240 training steps for every variant, with the shared VQC baseline at `32.97 +/- 0.22` dB. Same architecture, different training budget and protocol. I would say this explicitly: the 27 dB slide is a historical/demo reconstruction run; the 33 dB number is the later matched-ablation benchmark, and the latter is what should carry quantitative evidentiary weight.

Q: Does the random-latent and zero-latent sanity check really prove the decoder is not memorizing?

A: It is a useful sanity check, not a rigorous proof against memorization. In the talk, the point is modest: feeding random or all-zero latent vectors gives visibly broken output, so the decoder is not producing the Monalisa reconstruction completely independent of its input. But a sufficiently overfit decoder could still learn a strong image prior or coordinate-dependent template while remaining sensitive to bad latents. The rigorous follow-up is the freeze-based ablation and held-out validation, where freezing the classical side collapses to `10.93 +/- 0.00` dB and held-out real-image controls are tested separately.

Q: Do the observable distribution and correlation matrix prove that the quantum circuit is doing something non-trivial?

A: They show the observable channel is structured, not pathological noise. They do not prove that the trained entangling quantum circuit is load-bearing for reconstruction. The ablation paper is clear on that distinction: random VQC output is worse (`27.96 +/- 0.28` dB on Monalisa and `11.15 +/- 1.19` dB on held-out CIFAR), so structure matters, but no-entanglement, no-MPS, freeze-quantum, and small classical replacements tie or beat the trained VQC baseline. I would phrase the diagnostic slides as interpretability and health checks, not causal evidence of quantum advantage.

Q: If the demo images without kernel, positional-only, and full HVK look like a strong ablation, why doesn't that count as evidence the Hamiltonian kernel matters?

A: It counts as a qualitative demo, not as a controlled component attribution. Those slides show that the full pipeline can reconstruct and that trivial-looking alternatives can fail in that demo configuration. But the later paper asks a stricter question: after matching training budget, seeds, feature width, and controls, which component is necessary? Under that protocol, removing the energy loss improves performance (`33.40 +/- 0.06` dB versus `32.97 +/- 0.22` dB baseline), no-entanglement is tied (`32.98 +/- 0.15` dB), and no-MPS is tied (`33.04 +/- 0.15` dB). So the images motivate the question; they do not settle it.

Q: Is it a bait-and-switch to show an impressive demo first and then undercut it with a negative ablation result?

A: The clean framing is: "First I will show you why we thought the model was interesting; then I will show you what survived when we tested that belief." The demo establishes that HVK reconstructs images and produces interpretable diagnostics. The ablation study answers a different and harder question: whether the quantum components deserve credit for that reconstruction. The transition slide already says this directly: the demo shows HVK reconstructs images well, but it does not show which component is responsible.

Q: What does the 387-parameter versus 1.09-million-parameter split foreshadow?

A: It foreshadows the central ablation result. The quantitative slide says the trainable model side is tiny, 387 parameters, while the decoder has about `1.090304e6` parameters out of `1.090691e6` total. That does not by itself prove the decoder is doing the work, but it should make the audience suspicious about where the capacity lives. The later freeze result confirms that suspicion: train only the quantum side with the classical decoder frozen and reconstruction collapses to `10.93 +/- 0.00` dB; train the classical side with the quantum circuit frozen and it reaches `33.26 +/- 0.10` dB.

Q: Should the presenter flag the parameter imbalance earlier instead of waiting for the ablation pivot?

A: Yes, but as a caution, not as a conclusion. On the quantitative slide, say: "Notice the asymmetry here: the decoder has essentially all the parameters. That is exactly why the next question is attribution, not just reconstruction quality." That prepares the audience for the negative result and makes the transition feel honest rather than abrupt.

Q: Are the demo reconstruction slides misleading if the paper later says the classical decoder is load-bearing?

A: They are misleading only if presented as proof of quantum advantage. They are accurate as a first-pass demonstration that the complete HVK pipeline can reconstruct Monalisa under that run. The presenter should avoid saying "the Hamiltonian kernel causes this improvement" from those images alone. The safer wording is: "This is the behavior that motivated the ablation study; now we test whether the quantum part actually caused it."

Q: How should the presenter reconcile "physically interpretable representation" with the negative result?

A: Interpretability and utility are different claims. The observable distributions, correlation matrix, energies, and entropies make the latent representation inspectable in physics language. But the ablation results say that this interpretable structure is not necessary for held-out natural-image reconstruction under the tested controls. A compact phrasing is: "The representation is physically interpretable; the decoder is what is load-bearing."

Q: Does the positional-encoding-only slide prove position alone is insufficient?

A: Only for that demo configuration. It is fair to say position alone did not reconstruct the image in the restored demo, which supports the intuition that observable content matters. But the paper's stronger finding is more nuanced: useful structure is needed, yet it need not come from the trained entangling VQC. Raw/local controls and non-entangling variants can carry enough information under matched evaluation.

Q: If random VQC output is worse than the baseline, why not conclude the trained quantum circuit helps?

A: Because "better than random" is a weak control. The paper reports random VQC output at `27.96 +/- 0.28` dB on Monalisa, below the `32.97 +/- 0.22` dB shared VQC baseline, so arbitrary latents are worse. But stronger controls are the relevant comparison: freeze-quantum reaches `33.26 +/- 0.10` dB, no-entanglement reaches `32.98 +/- 0.15` dB, no-MPS reaches `33.04 +/- 0.15` dB, and classical tanh replacement reaches `33.50 +/- 0.00` dB. So the trained circuit beats nonsense, but not the matched alternatives.
