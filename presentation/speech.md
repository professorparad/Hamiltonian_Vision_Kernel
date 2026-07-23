# Speaking script — "Design of a Hamiltonian-Based Vision Kernel (HVK)"
(First-time audience — nobody in the room has seen this work before. ~12–15 minutes, paced to the 17-slide deck: presentation/hvk_full_presentation.tex)

---

### Slide 1 — Title
Good [morning/afternoon] everyone. This is the first time I'm presenting this work to you, so I'm going to start from scratch. I'll show you what the model is, walk you through what it actually produces, and then — because I want to be upfront about this from the start — I'll tell you about the follow-up study where we stress-tested our own result and found some real problems with it.

### Slide 2 — Overview
Here's the plan. First, the motivation and the core idea. Then the pipeline and how we prepare an image for it. Then I'll show you the actual reconstruction results — this is the fun part. Then some sanity checks and diagnostics that convinced us the model wasn't cheating. Then the quantitative numbers. And finally, the part I think is actually the most important: what happened when we tried to prove the quantum part was responsible for any of this.

### Slide 3 — Motivation & Idea
The starting problem: current quantum architectures don't scale to high-definition images — you'd need a prohibitive number of qubits and circuit depth to represent a full image directly. So the idea is to compress first. We break an image into patches, compress each patch into a matrix product state, extract a small set of quantum features from a variational circuit, and decode everything back into pixels with a classical network. On top of that we add a learnable Heisenberg energy term, borrowed from physics, as a regularizer — the idea being that a physically motivated inductive bias might help the model learn a better representation.

### Slide 4 — Model Pipeline
Here's the full pipeline end to end. Image in, resized to 256 by 256, cut into sixteen 64-by-64 patches. Each patch gets flattened, reshaped into a 12-qubit tensor, and compressed via an MPS contraction with bond dimension 4. We extract Pauli-Z and Pauli-X expectation values and a nearest-neighbor correlation map from that MPS, combine it with a positional encoding, feed the whole thing into a variational quantum circuit, compute a Heisenberg-inspired cost function from the circuit's observables, and decode it all back into pixels with a classical decoder that stitches the patches back together.

### Slide 5 — Positional Encoding / Image Preparation
Because we're processing patches independently, we need to tell the model where each patch came from. Each patch gets a normalized (x, y) coordinate, and we run that through a sinusoidal positional encoding — the same style of encoding you'd see in a transformer — so the decoder can put the pieces back in the right place.

### Slide 6 — Reconstruction Without the Hamiltonian Kernel
Before showing you the full result, here's a baseline: if you skip the Hamiltonian kernel step entirely, this is what you get. It's clearly broken — the reconstruction has no coherent structure. This is here to show you the floor.

### Slide 7 — Reconstruction Using Only Positional Field Encoding
If you use only the positional encoding — no quantum observable information at all — you get something with a rough silhouette, but no real detail. This tells us position alone isn't enough; the model needs the observable content too.

### Slide 8 — Reconstruction With the Full HVK
And here's the full pipeline. Original on the left, quantum reconstruction on the right. This is the headline demo result — the model is clearly recovering the image, not just an outline.

### Slide 9 — Decoder Sanity Check: Random and Zero Latent
Before you trust a reconstruction result, you have to ask: is the decoder actually using the quantum features, or is it just memorizing the image regardless of input? So we fed it garbage — a random latent vector and an all-zero latent vector — and both produce visibly broken, structureless output. That's what we want to see: it proves the decoder's output is genuinely dependent on the quantum observable content, not just decoder capacity alone.

### Slide 10 — Observable Distribution
This shows the distribution of the Pauli-Z and Pauli-X expectation values, and the combined observable distribution, across all the patches. Nothing pathological — it's a reasonably smooth, structured distribution, which is what you'd hope for from a working variational circuit.

### Slide 11 — Correlation Matrix
This is the correlation structure across all 27 quantum observables. You can see clear block structure — some observables are strongly correlated with each other, others aren't — which suggests the circuit is extracting some genuine, non-trivial structure from the patches rather than producing noise.

### Slide 12 — Positional Encoding Heat Map
And this is what the sinusoidal positional encoding actually looks like across the sixteen patches — you can see the expected smooth periodic structure across the encoding dimensions.

### Slide 13 — Quantitative Results
Here are the numbers for this specific training run: MSE around 1.95 times ten to the minus 3, PSNR about 27 dB, and a decoder that's using about 1.09 million parameters against just 387 parameters in the quantum/model side. One honest note — later in our more rigorous ablation study, using a longer and more carefully matched training protocol, the baseline PSNR for the same architecture actually comes out higher, around 33 dB. Same architecture, different training budget — I'll get to why that matters in a moment.

### Slide 14 — Does the Quantum Part Actually Help?
So that's the demo, and it works. But here's the question none of that tells you the answer to: which part of this pipeline is actually responsible for the quality of that reconstruction? Is it the quantum circuit? The entanglement? Or is it just the classical decoder — which has over a million parameters — doing what classical decoders always do? We didn't want to just assume the answer, so we built a resource-matched, freeze-based, multi-seed ablation study to test it directly. Here's what we found.

### Slide 15 — What We Found: The Classical Decoder Does the Work
This is the part I want to be completely honest about. If you freeze the classical decoder and only train the quantum circuit, reconstruction collapses to near noise. If you do the opposite — freeze the quantum circuit and only train the classical decoder — you tie or beat the fully trained baseline. Removing entanglement: no significant change. Removing the MPS compression: no significant change. Replacing the whole quantum circuit with a small classical nonlinearity: it wins. And on held-out CIFAR-10 images — real generalization, not memorization — a plain linear readout over raw pixel statistics beats the fully trained model, and it's statistically significant, p equals nine-point-five times ten to the minus six. On top of that, we caught a leakage bug in an earlier internal diagnostic that had looked like a perfect result — R-squared of 1.0 — and traced it to the target being built from the same features we'd handed the model, so we withdrew it. Two narrow, honestly-scoped positive results do survive: entanglement is genuinely necessary on a synthetic task built around distant-patch correlations, and a group-pooled variant of our feature map is provably symmetric under the symmetries of the square, verified to machine precision.

### Slide 16 — Conclusion
So, to bring it together: HVK reconstructs images well, and its internal representation is physically interpretable — the correlation structure and observable distributions you saw aren't noise. But when we actually tested it rigorously, the classical decoder — not the quantum circuit — turned out to be doing the load-bearing work on real held-out images. I'm presenting that as the finding, not hiding it, because I think the evaluation protocol we built to catch this is more valuable than an unverified positive claim would have been.

### Slide 17 — Thank you
Thank you — I'm happy to take questions.
