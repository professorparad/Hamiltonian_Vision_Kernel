# Literature survey for the PCAE manuscript

**Manuscript:** *Hybrid Quantum–Classical Pauli-Correlator Architecture for Image Reconstruction* (`main_paper.tex`, commit 8fa9c27)
**Date of survey:** 29 September 2026
**Scope:** work up to September 2026 that a referee at a QML venue (e.g. Quantum Machine Intelligence) is likely to know and expect to see.

---

## 0. How to use this report

Each theme below has four parts: what the field has done, where PCAE sits, what the paper already cites, and what is missing.
Recommended additions are tagged:

- **[MUST]** A referee is likely to raise it, or it directly affects a claim or a term in the paper.
- **[SHOULD]** Strengthens positioning; cheap to add in one sentence.
- **[OPTIONAL]** Context only; add if space allows.
- **[VERIFY]** Bibliographic details (volume, pages, final venue) must be checked from the DOI before the entry goes into `sn-bibliography.bib`.

This report recommends citations and wording. It does not change any result. Any sentence that would need a new number is flagged for the student, in line with the "no results changes" rule.

The paper currently cites **65** bibliography entries. Section 9 lists about 30 candidate additions, ranked. Section 8 lists places where the literature creates risk for a specific claim.

---

## 1. What the paper claims, and what each claim needs from the literature

| # | Claim in the paper | Section | Literature it must be placed against |
|---|---|---|---|
| C1 | PCAE is a new *assembly*: MPS patch features → 6-qubit VQC → measured Pauli-correlator vector → Fourier positional encoding → MLP decoder, plus a Hamiltonian energy term | §1, §2 | Hybrid QAEs with expectation-value latents; MPS→VQC hybrids; patch-based quantum vision |
| C2 | Per-image fitting across six datasets, 25.8–41.6 dB | §4.1 | Classical INRs (SIREN, Fourier features, COIN); quantum INRs (QIREN, QINC, QVF, 3D-QISR) |
| C3 | Held-out *closed-form surrogate* is within ±1 dB of four classical maps (TOST + Holm) | §4.3 | Dequantisation, classical surrogates, benchmarking practice, equivalence testing |
| C4 | Hardware replay on IBM Heron without retraining, 25.9–31.5 dB, unmitigated | §4.4–4.6 | Hardware QML inference studies; error mitigation |
| C5 | Exact D4 equivariance by Reynolds averaging of the observable grid | §4.7 | Classical group/frame averaging; equivariant QNNs for images |
| C6 | Pair observables are necessary for distant-product targets under a fixed linear readout | §4.8 | Projected kernels, reservoir readouts with quadratic expansion, Fourier-series view of VQCs |
| C7 | Hamiltonian energy term is an auxiliary objective and readout; it lowers PSNR by 5.93 dB | §5.1 | Hamiltonian learning; energy-style regularisers in QML (including a recent negative result) |
| C8 | Natural image patches are low-entanglement; a classical readout is close to sufficient | §5.2 | Entanglement of image data under MPS encodings |

---

## 2. Hybrid quantum autoencoders and quantum image compression (C1)

**State of the field.** Two lines exist.

1. *Register (fully quantum) autoencoders.* Romero et al. (2017) introduced the trash-qubit fidelity QAE. Image-oriented variants compress pixel amplitudes (Wang et al. 2024) or use the QAE as a feature extractor for classification (Asaoka & Kudo 2025; Lo, Hsu & Kuo 2025). Lo et al. report good MNIST feature extraction but weak *reconstruction*, which is a useful contrast for PCAE. A 2026 preprint applies a patch-wise QAE to brain-MRI anomaly detection (Ganguly, Liang & Makris 2026).
2. *Hybrid autoencoders with a measured-expectation bottleneck.* Sakhnenko et al. (2022) put a PQC between a classical encoder and a classical decoder and use the per-qubit expectation values as the latent space. This is architecturally the nearest published pattern to PCAE's "measured observables handed to a classical decoder". Eren (2026) uses a classical CNN encoder with a quantum INR decoder for MNIST/Fashion-MNIST reconstruction and generation. Jadhav et al. (Sept 2026) combine a quantum wavelet transform, a trainable QCNN and a classical SR-GAN for compression and report 30.07 dB PSNR against JPEG2000.

**Where PCAE sits.** PCAE belongs to line 2, not line 1. Its distinguishing points within line 2 are (a) MPS patch features as the circuit input, (b) *pairwise* correlators (ZZ/XX/YY) in the latent, not only single-qubit ⟨Z⟩, and (c) the energy term and D4 pooling.

**Already cited.** Romero2017, WangQAECompression2024, QAEClassification2025, Slabbert2024.

**Gaps.**
- **[MUST] Sakhnenko et al. 2022.** The intro's "Novelty relative to the nearest hybrid-vision architectures" lists three families and omits this one, which is the closest. A referee who knows it will read the novelty paragraph as incomplete. Add it as a fourth family ("hybrid autoencoders with an expectation-value bottleneck") and state the difference in one sentence (pair correlators + MPS input + energy term).
- **[SHOULD] Lo, Hsu & Kuo 2025** — reconstruction is hard for register QAEs; supports the choice of a classical decoder.
- **[OPTIONAL] Ganguly et al. 2026; Jadhav et al. 2026; Eren 2026** — recent image-domain hybrids; shows the area is active.

---

## 3. Per-image fitting: classical and quantum implicit neural representations (C2)

**State of the field.** Classical INRs fit one network to one signal: SIREN (Sitzmann 2020), Fourier features (Tancik 2020), NeRF, COIN and INR compression (Dupont 2021; Strümpler 2022), Deep Image Prior. Quantum INRs now form a small line: QIREN (Zhao et al., ICML 2024), QINC (Fujihashi & Koike-Akino, 2024; now in a Springer proceedings volume), Quantum Visual Fields (Wang, Theobalt & Golyanik, NeurIPS 2025), and 3D-QISR for novel view synthesis (Lizzio Bosco et al. 2026).

**Where PCAE sits.** The paper already places §4.1 correctly: per-image fitting, a patch autoencoder rather than a coordinate network, no bit-rate claim.

**Already cited.** Ulyanov2020DIP, Sitzmann2020, Tancik2020, Mildenhall2020NeRF, Dupont2021COIN, Strumpler2022, Zhao2024QIREN, Fujihashi2024QINC, LizzioBosco2026QISR.

**Gaps.**
- **[SHOULD] Wang, Theobalt & Golyanik 2025 (QVF, NeurIPS).** A peer-reviewed quantum INR at a top ML venue that "matches classical baselines". It belongs next to QIREN.
- **[VERIFY] Fujihashi2024QINC** is cited as an arXiv preprint; a published chapter now exists (Springer, 2025/26). Update the entry.
- **Bib-key note.** `LizzioBosco2026QISR` points to *Quantum Implicit Neural Representations for Novel View Synthesis* (3D scenes). The sentence in §4.1 says quantum INRs "fit a single signal or image". That is fine for a scene, but "signal, image or scene" would be more exact.
- **Referee risk (already listed in §5.5 of the paper).** The INR literature always compares against a matched-size classical network (SIREN or a Fourier-feature MLP). The paper states this is missing. That is honest; a referee may still ask for it. Adding it would be a *new result*, so it is for the student and the PI to decide.

---

## 4. Tensor networks for images and MPS as a front end for VQCs (C1, C8)

**State of the field.**
- *MPS/TN models of image data:* Stoudenmire & Schwab 2016; Han et al. 2018; Ran et al. 2020; Guala et al. 2023; review by Rieser, Köster & Raulf (Proc. R. Soc. A, 2023).
- *Entanglement of image data:* Latorre 2005; Martyn, Vidal, Roberts & Leichenauer 2020 (entanglement of MNIST in MPS classifiers); Lu, Kanász-Nagy, Kukuljan & Cirac (PRA 2025, "Tensor networks and efficient descriptions of classical data"), which shows that natural-image data has low mutual information and is efficiently described by TNs; Jobst et al. (Quantum 2024), which shows images with decaying Fourier spectra are well approximated by low-bond-dimension MPS.
- *MPS feeding a VQC:* Chen, Huang, Hsing & Kao (MLST 2021) — an MPS feature extractor followed by a VQC, trained end to end, beating PCA as the front end on MNIST/Fashion-MNIST. Dilip, Liu, Smith & Pollmann (PRR 2022) — MPS compression of images for loading into circuits.
- *MPS as circuit initialisation:* Dborin et al. (QST 2022); Rudolph et al. (Nat. Commun. 2023).

**Where PCAE sits.** PCAE uses MPS as a *classical, quantum-inspired* preprocessing whose features (local expectations, NN correlators, bipartite entropies) set circuit angles through a linear layer. This is closest to Chen et al. 2021, with two differences: PCAE's MPS is not trained, and the task is reconstruction rather than classification.

**Already cited.** Stoudenmire2016, Han2018, Ran2020TNCS, Guala2023, Cirac2021, Latorre2005, Araz2022, Shin2024Dequant.

**Gaps.**
- **[MUST] Chen et al. 2021 (MLST).** It is the direct precedent for "MPS features → VQC". The intro's TN-circuit-classifier paragraph should cite it and state the difference (fixed SVD-compressed MPS, reconstruction target).
- **[MUST] Martyn et al. 2020 and Lu et al. 2025.** §5.2 states that "natural image patches ... are low-entanglement, locally structured data". At present that sentence has no citation. These two papers support it directly; Jobst et al. 2024 adds the Fourier-spectrum reason.
- **[SHOULD] Dilip et al. 2022; Jobst et al. 2024.** §2 "Why MPS features" cites Latorre and Stoudenmire only. These two are the modern references for MPS compression of images for quantum use.
- **[SHOULD] Rieser et al. 2023** — one review citation for TN–QML.
- **[OPTIONAL] Dborin et al. 2022; Rudolph et al. 2023** — MPS used to initialise PQCs; a different use of MPS, worth one clause to show the distinction.

---

## 5. Measured observables as features: feature maps, kernels, reservoirs (C1, C6)

**State of the field.** Using measured expectation values as a feature vector for a classical model is standard: quantum-enhanced feature maps (Havlíček et al. 2019), projected quantum kernels built from one- and two-body reduced density matrices (Huang et al. 2021), and quantum reservoir computing / extreme learning machines, where single- and two-spin Pauli expectations are read out linearly, sometimes with a quadratic expansion (Fujii & Nakajima 2017; Kornjača et al. 2024, 108 neutral-atom qubits; quantum extreme learning machines for image classification, [arXiv:2409.00998](https://arxiv.org/abs/2409.00998) [VERIFY authors]). Classical shadows (Huang, Kueng & Preskill 2020) give sample-efficient estimation of such local observables.

**Where PCAE sits.** The "Pauli-correlator latent space" is a trained-circuit version of this readout. §4.8's result (a linear readout cannot form z_i z_j; adding the product makes it linearly representable) is the same observation the reservoir-computing literature makes when it adds quadratic output expansions.

**Already cited.** Havlicek2019, Huang2021PowerData, Huang2020Shadows, QuantumKernelDefect2022.

**Gaps.**
- **[SHOULD] One reservoir/QELM citation (Kornjača et al. 2024, or a review).** It shows that measured one- and two-body correlators as a linear-readout feature vector is established practice, which helps the paper's own framing that novelty lies in the assembly.
- **[SHOULD] Schuld, Sweke & Meyer 2021 (PRA).** VQC outputs are truncated Fourier series in the encoded inputs. PCAE encodes angles with R_X and adds positional R_Y; this reference is the natural one for why the correlator map is a smooth, band-limited function of the MPS features, and it links to the random-Fourier-feature dequantisation results already cited (Sweke2025).
- **Wording note for §4.8.** Say explicitly that the necessity result is a property of linear readouts, well known for polynomial feature expansions. The paper already says a classical quadratic model would also fit. One citation (a reservoir paper with quadratic readout) makes this read as an acknowledged point rather than a new finding.

---

## 6. Patch-based and structured quantum vision models (C1)

**State of the field.** Quanvolutional networks apply small circuits to image patches (Henderson et al. 2020). Patch-based quantum GANs generate images patch by patch on hardware (Huang et al., PR Applied 2021; PQWGAN, Tsang et al., IEEE TQE 2023). QCNNs (Cong 2019; Hur 2022), quantum capsule networks (Liu 2022) and quantum vision transformers (Cherrat et al., Quantum 2024) complete the picture.

**Where PCAE sits.** PCAE shares the patch decomposition with quanvolution and the patch-QGAN line (one small circuit per patch, classical assembly). Neither is cited, although both are the standard references for "handle qubit limits by patching".

**Already cited.** Cong2019, Hur2022, Liu2022, Slabbert2024.

**Gaps.**
- **[SHOULD] Henderson et al. 2020** and **Huang et al. 2021**. One sentence in the intro: PCAE follows the patch-wise design of quanvolutional networks and patch QGANs.
- **[OPTIONAL] Tsang et al. 2023; Cherrat et al. 2024.**

---

## 7. Symmetry: group averaging and equivariant QNNs (C5)

**State of the field.**
- *Classical:* group-equivariant CNNs (Cohen & Welling 2016), E(2)-steerable CNNs (Weiler & Cesa, NeurIPS 2019), geometric deep learning (Bronstein 2021). Symmetrisation by averaging over the group (the Reynolds operator) is the textbook way to make any map invariant/equivariant; frame averaging (Puny et al., ICLR 2022) and learned canonicalisation (Kaba et al., ICML 2023) generalise it.
- *Quantum:* EQNN theory (Nguyen et al. 2024; Schatzki et al. 2024), symmetry in VQML (Meyer et al. 2023), rotationally equivariant QML (West et al. 2024), C4-invariant image classifiers (San Sebastián Sein, Cañizo & Orús 2025), approximately p4m-equivariant QCNNs for images (Chang, Grossi, Le Saux & Vallecorsa, IEEE QCE 2023).

**Where PCAE sits.** PCAE's D4 construction is post-measurement Reynolds averaging. The paper already says this is not a quantum property. That is the correct framing.

**Already cited.** Cohen2016, Bronstein2021, Meyer2023, Nguyen2024EQNN, Schatzki2024, West2024 (+ erratum), SanSebastianSein2025.

**Gaps.**
- **[MUST] Puny et al. 2022 (frame averaging).** §4.7 derives the equivariance of Φ_D4 in-line. The contribution list calls it a "proposition". A referee from the equivariance community will point out that this is the group-averaging operator. Citing Puny et al. (and optionally Yarotsky 2022) and calling it "a standard group-averaging construction applied to the observable grid" pre-empts the objection. The paper's "Reynolds averaging" wording is already right; it only lacks the reference.
- **[SHOULD] Chang et al. 2023.** The only QNN paper that targets p4m (which contains D4) symmetry on images. It is the direct quantum comparator for §4.7 and should sit next to San Sebastián Sein 2025.
- **[OPTIONAL] Weiler & Cesa 2019.**

---

## 8. Dequantisation, classical surrogates and simulability (C3)

This theme matters most for the current draft, because the held-out result (C3) is evaluated on a *closed-form surrogate*.

**State of the field.**
- *Classical surrogates of trained QML models:* Schreiber, Eisert & Meyer (PRL 2023) define a "classical surrogate" as a classical model obtained from a trained quantum model that reproduces its input–output map, and argue that the surrogate ansatz is the natural benchmark a quantum model must beat. Follow-ups: Jerbi et al., "Shadows of quantum machine learning" (Nat. Commun. 2024); Hernicht et al. 2025 (scalable surrogates); Nair & Ferrie 2026 (local tensor-train surrogates with error certificates).
- *Random Fourier features:* Landman et al. (ICLR 2023); Sweke et al. (Quantum 2025).
- *Simulability and trainability:* Cerezo et al., "Does provable absence of barren plateaus imply classical simulability?" (Nat. Commun. 2025); Pauli propagation (Rudolph et al. 2025; Fontana et al., npj QI 2025); QCNNs are effectively classically simulable (Bermejo et al. 2026).
- *Benchmarking:* Bowles, Ahmed & Schuld 2024; Schuld & Killoran, "Is quantum advantage the right goal for QML?" (PRX Quantum 2022).

**Already cited.** Tang2019, Huang2021PowerData, Gyurik2023, Shin2024Dequant, Sweke2025, Bermejo2026, Bowles2024, McClean2018, Larocca2025.

**Gaps and risks.**
- **[MUST] Schreiber, Eisert & Meyer 2023 — terminology risk.** The paper uses "closed-form surrogate" for a hand-built analytic feature map (18 patch statistics + 6 local-pair products + sines + positions) that is *not* derived from, or fitted to, the trained circuit. In the QML literature "classical surrogate" means Schreiber's construction. A referee may read "surrogate of the PCAE2D observable map" as "a faithful classical copy of the trained circuit", which the supplement's real-circuit result (11.57 dB, at the random-latent floor) shows it is not. Recommendation: cite Schreiber et al. once in §4.3 and add one clause that the surrogate here is a hand-specified analytic stand-in, not a classical surrogate in their sense. Alternatively rename it ("closed-form proxy" or "analytic stand-in"). This is a wording change only.
- **[MUST] Cerezo et al. 2025.** §5.5 cites barren plateaus (McClean, Larocca) as the scaling concern. The present consensus pairs trainability with simulability; a six-qubit paper that says "whether the boundary moves at larger scale is open" should cite this.
- **[SHOULD] Jerbi et al. 2024.** The hardware pilot trains classically, then deploys on hardware; the "shadow model" line is the reverse (train quantum, deploy classical). One clause helps a reader place the pipeline.
- **[SHOULD] Landman et al. 2023.** The original RFF dequantisation paper; Sweke2025 is cited without it.
- **[SHOULD] Schuld & Killoran 2022.** The paper now frames itself as "a working architecture, no advantage claim". This is the standard reference for that position.
- **[OPTIONAL] Pauli propagation (Rudolph et al. 2025); Nair & Ferrie 2026; Hernicht et al. 2025.**

---

## 9. Hardware execution and error mitigation (C4)

**State of the field.** Error-mitigated expectation values at scale (Kim et al., Nature 2023); zero-noise extrapolation (Temme, Bravyi & Gambetta, PRL 2017); twirled readout-error extinction, TREX (van den Berg, Minev & Temme, PRA 2022). Closest application study: Singh, Jin & Merz, "Benchmarking MedMNIST dataset on real quantum hardware" (arXiv 2025; *Scientific Reports* 2026), which trains on a simulator and runs *inference* on 127-qubit IBM hardware with error suppression and mitigation, on the same MedMNIST family PCAE uses. Patch QGANs were also run on superconducting hardware (Huang et al. 2021).

**Where PCAE sits.** Same "train in simulation, replay on hardware" pattern as Singh et al., but for reconstruction, unmitigated, with a calibrated-noise sweep and cross-backend repeats.

**Already cited.** Huang2020Shadows, IonQBackends. No error-mitigation reference is cited.

**Gaps.**
- **[MUST] Singh, Jin & Merz 2026.** A published hardware-inference study on MedMNIST. Without it, a referee may think the paper is unaware of the closest hardware comparator. It also contrasts usefully: they used mitigation, PCAE did not.
- **[SHOULD] Kim et al. 2023; Temme et al. 2017; van den Berg et al. 2022.** The paper says "dynamical decoupling, Pauli twirling and readout-error mitigation were not enabled" and lists mitigation as future work. One citation group there shows what the omitted methods are.
- **[OPTIONAL] Huang et al. 2021** — earlier image-model hardware demonstration (also under §6).

---

## 10. Hamiltonian / energy terms as auxiliary objectives (C7)

**State of the field.** Parameterised Hamiltonian learning with circuits (Shi et al., TPAMI 2022) treats the Hamiltonian as the learned object. Energy-style regularisers in hybrid models are less studied. A December 2025 preprint (Strnadel, arXiv 2512.12581) adds a VQE-inspired energy term to an AC-GAN on MNIST and reports, after ablations, that it "provides no measurable causal benefit beyond trivial classical regularizers" and that classical variants match or beat it.

**Where PCAE sits.** §5.1 finds the signed energy term *lowers* PSNR (38.63 vs 44.56 dB), and a contrastive variant recovers part of the gap. This is the same direction as Strnadel 2025.

**Already cited.** Shi2022.

**Gaps.**
- **[SHOULD] Strnadel 2025.** Citing an independent negative result for a similar energy term makes §5.1 read as consistent with the field, not as an isolated weakness. It also supports reporting the term as an auxiliary objective plus readout.
- **[OPTIONAL] LeCun et al. 2006 (energy-based learning tutorial)** — for the contrastive variant: the collapse of an unsigned energy term is the classic EBM failure mode that contrastive objectives address. §5.1 already describes this mechanism in words.

---

## 11. Statistical methodology (C3)

**State of the field.** TOST (Schuirmann 1987; Lakens 2017); Holm step-down correction (Holm 1979); Wilcoxon signed-rank test (Wilcoxon 1945); bootstrap CIs (Efron & Tibshirani 1993). For ML benchmarks: variance across seeds and splits (Bouthillier et al., MLSys 2021); multiple-dataset comparisons (Demšar, JMLR 2006). On margins chosen after seeing data: Campbell & Gustafson (arXiv 1807.03413; published in *Meta-Psychology*) and Koobs & Koning (arXiv 2603.16213, 2026), who propose reporting a data-dependent margin instead.

**Already cited.** Schuirmann1987, Lakens2017.

**Gaps.**
- **[MUST] Holm 1979.** The paper applies Holm correction in the TOST table and in the text, but does not cite it.
- **[SHOULD] Wilcoxon 1945; Efron & Tibshirani 1993.** Both tests are used in §4.3 and §5.3 without citation.
- **[SHOULD] Campbell & Gustafson.** The ±1 dB margin is described as "a declared choice"; the earlier "pre-declared" wording was removed because the margin was fixed together with the results. One citation to the post-specified-margin literature, with one sentence that the margin was not pre-registered, is the transparent way to report it. Koobs & Koning 2026 offers the alternative (report the smallest margin the data support: from the 90 % CIs this can be read off the table without new computation, but writing the number into the paper counts as a new number → student note).
- **[OPTIONAL] Bouthillier et al. 2021** — for the seed-level estimand and n = 5 seeds.

---

## 12. Nearest-neighbour map (for the intro's novelty paragraph)

| Work | Input to circuit | Latent handed on | Pair correlators | Task | Symmetry | Hardware | Relation to PCAE |
|---|---|---|---|---|---|---|---|
| Romero et al. 2017; Wang et al. 2024 | Pixel amplitudes | Quantum register | — | Compression | — | Sim. | Different family (register QAE) — cited |
| **Sakhnenko et al. 2022** | Classical-encoder output | ⟨Z⟩ per qubit → classical decoder | No | Anomaly detection | — | Sim. | **Closest bottleneck pattern — not cited** |
| **Chen et al. 2021 (MLST)** | Trainable MPS features | Measured outputs → classifier | No | Classification | — | Sim. | **Closest MPS→VQC front end — not cited** |
| Henderson et al. 2020 | Image patch | Per-patch scalar features | — | Classification | — | Sim. | Patch-wise circuits — not cited |
| Huang et al. 2021; Tsang et al. 2023 | Latent noise | Per-patch pixels | — | Generation | — | HW (2021) | Patch-wise image model on HW — not cited |
| Zhao et al. 2024 (QIREN); Fujihashi 2024; Wang et al. 2025 (QVF) | Coordinates | Signal value | — | Per-signal fitting | — | Sim. | Per-image fitting line — QIREN, QINC cited; QVF not |
| Havlíček 2019; Huang 2021 | Data | Kernel / projected features | Two-body RDMs (Huang) | Classification | — | HW (2019) | Observable-as-feature — cited |
| Chang et al. 2023; San Sebastián Sein 2025 | Image | Class | — | Classification | p4m / C4 in circuit | Sim. | In-circuit symmetry vs PCAE post-hoc — only 2025 cited |
| Singh, Jin & Merz 2026 | MedMNIST image | Class | — | Classification | — | IBM HW, mitigated | Closest hardware-inference study — not cited |
| **PCAE** | Fixed SVD-MPS patch features | 19–27-D local + pair Pauli correlators → MLP | Yes (ZZ, or ZZ/XX/YY) | Per-image reconstruction | D4 by post-measurement averaging | IBM HW replay, unmitigated | — |

**Reading of the table.** Every single ingredient of PCAE has a precedent. The paper's own position ("the contribution is this specific assembly") is the defensible one and is consistent with this survey. The novelty paragraph in §1 is currently written against three families; adding Sakhnenko (expectation-value bottleneck) and Chen (MPS→VQC) as named relatives, with one sentence each on the difference, makes the assembly claim stronger, not weaker, because it shows the authors checked the closest work.

---

## 13. Claims at risk from the literature (for the PI's attention)

1. **"Surrogate" terminology (§4.3, abstract-level contribution 3).** Collides with Schreiber et al. 2023. See §8. Wording fix only.
2. **Novelty paragraph (§1).** Omits the two nearest precedents (Sakhnenko 2022; Chen 2021). See §2, §4.
3. **D4 "proposition" (contribution 5).** It is the group-averaging operator; cite Puny et al. 2022. Consider "construction" instead of "proposition". Wording only.
4. **"Natural image patches are low-entanglement" (§5.2).** Uncited. Martyn 2020, Lu et al. 2025, Jobst 2024 support it.
5. **Holm correction uncited (§4.3, TOST table).**
6. **Equivalence margin chosen after the fact.** Already reworded to "a declared choice". A post-specified-margin citation completes the disclosure.
7. **No matched-size classical INR baseline (§5.5 bullet 2).** Standard in the INR literature. Needs a new experiment if it is to be addressed; otherwise the current disclosure stands.
8. **Hardware without mitigation.** Fine for a pilot. Singh et al. 2026 used mitigation on the same dataset family; citing them keeps the comparison honest.
9. **Energy term lowers PSNR (§5.1).** Consistent with Strnadel 2025; citing it helps.

None of these needs a change to any reported number.

---

## 14. Housekeeping in the current bibliography

- `Fujihashi2024QINC` — arXiv → published chapter exists. **[VERIFY]**
- `Bowles2024`, `Gyurik2023`, `Bronstein2021` — cited as arXiv; check whether a journal version now exists before submission. **[VERIFY]**
- `Slabbert2024` — arXiv 2410.18814; check for a published version. **[VERIFY]**
- `LizzioBosco2026QISR` — key name says "QISR"; the paper is 3D-QISR (novel view synthesis). Harmless, but the §4.1 sentence could say "signal, image or scene".
- `IonQBackends` — a documentation page with an access date. Some Springer journals ask for software/URL references in a separate format; check the QMI author guide.

---

## 15. Ranked list of recommended additions

Priority: 1 = add before submission; 2 = add if possible; 3 = optional.
"Where" gives the paper section and the sentence it would support.

| P | Reference | Where | Why |
|---|---|---|---|
| 1 | Schreiber, Eisert & Meyer, *Classical surrogates for quantum learning models*, PRL 131, 100803 (2023). [arXiv:2206.11740](https://arxiv.org/abs/2206.11740) | §4.3 first paragraph | Terminology clash with "surrogate" |
| 1 | Sakhnenko et al., *Hybrid classical–quantum autoencoder for anomaly detection*, Quantum Mach. Intell. (2022) [VERIFY vol.]. [arXiv:2112.08869](https://arxiv.org/abs/2112.08869) | §1 novelty paragraph | Nearest expectation-value-bottleneck autoencoder |
| 1 | Chen, Huang, Hsing & Kao, *An end-to-end trainable hybrid classical–quantum classifier*, Mach. Learn.: Sci. Technol. (2021) [VERIFY]. [arXiv:2102.02416](https://arxiv.org/abs/2102.02416); earlier version [arXiv:2011.14651](https://arxiv.org/abs/2011.14651) | §1 TN-circuit paragraph | MPS → VQC precedent |
| 1 | Puny et al., *Frame averaging for invariant and equivariant network design*, ICLR 2022. [arXiv:2110.03336](https://arxiv.org/abs/2110.03336) | §4.7, contribution 5 | Group averaging is standard |
| 1 | Holm, *A simple sequentially rejective multiple test procedure*, Scand. J. Stat. 6, 65–70 (1979) | §4.3, TOST table | Method used, uncited |
| 1 | Martyn, Vidal, Roberts & Leichenauer, *Entanglement and tensor networks for supervised image classification* (2020). [arXiv:2007.06082](https://arxiv.org/abs/2007.06082) | §5.2 low-entanglement sentence | Supports an uncited claim |
| 1 | Lu, Kanász-Nagy, Kukuljan & Cirac, *Tensor networks and efficient descriptions of classical data*, PRA 111, 032409 (2025). [arXiv:2103.06872](https://arxiv.org/abs/2103.06872) | §5.2 same sentence | Same |
| 1 | Singh, Jin & Merz, *Benchmarking MedMNIST dataset on real quantum hardware*, Sci. Rep. (2026) [VERIFY vol.]. [arXiv:2502.13056](https://arxiv.org/abs/2502.13056) | §4.4 | Closest hardware-inference study |
| 1 | Cerezo et al., *Does provable absence of barren plateaus imply classical simulability?*, Nat. Commun. 16, 7907 (2025). [doi:10.1038/s41467-025-63099-6](https://doi.org/10.1038/s41467-025-63099-6) | §5.5 last bullet | Scaling caveat |
| 2 | Jobst et al., *Efficient MPS representations and quantum circuits from the Fourier modes of classical image data*, Quantum 8, 1544 (2024). [arXiv:2311.07666](https://arxiv.org/abs/2311.07666) | §2 "Why MPS features"; §5.2 | Modern MPS–image reference |
| 2 | Dilip, Liu, Smith & Pollmann, *Data compression for quantum machine learning*, PRR 4, 043007 (2022) | §2 "Why MPS features" | MPS compression of images for QML |
| 2 | Wang, Theobalt & Golyanik, *Quantum Visual Fields with neural amplitude encoding*, NeurIPS 2025. [arXiv:2508.10900](https://arxiv.org/abs/2508.10900) | §4.1 | Quantum INR at a top ML venue |
| 2 | Chang, Grossi, Le Saux & Vallecorsa, *Approximately equivariant QNN for p4m group symmetries in images*, IEEE QCE 2023. [arXiv:2310.02323](https://arxiv.org/abs/2310.02323) | §4.7 | Quantum image model with D4-containing symmetry |
| 2 | Henderson et al., *Quanvolutional neural networks*, Quantum Mach. Intell. 2, 2 (2020) | §1 | Patch-wise circuits |
| 2 | Huang et al., *Experimental quantum GANs for image generation*, PR Applied 16, 024051 (2021) | §1; §4.4 | Patch-wise image model on hardware |
| 2 | Schuld, Sweke & Meyer, *Effect of data encoding on the expressive power of VQML models*, PRA 103, 032430 (2021) | §2 or §4.8 | Fourier-series view of the correlator map |
| 2 | Landman et al., *Classically approximating VQML with random Fourier features*, ICLR 2023. [arXiv:2210.13200](https://arxiv.org/abs/2210.13200) | §5.2 next to Sweke2025 | Original RFF dequantisation |
| 2 | Jerbi et al., *Shadows of quantum machine learning*, Nat. Commun. 15, 5676 (2024) | §5.2 | Train/deploy split in QML |
| 2 | Schuld & Killoran, *Is quantum advantage the right goal for QML?*, PRX Quantum 3, 030101 (2022) | §1 or §5.2 | Framing without an advantage claim |
| 2 | Kim et al., *Evidence for the utility of quantum computing before fault tolerance*, Nature 618, 500 (2023); Temme, Bravyi & Gambetta, PRL 119, 180509 (2017); van den Berg, Minev & Temme, PRA 105, 032620 (2022) | §4.4 "not enabled" sentence; Conclusion future work | Name the omitted mitigation methods |
| 2 | Strnadel, *Differentiable energy-based regularization in GANs: VQE-inspired auxiliary losses* (2025). [arXiv:2512.12581](https://arxiv.org/abs/2512.12581) | §5.1 | Independent negative result for an energy term |
| 2 | Wilcoxon (1945), Biometrics Bull. 1, 80–83; Efron & Tibshirani, *An Introduction to the Bootstrap* (1993) | §4.3 | Methods used, uncited |
| 2 | Campbell & Gustafson, *What to make of equivalence testing with a post-specified margin?* [arXiv:1807.03413](https://arxiv.org/abs/1807.03413) [VERIFY venue] | §4.3 margin sentence | Transparent margin reporting |
| 2 | Kornjača et al., *Large-scale quantum reservoir learning with an analog quantum computer* (2024). [arXiv:2407.02553](https://arxiv.org/abs/2407.02553) | §1 or §4.8 | Pauli correlators as linear-readout features |
| 3 | Rieser, Köster & Raulf, *Tensor networks for quantum machine learning*, Proc. R. Soc. A 479, 20230218 (2023) | §1 | TN–QML review |
| 3 | Dborin et al., QST 7, 035014 (2022); Rudolph et al., Nat. Commun. 14, 8367 (2023) | §2 | MPS as PQC initialisation (contrast) |
| 3 | Lo, Hsu & Kuo (2025). [arXiv:2502.07667](https://arxiv.org/abs/2502.07667) | §1 | Register QAEs reconstruct poorly |
| 3 | Tsang et al., PQWGAN, IEEE Trans. Quantum Eng. 4 (2023); Cherrat et al., *Quantum vision transformers*, Quantum 8, 1265 (2024) | §1 | Wider quantum-vision context |
| 3 | Weiler & Cesa, *General E(2)-equivariant steerable CNNs*, NeurIPS 2019; Kaba et al., *Equivariance with learned canonicalization functions*, ICML 2023 | §4.7 | Classical symmetry context |
| 3 | Rudolph et al., *Pauli propagation* (2025) [arXiv:2505.21606](https://arxiv.org/abs/2505.21606); Nair & Ferrie (2026) [arXiv:2604.25631](https://arxiv.org/abs/2604.25631) | §5.2 | Recent simulation/surrogate tools |
| 3 | Koobs & Koning (2026) [arXiv:2603.16213](https://arxiv.org/abs/2603.16213); Bouthillier et al., MLSys 2021 | §4.3 | Margin and seed-variance methodology |
| 3 | Eren (2026) [arXiv:2603.06755](https://arxiv.org/abs/2603.06755); Ganguly et al. (2026) [arXiv:2606.27411](https://arxiv.org/abs/2606.27411); Jadhav et al. (2026) [arXiv:2609.28387](https://arxiv.org/abs/2609.28387) | §1 | Very recent hybrid image-reconstruction preprints |

Adding all priority-1 and priority-2 items gives roughly 25 new entries (about 90 in total), which is normal for a QMI article.

---

## 16. Suggested next steps

1. PI decides on the "surrogate" wording (§8 / §13 item 1). This is the only point that touches how a headline result is read.
2. Add the nine priority-1 citations with `\add{}` markup, one sentence each, in the sections listed. No numbers change.
3. Student generates BibTeX from the DOIs for all [VERIFY] entries rather than typing them by hand.
4. Priority-2 items in a second small pass after the PI reads the marked version.
5. Items that would need new numbers (matched classical INR baseline; data-dependent equivalence margin) go to the student as red notes, not into the text.

---

### Sources consulted

- [Schreiber, Eisert & Meyer, PRL 2023](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.131.100803)
- [Jerbi et al., Nat. Commun. 2024](https://www.nature.com/articles/s41467-024-49877-8)
- [Cerezo et al., arXiv:2312.09121 / Nat. Commun. 2025](https://arxiv.org/pdf/2312.09121)
- [Landman et al., ICLR 2023](https://iclr.cc/virtual/2023/poster/11652)
- [Sweke et al., Quantum 2025](https://quantum-journal.org/papers/q-2025-02-20-1640/)
- [Nair & Ferrie, arXiv:2604.25631](https://arxiv.org/abs/2604.25631)
- [Hernicht et al., arXiv:2508.06131](https://arxiv.org/abs/2508.06131)
- [Rudolph et al., Pauli propagation, arXiv:2505.21606](https://arxiv.org/pdf/2505.21606)
- [Schuld & Killoran, PRX Quantum 2022](https://link.aps.org/doi/10.1103/PRXQuantum.3.030101)
- [Schuld, Sweke & Meyer, PRA 2021 (code repository)](https://github.com/XanaduAI/expressive_power_of_quantum_models)
- [Sakhnenko et al., Quantum Mach. Intell. 2022](https://link.springer.com/article/10.1007/s42484-022-00075-z)
- [Lo, Hsu & Kuo, arXiv:2502.07667](https://arxiv.org/html/2502.07667v1)
- [Eren, arXiv:2603.06755](https://arxiv.org/abs/2603.06755)
- [Ganguly, Liang & Makris, arXiv:2606.27411](https://arxiv.org/pdf/2606.27411)
- [Jadhav et al., arXiv:2609.28387](https://arxiv.org/html/2609.28387)
- [Wang, Theobalt & Golyanik, QVF, arXiv:2508.10900](https://arxiv.org/abs/2508.10900)
- [Lizzio Bosco et al., arXiv:2601.05250](https://arxiv.org/abs/2601.05250)
- [Fujihashi & Koike-Akino, QINC, Springer chapter](https://link.springer.com/chapter/10.1007/978-3-032-15931-1_5)
- [Chen et al., MLST 2021](https://iopscience.iop.org/article/10.1088/2632-2153/ac104d); [arXiv:2011.14651](https://arxiv.org/abs/2011.14651v1)
- [Dborin et al., QST 2022](https://iopscience.iop.org/article/10.1088/2058-9565/ac7073)
- [Rudolph et al., Nat. Commun. 2023](https://www.nature.com/articles/s41467-023-43908-6)
- [Dilip et al., PRR 2022](https://link.aps.org/doi/10.1103/PhysRevResearch.4.043007)
- [Jobst et al., Quantum 2024](https://quantum-journal.org/papers/q-2024-12-03-1544/)
- [Martyn et al., arXiv:2007.06082](https://arxiv.org/abs/2007.06082)
- [Lu et al., PRA 2025](https://link.aps.org/doi/10.1103/PhysRevA.111.032409)
- [Rieser, Köster & Raulf, Proc. R. Soc. A 2023](https://royalsocietypublishing.org/rspa/article/479/2275/20230218/54586/Tensor-networks-for-quantum-machine-learningTensor)
- [Henderson et al., Quantum Mach. Intell. 2020](https://link.springer.com/article/10.1007/s42484-020-00012-y)
- [Huang et al., PR Applied 2021](https://link.aps.org/doi/10.1103/PhysRevApplied.16.024051)
- [Cherrat et al., Quantum 2024](https://quantum-journal.org/papers/q-2024-02-22-1265/)
- [Chang et al., arXiv:2310.02323](https://arxiv.org/abs/2310.02323)
- [Puny et al., arXiv:2110.03336](https://arxiv.org/abs/2110.03336)
- [Kornjača et al., arXiv:2407.02553](https://arxiv.org/abs/2407.02553)
- [Singh, Jin & Merz, Sci. Rep. 2026](https://www.nature.com/articles/s41598-026-35605-3)
- [Strnadel, arXiv:2512.12581](https://arxiv.org/abs/2512.12581)
- [Koobs & Koning, arXiv:2603.16213](https://arxiv.org/html/2603.16213)
- [Campbell & Gustafson, arXiv:1807.03413](https://arxiv.org/abs/1807.03413)
