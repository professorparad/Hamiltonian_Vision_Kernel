# TODO — Finish the paper for IEEE TQE

**Target:** IEEE Transactions on Quantum Engineering.
**Framing A:** our architecture, honestly characterized, hardware pilot as the centerpiece.
Do **A → B → C → D**. A and B gate submission. In the meeting, say **done / not done / blocked** per task.

---

## A. Verify before shipping (do first)
- [ ] **A0** — Fill `results-core-map.md`: for every paper table/figure, give the script + command + params + artifact. Every `FILL:` is a task; a row with no source doesn't ship.
- [ ] **A1** — Table I (map R1): only PneumoniaMNIST has a source file. Re-run the other 5 datasets into one summary, or delete those rows.
- [ ] **A2** — "Monalisa 40.70 vs 34.72 dB" (§V-D) is backed by no file. Reproduce it or delete the sentence.
- [ ] **A3** — Zero-shot 7.78 / 28.31 dB (§IV-B) has no source. Point to a file or re-run.

## B. Commit missing evidence
- [ ] **B1** — `run_core_multiseed_240.py` is **not in the repo at all** (not just untracked). Locate it or confirm the current paper no longer needs it; if needed, rebuild + commit.
- [ ] **B2** — `verify_shuffle_permutations.py` is also **absent**. Same: locate/rebuild or confirm unused, then commit.
- [ ] **B3** — Narrow `.gitignore` so result CSV/JSON are tracked (`experiments/quantum_contribution/results/` is currently ignored; ignore media only).

## C. Rewrite for Framing A
- [ ] **C1** — Remove every "existing HVK"; state it is our architecture.
- [ ] **C2** — Add one §II paragraph: why MPS features, why Pauli latent, why grid topology.
- [ ] **C3** — Retitle: drop "Physics-Informed" and "Provable Symmetry"; keep hardware.
- [ ] **C4** — Move the hardware pilot to lead Results; make D4 / entanglement / phase "additional diagnostics".
- [ ] **C5** — Delete "advantage" / "significantly" wording not backed by a test.
- [ ] **C6** — Add the **representativeness paragraph**: argue HVK is a representative instance of the "MPS features + Pauli-correlator latent + physics regularizer" design pattern, so its boundary is the pattern's boundary — not an idiosyncratic failure. *(Biggest rejection risk; do not skip.)*

## D. Submission hygiene
- [ ] **D1** — Keep one `.tex`; delete stray tex/pdf copies. **Compile `paper_hvk.pdf` and proofread the rendered paper.**
- [ ] **D2** — Write `REPRODUCE.md`: one script per table/figure.
- [ ] **D3** — Add Table I + topology to `submission_claim_audit.json` (after A1/A2).
- [ ] **D4** — *(Supervisor)* confirm TQE APC/waiver, article type, length.
- [ ] **D5** — Check every shared number agrees between `paper_hvk.tex` and `supplementary_study.tex` (held-out 18.80/18.12, leakage audit, hardware figures).
- [ ] **D6** — Make the repo release-ready and mint a citable snapshot (Zenodo DOI); confirm it runs from clean. *(Depends on B1–B3.)*
- [ ] **D7** — Submission packet: cover letter, author-contribution statement, ORCIDs, 2–3 suggested reviewers.

## E. Fix the bibliography (`literature_review.tex` — verified against primary sources)
- [ ] **E1** — `Chakraborty2018`: change year **2018 → 2022** (IJIT 14(1), 475–489).
- [ ] **E2** — `West2024`: add the erratum **PRX Quantum 6, 020902 (2025)** + Comment (arXiv:2504.16950).
- [ ] **E3** — `Larocca2025`: add vol/pages — **Nat. Rev. Phys. 7, 174–189 (2025)**.
- [ ] **E4** — `QCAE2023`: add authors (Wu, Fu, Zhu, Zhang, Xie, Li) + **Phys. Rev. A 109, 032623 (2024)**.
- [ ] **E5** — `QuantumImageRepClassification2025`: add authors (Parigi, Khosrojerdi, Caruso, Banchi) + **AVS Quantum Sci. 8(1), 013801 (2026)**.
- [ ] **E6** — `Guala2023`: **authors are wrong/fabricated** — replace with Guala, Zhang, Cruz, Riofrío, Klepsch, Arrazola. (vol/page 13, 4427 are correct.)
- [ ] **E7** — Add author names to the ~10 author-less entries (QIPSystematicReview2025, WangQAECompression2024, QAEClassification2025, QCAE2023, HybridGANHighRes2022, MosaiQ2023, QuantumDiffusion2024, HybridLatentDiffusion2025, QuantumKernelDefect2022, QuantumImageRepClassification2025).
- [ ] **E8** — **Audit every remaining author list** — the Guala fabrication means names in this bib can't be trusted without a check.
- [ ] **E9** — Confirm whether the lit-review is even part of the TQE submission; if not, these are lower priority than A–D.

## Guardrails — do NOT undo
- [ ] Keep `cifar_nonlocal_advantage` **out** of the paper (label leak); no "quantum advantage".
- [ ] Keep the phase-transition claim at **16/24**; do not re-inflate.
