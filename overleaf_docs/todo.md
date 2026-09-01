A. FIRST **undo** the wrong todo implementation.


F. RESULTS ↔ CODE MAP — re-prepare for the Springer manuscript  (assigned to student, 2026-08-27)

Context: the existing map `TODO/results-core-map.md` is written against the OLD
`paper_hvk.tex` (IEEEtran section labels §IV-A, §V-C…, old table/figure names). The
canonical manuscript is now `overleaf_docs/paper_hvk_springer.tex` (Springer sn-jnl),
which has renumbered sections and some merged/trimmed subsections. Supervisor already
confirmed every headline NUMBER still appears unchanged in the Springer paper (no drift
from the framing/trim edits) — so this is a re-mapping + hardware-provenance task, not a
re-run of any simulation.

  F1. Produce a NEW map beside the Springer sources: `overleaf_docs/RESULTS_MAP.md`
      (do NOT overwrite TODO/results-core-map.md; that stays as the historical map).
      One row per Springer table/figure → driver script + exact command + key params
      + result artifact + status. Re-target every row from the old §IV/§V labels to the
      CURRENT Springer section numbers and the actual table/figure \label names in
      paper_hvk_springer.tex and supplementary_study.tex.

  F2. REAL-HARDWARE CROSS-CHECK (needs the IBM Quantum account — only you have access):
      For every hardware result, log into the IBM Quantum account and confirm, per job:
        - the instance / CRN (record it in the map),
        - the job ID exists and is retrievable,
        - it ran on the stated backend (ibm_fez, ibm_marrakesh, ibm_kingston),
        - at the stated shot count (256 / 1024),
        - and the decoded PSNR matches the paper.
      Job IDs to verify (from results-core-map R3/R5 and the hardware ledger):
        d9ecu34inv1c73aq3qt0 (Monalisa pilot, ibm_fez/256),
        d9edfo2neu4c739ob2ig (CIFAR pilot, one of 4 ibm_fez jobs),
        d9gqm58gk0ls73f219m0 (anchor, e.g. ibm_marrakesh/kingston),
        + the remaining pilot/anchor/ledger IDs.
      Record instance ID + each job ID + backend + shots + PSNR in RESULTS_MAP.md so the
      hardware provenance is auditable from the account, not just from local JSON.

  F3. Close the one open provenance row: R10 `contrastive+no-energy` (33.33 dB) still has
      NO artifact of any kind. Either produce an artifact for it or cut/footnote the value.
      (Everything else in results-core-map.md is already `backed`.)

  F4. When F1–F3 are done, note completion at the top of overleaf_docs/EDITS.md.

---

G. PHYSICS REVIEW — data-dependent items  (assigned to student, 2026-08-27)

Context: supervisor did a full high-effort physics read of both manuscripts + the
.bib. Six text/reference fixes were applied directly (order-parameter rename,
R_ES-tracks-H clause, linear-readout mechanism, entangling-ansatz-vs-pair-observable
clause, Hamiltonian-is-read-only in main text, and the Chakraborty2018 author-name
correction — see EDITS.md). The three items below need code/data/citation-manager
access the supervisor does not have, so they are yours:

  G1. Fei2021 citation looks like a MISCITATION. The .bib cites "S.-M. Fei,
      Physical Science & Biophysics Journal 5(1):000174 (2021)" for tensor-network
      compressed sensing. The canonical, peer-reviewed paper on that exact topic is
      Ran, Sun, Fei, Su, Lewenstein, "Tensor network compressed sensing with
      unsupervised machine learning", Phys. Rev. Research 2, 033293 (2020)
      (arXiv:1907.10290). Confirm which you actually meant to cite; if the intended
      reference is the TNCS paper, replace the entry with the PRResearch one (much
      stronger venue). Do NOT keep the "Physical Science & Biophysics Journal"
      version unless you can confirm it is a real, intended, peer-reviewed source.

  G2. Topology real-circuit absolute PSNR is very low (11.73 / 11.57 dB;
      supplementary_study.tex Table tab:topology_real_circuit, 90-step budget). That
      sits close to the random-latent floor (~11-15 dB elsewhere in the study). The
      HVK1D-minus-HVK2D DIFFERENCE (0.16 dB) is fine and the reduced budget is
      disclosed, but add ONE sentence stating whether 11.7 dB is meaningfully above
      the random-VQC control AT THIS SAME 90-step budget (check the artifact / run a
      matched random-VQC row if needed). If it is not clearly above the floor, say so
      — otherwise a referee will ask whether the run simply did not train.

  G3. The MPS bond-dimension NON-MONOTONICITY (chi=1,2 outperform chi=4 at 200
      steps; supplementary_study.tex §Scaling, Fig capacity_scaling_sweeps) is
      currently single-seed and reported only as a "directional diagnostic." It is
      physically interesting: if real, it says the reconstruction target is
      low-entanglement and chi=4 over-parameterizes (consistent with the
      dequantization framing). Decide: either (a) confirm it survives multiple seeds
      and add one interpretive sentence, or (b) keep it explicitly single-seed and
      make no physical claim. Do not leave it ambiguous.

  G4. When G1-G3 are done, note completion at the top of overleaf_docs/EDITS.md.