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