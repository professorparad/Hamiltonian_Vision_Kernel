# Student punch-list — QMI/Springer submission bundle

Newest task block at the bottom. Completed blocks are recorded, not deleted, so the
history is legible.

---

## DONE (verified by supervisor 2026-09-03)

**Block F — results ↔ code map + hardware provenance:**
- F1 ✅ `overleaf_docs/RESULTS_MAP.md` created, re-targeted to the Springer sections.
- F2 🟡 anchor job IDs recorded; `IBM_Cloud/verify_hardware_jobs.py` written. The
  **live IBM-account check is still open** — carried forward as H1 below.
- F3 ✅ R10 (`contrastive+no-energy`, 33.33 dB) closed: the value is in neither Springer
  manuscript, so no artifact is needed; documented in RESULTS_MAP.md §F3.

**Block G — physics review:**
- G1 ✅ Fei2021 miscitation replaced with `Ran2020TNCS` (Ran, Sun, Fei, Su, Lewenstein,
  Phys. Rev. Research 2, 033293, 2020); citation key updated, resolves clean.
- G2 ✅ Topology floor measured directly: a random-VQC control at the identical 90-step
  protocol reaches 12.74/12.64 dB, so the trained 11.7 dB rows sit ~1 dB **below** the
  floor (Wilcoxon p=0.031). Absolutes now explicitly not read as reconstruction quality.
- G3 ✅ Bond-dimension non-monotonicity de-claimed: single-seed, no mechanism attached.

Both manuscripts compile clean: main 24 pp, supplement 27 pp, 0 undefined citations.
Good work — G2 in particular went beyond the ask (ran a real control rather than
just adding a sentence).

---

## H. TO DO NEXT (assigned to student, 2026-09-03)

  H1. RUN THE LIVE IBM-ACCOUNT CHECK. You built `IBM_Cloud/verify_hardware_jobs.py`
      but it has not been run against the service (no credentials on the build machine).
      The jobs are under your IBM Quantum account, so this is yours to run:

          python IBM_Cloud/verify_hardware_jobs.py --write-map

      For every job ID in RESULTS_MAP.md's F2 ledger, confirm: the job retrieves, the
      instance / CRN, the backend (ibm_fez / ibm_marrakesh / ibm_kingston), the shot
      count, and that the decoded PSNR matches the paper. Fill the instance/CRN line and
      the per-job "Account: ok / mismatch" column in RESULTS_MAP.md, then commit. If any
      job no longer resolves or a value disagrees, FLAG it in the map — do not silently
      overwrite. (The IonQ rows are not IBM jobs; verify those in the IonQ console
      separately, or leave them marked "IonQ — not IBM-retrievable".)

  H2. CONFIRM A CLEAN BUILD ON YOUR MACHINE. Recompile both manuscripts from a clean
      state (delete aux first) and confirm 0 undefined citations and that
      `overleaf_docs/assets/` holds the current PDFs:

          cd overleaf_docs
          python ../latex_outputs/compile_tex.py paper_hvk_springer.tex
          python ../latex_outputs/compile_tex.py supplementary_study.tex
          # then move outputs into assets/ per the build rule in EDITS.md

      Supervisor verified this here (24 pp / 27 pp, clean); confirm it on your side
      before we freeze the source.

  H3. When H1–H2 are done, note completion at the top of overleaf_docs/EDITS.md.

---

## NOT your items — supervisor (Siddhartha) will handle

These are in the cover letter as `[FILL]` and are judgment calls, not student tasks —
do NOT fill them:
  - Suggested reviewers (2–3 names) — `cover_letter.tex` line ~50.
  - Siddhartha Patra's ORCID — `cover_letter.tex` line ~68 (Sparsho's is filled).
  - arXiv preprint timing (post now vs. on acceptance) and the Zenodo DOI.
