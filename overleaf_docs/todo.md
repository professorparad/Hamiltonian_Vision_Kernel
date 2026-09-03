# Student punch-list — QMI/Springer submission bundle

Only open items live here. Completed work is recorded in `EDITS.md` and git history.

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

## I. TO DO NEXT (assigned to student, 2026-09-03)

  Supervisor audit finding. I re-derived the held-out statistics from the raw
  artifacts. Everything checks out -- the TOST t-statistics, p-values and 90% CIs
  all recompute exactly from `tost_equivalence.json`, the hardware PSNRs all
  recompute from their stored MSEs, and `tab:hvk_pair_diagnostic` matches
  `full_ablation_summary.json` to 5 s.f. Good work. One presentation problem:

  I1. MERGE THE DUPLICATED CONTROL. `raw-linear-classical` and
      `local-observables-only` are THE SAME FEATURE MAP, not two controls. The base
      patch descriptor is exactly 26-D (12 stats + 6 low-freq block means + 8 Fourier
      position channels, see `real_patch_base_features` in
      `Main2/newHVK/run_newhvk_suite.py`), so

          real_raw_linear_features(base)    = select_same_width(base, 32)
          real_local_observables_only(base) = select_same_width(base[:, :26], 32)

      are identical, both zero-padded 26 -> 32. They are bit-identical in all 20
      held-out cells and carry identical seed-difference vectors in
      `tost_equivalence.json`.

      You already spotted this -- supplement line ~79 says "(identical by
      construction here)" -- so this is a presentation fix, NOT a re-run. Nothing
      downstream changes and no headline number moves. In `supplementary_study.tex`:

        (a) Merge the two rows into one in all three tables: `tab:hvk_real_cifar`
            (lines ~90-91), `tab:hvk_real_cifar_stats` (~112-113), and
            `tab:tost_equivalence` (~145-146). Label the merged row something like
            "Raw-linear / local-observables control".
        (b) Add a short note under each of those tables giving the reason (the 26-D
            descriptor, zero-padded to the shared 32-D width), so a referee who sees
            the merge understands it immediately.
        (c) Line ~157: "Six of the eight controls are statistically equivalent" is
            now a double-count. The true numbers are FIVE of SEVEN distinct controls
            (5 equivalent + 2 not-equivalent = 7). Fix the sentence and drop the
            duplicated name from the list that follows it.
        (d) Line ~79: promote "(identical by construction here)" from a parenthetical
            to an explicit clause stating the mechanism.
        (e) Line ~593 (conclusion): singularize "local-observable and raw-linear
            controls" to the one merged control.

      Then the same count fix in `paper_hvk_springer.tex`: the abstract (~line 53),
      the contributions bullet (~163, "six of eight"), and the two body passages
      (~408 and ~418, "six of eight tested controls"). Keep the register rules --
      still "competitive," still TOST-only, no "beats."

      Watch out: both .tex files are CRLF. A careless regex edit can mangle line
      endings or silently match nothing -- check `git diff` before you commit.

  I2. ONE HONESTY CLAUSE ON RESOURCE-MATCHING. Section ~line 29 says every control
      has "identical feature width (32-D)". True for what the readout sees, but this
      control's real descriptor is 26-D plus 6 zeros. Add a clause saying matching is
      on the width the readout sees and narrower descriptors are zero-padded, which
      adds no information and leaves the 2112 readout parameter count unchanged.
      (`tab:resource_capacity` is already correct -- it uses a single "Raw/local
      controls" row -- so no change needed there.)

  I3. RECOMPILE BOTH and confirm 0 errors / 0 undefined citations / 0 undefined
      references before committing. I test-applied these edits locally to check they
      are safe: main paper stays at 24 pp, supplement goes 27 -> 28 pp (the three added
      notes push it over a page break). Both compiled with 0 errors / 0 undefined. If
      you land somewhere different, something else changed -- check before committing.
      Record the change at the top of `EDITS.md`.

---

## NOT your items — supervisor (Siddhartha) will handle

These are `[FILL]`s in the cover letter — judgment calls, do NOT fill them:
  - Suggested reviewers (2–3 names) — `cover_letter.tex` line ~50.
  - Siddhartha Patra's ORCID — `cover_letter.tex` line ~68 (Sparsho's is filled).
  - arXiv preprint timing (post now vs. on acceptance) and the Zenodo DOI.
