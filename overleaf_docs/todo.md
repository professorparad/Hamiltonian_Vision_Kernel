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

## I. QUESTION FOR YOU — please answer before editing anything (2026-09-03)

  First, the good news. I re-derived the held-out statistics from the raw artifacts
  rather than reading them off the manuscript, and everything checks out: the TOST
  t-statistics, p-values and 90% CIs all recompute exactly from
  `tost_equivalence.json`, every hardware PSNR recomputes from its stored MSE, and
  `tab:hvk_pair_diagnostic` matches `full_ablation_summary.json` to 5 significant
  figures. The bibliography is also clean (48 entries, 48 cited, no orphans, no
  undefined keys). That is careful work.

  One thing I want you to confirm before either of us touches the text.

  I1. WERE `raw-linear-classical` AND `local-observables-only` MEANT TO BE TWO
      SEPARATE CONTROLS?

      As the code currently stands they compute the same thing. The base patch
      descriptor from `real_patch_base_features` (in
      `Main2/newHVK/run_newhvk_suite.py`) is exactly 26-D — 12 intensity/gradient
      statistics, 6 low-frequency block means, 8 Fourier position channels — and:

          real_raw_linear_features(base)    = select_same_width(base, 32)
          real_local_observables_only(base) = select_same_width(base[:, :26], 32)

      Since `base` has 26 columns, `base[:, :26]` is all of `base`, so both return
      the same vector zero-padded 26 -> 32. I checked this against the data too: they
      are bit-identical in all 20 held-out cells of `real_cifar_holdout.csv`, and
      they carry identical seed-difference vectors in `tost_equivalence.json`.

      I can see two readings and I do not know which is yours:

        (A) They were MEANT to differ. `local-observables-only` was supposed to be a
            strict subset — local statistics only, WITHOUT the 8 positional channels,
            i.e. `base[:, :18]` — and the `[:, :26]` slice is a bug that silently
            turned it into a copy of the raw control. If so this is a real (small)
            code fix plus a re-run of that one control, and the held-out /
            TOST / multi-dataset tables gain a genuinely distinct row. Nothing else
            in the paper moves.

        (B) They were always the same map, kept as two rows only because they came
            from two different naming conventions in the harness. You did already
            note "(identical by construction here)" at supplement line ~79, which
            points this way. If so, no code changes — it is purely a presentation
            fix: merge the rows so a referee does not read two identical lines as a
            copy-paste error.

      TELL ME WHICH IT IS. Do not start editing until we have settled it, because
      (A) and (B) lead to different work and I do not want you re-running anything
      unnecessarily. If you are unsure, say so and we will look at it together —
      "I do not remember" is a perfectly fine answer here.

      Either way, please also check the same question for the multi-dataset section
      (~line 159), which reports "the local-only control" on five further datasets.
      If (A) holds, is that row the same collapsed map?

  I2. ONE THING THAT NEEDS SAYING WHICHEVER WAY I1 GOES. The methods section (~line
      29) says every retained control has "identical feature width (32-D)". That is
      true of what the readout sees, but this control's real descriptor is 26-D plus
      6 zeros. Worth one clause noting that matching is on the readout-facing width
      and narrower descriptors are zero-padded — which adds no information and leaves
      the 2112 readout parameter count unchanged. (`tab:resource_capacity` is already
      correct: it uses a single "Raw/local controls" row.)

  ---

  REFERENCE — exactly where this shows up (line numbers are current as of this
  commit; table numbers are as rendered in the compiled supplement PDF).

  `overleaf_docs/supplementary_study.tex` — the two identical rows:

    | Table | Label                      | Caption (short)                  | Duplicate rows |
    |-------|----------------------------|----------------------------------|----------------|
    | 2     | `tab:hvk_real_cifar`       | Held-out CIFAR-10 validation layer | lines 90, 91 |
    | 3     | `tab:hvk_real_cifar_stats` | Seed-level paired tests          | lines 112, 113 |
    | 4     | `tab:tost_equivalence`     | TOST equivalence test            | lines 145, 146 |

    In Table 2 the order is "Local observables only" then "Raw-linear classical";
    in Tables 3 and 4 it is the other way round. Both rows carry identical numbers
    in all three.

  Related, in the same file:

    - line  29  — the "identical feature width (32-D)" methods sentence (item I2).
    - line  79  — prose: "Local-observables-only and raw-linear-classical controls
                  (identical by construction here)". Your existing disclosure.
    - line 157  — "Six of the eight controls are statistically equivalent ..." — the
                  count that double-counts if reading (B) is right.
    - line 159  — multi-dataset paragraph, "Against the local-only control ...",
                  referring to Table 5 (`tab:multi_dataset_reconstruction`).
                  This is the one I ask you to check separately.
    - line 238  — Table 8 (`tab:resource_capacity`). Already correct: it uses one
                  merged "Raw/local controls" row. No change needed.
    - line 593  — conclusion, repeats "local-observable and raw-linear controls".

  `overleaf_docs/paper_hvk_springer.tex` — the same count, four places:

    - line  53  — abstract: "resource-matched local-observable and raw-linear
                  classical controls".
    - line 163  — contributions bullet: "... to six of eight resource-matched
                  classical/ablated controls".
    - line 408  — body: "local-observable and raw-linear controls reach 18.80 dB".
    - line 417  — body: "six of eight tested controls are statistically equivalent".

  Source of truth for the numbers, if you want to re-derive any of this yourself:
    - `Main2/newHVK/results/q1_validation/real_cifar_holdout.csv`  (per-seed, per-image)
    - `Main2/newHVK/results/q1_validation/tost_equivalence.json`   (seed_diffs_db)
    - `Main2/newHVK/run_newhvk_suite.py`                           (the feature maps)

  Note: both .tex files are CRLF. A careless regex edit can mangle the line endings
  or silently match nothing — always check `git diff` before committing.

---

## NOT your items — supervisor (Siddhartha) will handle

These are `[FILL]`s in the cover letter — judgment calls, do NOT fill them:
  - Suggested reviewers (2–3 names) — `cover_letter.tex` line ~50.
  - Siddhartha Patra's ORCID — `cover_letter.tex` line ~68 (Sparsho's is filled).
  - arXiv preprint timing (post now vs. on acceptance) and the Zenodo DOI.
