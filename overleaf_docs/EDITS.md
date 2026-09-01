# Edit log — QMI/Springer submission bundle

Running record of every edit to the manuscripts in this folder, newest first.
Target venue: **Quantum Machine Intelligence** (Springer, journal 42484). Free to
publish via the subscription route (no APC); arXiv preprint for free-to-read.

**Folder layout:** source files (`*.tex`, `sn-bibliography.bib`, `sn-jnl.cls`,
`sn-basic.bst`) at root; figures and all compiled/aux outputs under `assets/`
(`assets/figures/`, `assets/*.pdf`, etc.). Figures resolve via
`\graphicspath{{assets/}}` in both manuscript preambles, so `\includegraphics`
paths stay `figures/...`.

**Build rule:** compile from the folder root (NOT `-outdir=assets`, which breaks
bibtex's search for the `.bst`/`.bib`), then move outputs into `assets/`:

    cd overleaf_docs
    python ../latex_outputs/compile_tex.py paper_hvk_springer.tex
    for e in pdf aux bbl blg fdb_latexmk fls log out; do mv -f paper_hvk_springer.$e assets/; done

**Current state:** `paper_hvk_springer.pdf` = 24 pp, 0 undefined citations,
abstract 242 words (QMI wants 150-250). `supplementary_study.pdf` = 26 pp, 0
undefined citations, self-contained `thebibliography` (does not use the `.bib`).

---

## 2026-09-01

### F1 done, F3 closed, F2 half done (item F, student)
**F1 — `overleaf_docs/RESULTS_MAP.md` written.** One row per Springer table and figure:
driver script, exact command, key parameters, retained artifact, status. All 8 tables and
6 figures of the paper and all 15 tables and 8 figures of the supplement are covered, with
numbering taken from the compiled PDFs' `.aux` `\newlabel` entries rather than guessed, and
a `Prior ID` column carrying the old `R1`-`R14` IDs so it reads side by side with
`TODO/results-core-map.md` (which is untouched, and was restored after being deleted --
F1 and F3 both refer to it). Two items are `descriptive` (`tab:variants`,
`tab:differentiation` -- definitions, nothing to reproduce); one is `partial`
(`tab:hamiltonian_controls` / `tab:hamiltonian_reproducibility`, whose gap the supplement
already discloses at length); everything else is `backed`, with the artifact value quoted
against the printed one where it was recomputed.

**F3 -- closed, no artifact needed.** R10's open row was `contrastive+no-energy` (33.33 dB).
That value is not in either Springer manuscript: `33.33` appears nowhere in
`paper_hvk_springer.tex` or `supplementary_study.tex`. What survives is §3.7's account of
the *withdrawal*, which quotes `32.24 -> 33.30` and contrastive `32.84` precisely to say
they are superseded and that three of the old table's six rows are not independently
reproducible. Those are cited as history, not as results. Nothing to produce, nothing to
cut; the row reopens only if a withdrawn value is ever promoted back into a claim.

**F2 -- the map is ready, the account check is not done.** Every job identifier the
manuscripts cite is now in one ledger in RESULTS_MAP.md -- 25 jobs (5 reconstruction
pilot, 4 repeated-execution anchors, 16 archived replays) with backend, shots, circuit
count and PSNR read back from the retained JSON, all marked `pending` in an `Account`
column. `IBM_Cloud/verify_hardware_jobs.py` was written to close that column: it retrieves
each job, compares backend/shots/status against what the manuscripts claim, prints the
account's instance/CRN, and with `--write-map` rewrites the column in place.

It has **not been run against the service**: this machine has no IBM Quantum credentials
(no `~/.qiskit/qiskit-ibm.json`, no `IBM_QUANTUM_TOKEN`), so nothing below the `Account`
column has been confirmed by anything except local artifacts. Whoever holds the account
runs `python IBM_Cloud/verify_hardware_jobs.py --write-map` and records the instance/CRN
line. The 16 IonQ rows are not IBM jobs and are skipped by design -- they need the IonQ
console if provenance for them is wanted. The job-ID enumeration and the `--write-map`
rewrite were both exercised offline (`--list`, and a dry run against a copy of the map).

---

## 2026-08-31

### Manuscript folders consolidated: `overleaf_docs/` is now the only copy
`latex_outputs/paper_latex/` (IEEEtran `paper_hvk.tex`, the Aug-20 `paper_hvk_sn.tex`
Springer snapshot, an Aug-13 supplement, a cover letter still addressed to IEEE TQE, and
a 73-file figure archive) and `submission_bundle/` (the abandoned TQE bundle, IEEE
sources) were deleted; both remain in git history. Every duplicate was older than its
`overleaf_docs/` counterpart, so there is now one source per document.

Live references retargeted rather than left dangling:
- `.github/workflows/pages.yml` stages `overleaf_docs/assets/paper_hvk_springer.pdf`
  (the Pages site would otherwise fail on a missing file).
- The nine `IBM_Cloud/plot_*`/`generate_ansatz_figures.py` scripts now write figures to
  `overleaf_docs/assets/figures/`.
- `tests/test_submission_claim_audit.py` reads the Springer manuscript for both its
  headline-string guard (now `R^2=0.9735`, the Springer rounding, instead of the IEEE
  file's `0.974`) and its no-transition-claims scan; `submission_claim_audit.json`
  updated to match. 19 tests pass.
- `README.md`, `REPRODUCE.md`, `latex_outputs/README.md`, `project_artifacts/results.md`,
  `Main_new/README.md`, `latex_outputs/compile_tex.py` docstring.

`author_contributions.pdf` and `literature_review.pdf` had no build under
`overleaf_docs/`; both were compiled into `assets/` so every source here has its PDF.
`latex_outputs/` now holds only `compile_tex.py`, `images_latex/` and its README.

## 2026-08-27

### Physics review pass (both manuscripts + .bib) — 6 fixes applied, 3 sent to student
Supervisor high-effort read. Applied directly (text/reference only, no numbers change):
- **A1 (supplement):** renamed the "global order parameter" $M_z(t)$ to "global
  magnetization summary" and added a sentence stating we avoid "order parameter"
  since there is no symmetry breaking/transition. Fixes a term-of-art contradiction
  with the paper's own no-transition proof.
- **A2 (supplement):** added that $S$ (MPS bond entropy) is constant in $t$, so
  $R_{ES}(t)$ tracks the learned energy $H(t)$ up to a fixed scale — pre-empts
  "why divide by a constant?".
- **A3 (main, entanglement-necessity):** added the explicit mechanism — a linear
  readout of $\{\langle O_i\rangle\}$ cannot form $\langle O_i\rangle\langle O_j\rangle$,
  while the entangling measurement supplies $\langle O_iO_j\rangle$ directly; scoped
  the claim to the linear protocol (a nonlinear classical readout could recover it).
- **A4 (main, §dequant):** added a clause distinguishing the circuit's entangling
  power (strongly-entangling ansatz, genuinely entangled) from whether the measured
  pair-observable channel carries task-relevant info (a property of the target).
- **A5 (main, §regularizer_inert):** promoted from supplement — the learned couplings
  enter only the energy term and never reach the decoder, so the Hamiltonian is a
  read-only observable, not an inductive bias; that is the precise sense of "diagnostic."
- **B1 (.bib):** corrected Chakraborty2018 author names to the verified
  Sanjay Chakraborty, Sudhindu Bikash Mandal, Soharab Hossain Shaikh (were garbled).
  Verified against the Springer record; bbl now renders "Mandal SB, Shaikh SH".
Both recompiled: main 24 pp, supplement 26 pp, 0 undefined citations.
Assigned to student as todo.md block G (need code/data/citation access): **G1** Fei2021
likely-miscitation (→ Ran et al. PRResearch 2, 033293, 2020?), **G2** topology 11.7 dB
vs random floor, **G3** bond-dim non-monotonicity (single-seed → confirm or scope).

### Main paper: Tier-1 quality enhancements (`paper_hvk_springer.tex`)
Senior-author critical-read pass; three fixes:
1. **Data-availability contradiction fixed.** The Statements block still said the
   audit covers "the held-out comparison the tested HVK map **loses**" --- a
   deficit-register phrase that survived the framing pass. Now: "the
   resource-matched held-out comparison, on which the tested HVK map is competitive
   with rather than superior to the classical controls." Matches the body register.
2. **Author-contributions formatting fixed.** Was a broken `description` list with an
   empty `\item[Author contributions]` label followed by dangling name items. Now one
   clean CRediT paragraph (both authors), + "Both authors reviewed and approved."
3. **Formal problem statement added to Sec. 2.** New `\paragraph{Problem statement.}`
   with an explicit objective: patches $x_p$, MPS features, the Pauli-correlator map
   $\Phi(x_p)\in\mathbb{R}^d$ ($d=27$/$19$), decoder $D_\theta$ with Fourier encoding
   $\pi(p)$, and the reconstruction loss $\mathcal{L}(\theta)=\sum_p\|x_p-\hat{x}_p\|^2
   + \lambda E(\Phi(x_p))$ (Eq. objective). Notation $\Phi$ is consistent with the D4
   section. Raises technical polish; the loss was previously only implicit.
Recompiled: 24 pp, 0 undefined citations (the added problem-statement paragraph +
display equation restored the page the Discussion trim had reclaimed). (Considered
but deferred to the student's pass: SSIM precision normalization, negative-$R^2$
footnote.)

### Assigned to student: re-prepare results-code map for the Springer manuscript
Added task block F to `overleaf_docs/todo.md`:
- **F1** — produce `overleaf_docs/RESULTS_MAP.md` re-targeting every row from the old
  `paper_hvk.tex` §IV/§V labels to the current Springer section/table/figure IDs.
  (Do not overwrite `TODO/results-core-map.md`.)
- **F2** — real-hardware cross-check against the IBM Quantum account (only the student
  has access): confirm each job ID exists, backend, shots, PSNR, and record the
  instance/CRN + job IDs in the map. Supervisor verified all headline NUMBERS already
  survive unchanged in the Springer paper (no drift), so this is re-mapping +
  HW-provenance, not a re-run.
- **F3** — close R10 (`contrastive+no-energy`, 33.33 dB, no artifact).
- Reference punch-list items A1/A2/B1/B2 in that file marked DONE (verified by
  supervisor; no bib edits needed).

### Cover letter re-targeted to QMI (`cover_letter.tex`)
- Addressee "Editor-in-Chief, IEEE TQE" -> "Editor-in-Chief, *Quantum Machine
  Intelligence* (Springer)"; "Article in IEEE TQE" -> "research article in QMI".
- Scope paragraph rewritten from "fits TQE's scope (quantum-circuit design,
  hybrid learning, hardware-validated evaluation)" to QMI's scope (quantum machine
  learning, hybrid classical-quantum architectures, tensor-network methods for
  structured data, hardware-validated).
- Register fix: "still beating the two weakest controls by a wide margin" ->
  "substantially outperforms the two null controls" (same rule as the manuscripts).
- Closing: "IEEE's author-posting policy ... traditional (non-open-access) route"
  -> "Springer Nature's self-archiving policy ... standard subscription route".
- Compiles: 2 pp. STILL TO FILL (supervisor): suggested reviewers (line ~50),
  Siddhartha Patra ORCID (line ~68).

### Reference punch-list (student's `todo.md`) — all verified, no bib edits needed
- **West et al. 2024 erratum:** `sn-bibliography.bib` already has both `West2024`
  (PRX Quantum 5, 030320) and `West2024Erratum` (PRX Quantum 6, 020902, 2025, DOI
  10.1103/g4zk-79l9); both are cited in the paper. Resolved.
- **Krizhevsky 2009 co-author:** single-author `{Krizhevsky, Alex}` is correct —
  Hinton is advisor, not a listed author on the 2009 CIFAR tech report. No change.
- **Xu2021 pages:** `13569--13578` matches the official CVF open-access proceedings
  and is used consistently in both docs. No change.
- **Wolberg1993:** present in the supplement's own `thebibliography` with correct
  UCI attribution + DOI 10.24432/C5DW2B. No change. (It is NOT in the `.bib` because
  the supplement uses a self-contained bibliography, not `sn-bibliography.bib`.)

### Supplement — positive-framing sweep (`supplementary_study.tex`)
- **Table caption (was ~line 128):** "does not beat local/raw controls, although it
  remains far above random-VQC" -> "is competitive with local/raw controls (TOST
  equivalence, +/-1 dB) and substantially outperforms the random-VQC null control."
  Removes the banned verb "beat" and the deficit-first register.
- **Sec. Held-Out (was ~line 157):** "HVK2D exceeds both by a wide, practically
  meaningful margin" -> "HVK2D substantially outperforms both." Drops the
  null-control brag phrasing, consistent with the main paper.
- **LEFT as-is (correct register):** the meta-discussion of "beats"/"loses to" as
  terminology (quoted), and "ties ... rather than beating it" describing the
  internal energy-loss ablation (not an HVK-vs-classical claim).

### Main paper — Discussion trim (`paper_hvk_springer.tex`)
- Merged `\subsection{HVK as a representative instance of a broader design pattern}`
  (~40 lines) into `\subsection{Task-dependent representational scope}` as one
  appended paragraph; re-homed `\label{sec:representativeness}` (cross-referenced in
  Secs. 1 and 6) onto the merged subsection. Removed triple repetition + hedging.
- Demoted `\subsection{Architectural differentiation matrix}` (a 6-line pointer) to
  a lead-in paragraph on the table; unreferenced `\label{sec:differentiation}`
  removed. Discussion 8 -> 6 subsections. PDF 24 -> 23 pp.

### Main paper — positive-framing pass (`paper_hvk_springer.tex`)
- **Abstract:** led the held-out result with the positive TOST equivalence finding
  ("statistically competitive with resource-matched controls") instead of "does not
  exceed classical controls (18.12 vs 18.80)"; exact numbers moved to the body (Sec.
  5.3). Still 242 words.
- **Contribution 3:** "beating the two weakest controls by a wide margin" ->
  "substantially outperforms the two null controls (strict classical random features
  and a random-VQC)."
- **Sec. 5.3:** dropped "by a wide, practically meaningful margin" brag.
- Register rule (locked): *competitive* => resource-matched baselines (TOST only);
  *outperforms* => null controls only; never "beats." Body honesty (-0.68 dB, CI,
  Wilcoxon p) unchanged.

### Folder reorganization
- Created `assets/`; moved figures to `assets/figures/` and compiled outputs to
  `assets/`. Added `\graphicspath{{assets/}}` to both manuscript preambles.
