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

**Current state:** `paper_hvk_springer.pdf` = 23 pp, 0 undefined citations,
abstract 242 words (QMI wants 150-250). `supplementary_study.pdf` = 26 pp, 0
undefined citations, self-contained `thebibliography` (does not use the `.bib`).

---

## 2026-08-27

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
