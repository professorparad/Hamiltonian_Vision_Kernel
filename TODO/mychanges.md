# My Changes — supervisor edits (hand to student)

Running log of every change the supervisor makes directly from now until this is
handed over. Each entry: what, where, why. Newest at top.

---

## 2026-08-27 — Main paper: Discussion trim (merge two overlapping subsections)

**Why:** the Discussion had 8 subsections; two of them argued the same point twice
in a slightly anxious/over-defensive register that undercut the confident positive
framing. No results changed — this is prose tightening only.

**Files touched:** `overleaf_docs/paper_hvk_springer.tex` (Discussion, Sec. 6).

1. **Merged `\subsection{HVK as a representative instance of a broader design
   pattern}` (~40 lines) INTO `\subsection{Task-dependent representational
   scope}`** as a single appended paragraph (~15 lines). The kept idea: HVK's
   three ingredients each recur individually in the literature and its
   instantiation sits within normal parameter ranges, so the held-out tie is not a
   hyperparameter/strawman artifact. Dropped the triple-repetition of "sits within
   parameter ranges / not a strawman" and the "we think, more useful" hedging.
   - **Label preserved:** `\label{sec:representativeness}` is referenced at lines
     ~149 and ~960, so it was re-homed onto the merged subsection (now carries both
     `\label{sec:dequant}` and `\label{sec:representativeness}`). Compile confirms
     0 undefined refs, no multiply-defined warning.

2. **Demoted `\subsection{Architectural differentiation matrix}`** (a 6-line
   pointer to Table~\ref{tab:differentiation}) to a plain "Finally, Table X
   contrasts…" lead-in paragraph on the table. `\label{sec:differentiation}` had
   no inbound refs, so removed. Net: Discussion 8 -> 6 subsections.

**Result:** paper recompiled, `overleaf_docs/assets/paper_hvk_springer.pdf` now
**23 pp** (was 24), 0 undefined citations. Supplement was NOT further shortened —
it already meets the round-2 target (604 lines / 15 tables / 8 figures, from
~1049/30/23); cutting more would remove load-bearing rigor.

---

## 2026-08-27 — `overleaf_docs/` layout: sources at root, everything else in `assets/`

**Change:** in `overleaf_docs/`, only source files stay at the root
(`*.tex`, `sn-bibliography.bib`, `sn-jnl.cls`, `sn-basic.bst`, `todo.md`).
Figures and all compiled/aux outputs now live under **`overleaf_docs/assets/`**:
`assets/figures/*.pdf` and `assets/paper_hvk_springer.{pdf,aux,bbl,blg,log,...}`.

**How the figures still resolve:** added `\graphicspath{{assets/}}` to the
preamble of BOTH `paper_hvk_springer.tex` and `supplementary_study.tex`. The
`\includegraphics{figures/...}` calls are UNCHANGED — LaTeX prepends the graphics
path, so `figures/x.pdf` resolves to `assets/figures/x.pdf`. Do not rewrite the
include paths.

**BUILD RULE (important — do not "fix" this):** compile from the `overleaf_docs/`
**root**, not with `-outdir=assets`. bibtex must run where `sn-basic.bst` and
`sn-bibliography.bib` live (the root); forcing `-outdir=assets` makes bibtex fail
to find them → 122 undefined citations. Correct workflow:

    cd overleaf_docs
    python ../latex_outputs/compile_tex.py paper_hvk_springer.tex   # builds in root
    # then move outputs into assets/:
    for e in pdf aux bbl blg fdb_latexmk fls log out; do mv -f paper_hvk_springer.$e assets/ 2>/dev/null; done

Verified: paper compiles 24 pp, **0 undefined citations**, figures found via the
graphics path. (On Overleaf this layout also works as-is — Overleaf searches
subfolders and honors `\graphicspath`.)

---

## 2026-08-27 — Springer version: positive-framing pass on the held-out result

**Context:** your `overleaf_docs/paper_hvk_springer.tex` (QMI / sn-jnl, sn-basic,
real `.bib`) is a clean job and is now the canonical main manuscript — 24 pp,
0 undefined citations. I copied the 7 figures into `overleaf_docs/figures/` so it
compiles locally (Overleaf already had them). Only edits below are content.

**Rule enforced (the locked framing):** we present HVK by what it *is and does*,
never by what it fails to do. The held-out CIFAR result is a **positive
equivalence finding (TOST, ±1 dB)**, not a "classical wins / does not exceed"
concession. Register discipline: *competitive* ⇒ resource-matched baselines (TOST
only); *outperforms / substantially outperforms* ⇒ the two **null controls**
(random-VQC, classical-random-features) only; **never "beats," never "by a wide
margin."** Exact held-out numbers (18.12/18.80, −0.68 dB) live in the **body/
supplement**, not the abstract.

**Files touched:** `overleaf_docs/paper_hvk_springer.tex` (3 edits).

1. **Abstract (was ~line 49).** Removed the deficit-first opener "the tested HVK2D
   map does not exceed … classical controls (18.12 against 18.80 decibels) …
   rather than merely indistinguishable." Now leads positive: "a pre-declared
   TOST equivalence procedure … establishes that HVK2D is statistically
   *competitive* with resource-matched … controls, and it substantially
   outperforms random-circuit and classical-random-feature controls." The exact
   numbers were dropped from the abstract (they stay in §5.3). Abstract = 242
   words, still inside QMI's 150–250.

2. **Contribution 3 (was ~line 164).** "while still **beating** the two weakest
   controls **by a wide margin**" → "and substantially **outperforms** the two
   null controls (strict classical random features and a random-VQC)." Kills the
   banned verb; keeps the honest six-of-eight TOST statement intact.

3. **§5.3 ablation-relationship (was ~line 402).** "HVK2D outperforms them **by a
   wide, practically meaningful margin**" → "HVK2D **substantially outperforms
   them**." Same null-control brag removed; the −0.68 dB, CI, Wilcoxon p, and the
   "not shown to outperform … competitive via TOST" body sentences are LEFT as-is
   (honesty-as-scope belongs in the body).

**Deliberately LEFT as-is (correct register, do not touch):** §5.3 body
"the tested HVK map is informative but is not shown to outperform the simpler
maps" (honest, immediately pivots to the positive TOST finding); "single-image
pilot cannot distinguish …" (scope); "cannot be recovered/audited" (provenance);
the nonlocal "raw or local features cannot represent" (that's the *positive*
argument for the entangling channel). "We make no claim of quantum advantage"
stays — it's a disciplined boundary on the claim, not a deficit.

**PDF regenerated:** `overleaf_docs/paper_hvk_springer.pdf` (24 pp, 0 undefined
citations) via `latex_outputs/compile_tex.py`.

**Status:** main manuscript now clean of every deficit-first / "beats" wrinkle in
the Springer version. Not committed — supervisor review pending.

---

## 2026-08-20 — Abstract & §held-out: verb discipline on "beats"

**Rule enforced:** never say HVK "beats" anything. Use *competitive* (TOST, ±1 dB)
for resource-matched **baselines**; use *outperforms/exceeds* only for **null
controls** (random-VQC, strict-classical-random-features). This is a locked
honesty guardrail — "beats … by a wide margin" reads as an unquantified brag and
sits awkwardly next to "does not exceed classical controls."

**Files touched:** `latex_outputs/paper_latex/paper_hvk.tex` (2 edits).

1. **Abstract (line 25).**
   - was: `while HVK2D still beats a random-VQC and a strict-classical-random-feature control by a wide margin.`
   - now: `while HVK2D substantially outperforms a random-VQC and a strict-classical-random-feature control.`
   - why: drop "beats"; drop unquantified "by a wide margin" in the abstract.

2. **§ held-out discussion (line 136).**
   - was: `because HVK2D beats them by a wide, practically meaningful margin.`
   - now: `because HVK2D outperforms them by a wide, practically meaningful margin.`
   - why: drop "beats"; kept "practically meaningful margin" — it *is* backed by
     the supplement's numbers (Δ=+2.27 / +6.97 dB).

**Swept but deliberately LEFT as-is (correct register, do not touch):**
- `supplementary_study.tex:127` (caption) — "does **not beat** local/raw controls":
  honest disclaimer, fine.
- `supplementary_study.tex:156` — "HVK2D **exceeds** both by a wide … margin":
  already the right verb, backed by numbers.
- `literature_review.tex:227` — Cotler "match or **beat** it": about a cited
  dequantization theorem, not about HVK.

**Status:** manuscript content now clean of the "beats" wrinkle.

**PDFs regenerated (supervisor did this):** recompiled both docs and refreshed the
bundle — `latex_outputs/paper_latex/{paper_hvk,supplementary_study}.pdf` and the
copies in `submission_bundle/pdf/`. Verified via `pdftotext` that the paper PDF now
renders "substantially outperforms a random-VQC …" (no "beats"). Both compiles were
clean (2 passes, zero undefined refs).

---

## 2026-08-20 — New: `latex_outputs/compile_tex.py`

Added a small module so PDF regeneration is one call, not a manual fight with
MiKTeX:

    from compile_tex import compile_tex
    compile_tex("paper_latex/paper_hvk.tex")   # -> Path to the .pdf

It encapsulates the three MiKTeX traps we hit today: (1) auto-install prompts hang
under `nonstopmode`, (2) the filename DB is stale right after `mpm --install` (needs
`initexmf --update-fndb`), (3) `latexmk` caches a failed run and refuses to rebuild.
On a missing `.sty` it installs the providing package, refreshes the fndb, and
retries once. **Rule for the student: after ANY LaTeX edit, run this to regenerate
the PDF and re-copy into `submission_bundle/pdf/` before the bundle is called final.**
