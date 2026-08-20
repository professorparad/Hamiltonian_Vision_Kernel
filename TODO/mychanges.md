# My Changes — supervisor edits (hand to student)

Running log of every change the supervisor makes directly from now until this is
handed over. Each entry: what, where, why. Newest at top.

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
