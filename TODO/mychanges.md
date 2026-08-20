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

**Status:** manuscript content now clean of the "beats" wrinkle. Needs a 2-pass
recompile of `paper_hvk.tex` + refresh of `submission_bundle/pdf/paper_hvk.pdf`
before the bundle is final (student's compile step).
