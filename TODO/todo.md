# HVK — Remaining Tasks Before Submission

## Thesis the paper defends
> HVK is a **new hybrid quantum–classical image-reconstruction architecture** that works, is
> **competitive** with resource-matched classical baselines, runs on **real hardware**, and adds
> **capabilities** (entanglement-sensitive channel, interpretable Hamiltonian diagnostics) classical
> maps don't. **No quantum advantage claimed.**

The reframe is done (novelty paragraph, TOST, thermo demoted, retitle). Below is what remains.
Keep producing **no new simulations** — this is writing/cleanup only.

---

## 1. Resolve the shuffle contradiction — BLOCKS SUBMISSION
- **Problem:** the supplement's shuffle-observables ablation reports **0.301 dB and ~16 dB** for the
  same thing and calls it "not resolved." A reviewer reads this as "the paper can't reproduce its own
  result" — the single most reject-triggering item.
- **Do:** **delete the shuffle subsection** (it is not load-bearing under this thesis). Replace with at
  most a one-line footnote noting the ablation was withdrawn pending a reproducible protocol. Do not
  ship two numbers 50× apart.

## 2. Delete the duplicate / merged .tex — one source per document
- **Problem:** `hvk_combined_report.tex` (+ `.pdf`) is a **text-copied merge** of all three real docs
  (no `\input`). Every edit to `paper_hvk.tex` silently makes it stale — that's how contradictory
  numbers appear.
- **Do:** **delete `hvk_combined_report.tex` and `hvk_combined_report.pdf`.** Keep exactly three
  sources: `paper_hvk.tex`, `supplementary_study.tex`, `literature_review.tex`. If a combined PDF is
  ever needed, generate it on demand or via `\input` — never commit a copied merge.
- **Rule:** `.tex` is the source of truth; `.pdf` files are builds — recompile, never hand-edit.

## 3. Soften "provable D4 symmetry" to match the honest body
- **Problem:** intro/contributions still call it "**provable $D_4$ symmetry**," implying a headline
  quantum result. The body correctly shows it's a **Reynolds group-averaging construction** that a
  purely classical map achieves identically (supplement) and that isn't in the trainable pipeline.
- **Do:** reword intro + Contribution 5 to "**an exact, numerically-verified $D_4$-equivariant
  observable-pooling construction**" (drop "provable symmetry" as a selling point). State once, plainly,
  that the guarantee comes from group averaging, not from anything quantum. Leave the body as is.

## 4. Consistency & hygiene
- **4a. Author line:** title block reads "Dr. Siddhartha Patra" — IEEE omits titles; use
  "Siddhartha Patra."
- **4b. Cross-check numbers** agree between `paper_hvk.tex` and `supplementary_study.tex` after tasks
  1–3 (held-out 18.12/18.80, Table X 38.63/44.56, hardware, TOST).
- **4c. Literature-review bibliography** (only if it ships with the submission): fix the remaining
  verified entries — West erratum (PRX Quantum 6, 020902, 2025), Larocca (Nat. Rev. Phys. 7, 174–189),
  QCAE authors + Phys. Rev. A 109, 032623 (2024), the ~10 author-less entries; audit all author lists.
- **4d. Final build:** recompile all three PDFs clean, proofread the rendered pages.

## 5. Submission packet (supervisor + student)
- Confirm venue (**IEEE TQE** or **Quantum Machine Intelligence**), APC/waiver, length.
- Cover letter, author-contribution statement, ORCIDs, 2–3 suggested reviewers (drafts exist).

---

## Order
**1 → 2 → 3 → 4 → 5.** Tasks 1 and 2 gate a clean, self-consistent draft; do them first.

## Owners
- **Student:** 1, 2, 4b, 4c, 4d.
- **Supervisor:** 3 (claim wording), 4a, 5 (venue + packet).
