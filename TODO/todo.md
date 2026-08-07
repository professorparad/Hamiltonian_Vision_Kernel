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

## 5. Shorten the supplementary Material
- **Problem:** the supplement is **1049 lines, 30 tables, 23 figures** — a dumping ground, not a
  companion. It reads as "broad but not deep" (the exact reviewer criticism), and most of the bulk is
  the training-dynamics ("phase-transition") material — which the main paper already concludes is
  unreliable, because its own check shows the detector fires whether the Hamiltonian is switched on or
  off, so it isn't actually measuring the physics it claims to. A supplement should
  *support the paper's claims*, not run a second paper.
- **Keep (load-bearing — these back main-text claims):** held-out + multi-dataset "competitive" result;
  TOST; leakage audit + resource-matching + the design-guidelines box; dataset-level generalization
  (real trained model); hardware-pilot methodology + job IDs; exact circuit/Hamiltonian definitions;
  D4 output-level experiment; reproducibility/artifact map.
- **Cut or compress hard (bloat / self-disowned / redundant):**
  - Shuffle reproducibility note → **cut** (same as Task 1).
  - Thermodynamic apparatus (finite-size, χ bond-dimension sweep, $R_{ES}$ simulator+hardware+IonQ
    replays, checkpoint bond-dimension replays): collapse to **one** compact subsection — the
    change-point diagnostic, its negative control, and a **single** sensitivity figure. Move all the
    per-shot IBM/IonQ replay figures and job tables to the machine-readable archive; cite job IDs in
    one compact table, don't reproduce every plot.
  - Energy-as-decoder-feature follow-up (2/3 seeds, disowned) → **one sentence**, not a table.
  - Any Hamiltonian-controls / number table duplicated between main and supplement → keep **once**.
- **Target:** aim to roughly halve it (≈500 lines, ≤15 tables, ≤10 figures). Every retained table/figure
  must map to a claim the reader needs; if it doesn't, it goes to the archive.

## 6. Submission packet (supervisor + student)
- Confirm venue (**IEEE TQE** or **Quantum Machine Intelligence**), APC/waiver, length.
- Cover letter, author-contribution statement, ORCIDs, 2–3 suggested reviewers (drafts exist).

---

## Order
**1 → 2 → 3 → 4 → 5.** Tasks 1 and 2 gate a clean, self-consistent draft; do them first.

## Owners
- **Student:** 1, 2, 3, 4a, 4b, 4c, 4d.
- **Supervisor:** 5
