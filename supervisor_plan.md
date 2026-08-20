# HVK — Supervisor Plan & Decisions (Round 2)
*Dated 2026-08-13. Supervisor-side record of the decisions behind `TODO/todo.md`
(the student's working checklist). This file lives in the repo root, not `TODO/`.*

## Senior-author verdict (2026-08-13, after full re-read of `paper_hvk.tex`)
- **The main manuscript is essentially DONE and publishable.** Abstract, contributions,
  results, discussion, limitations, and conclusion are internally consistent, every claim is
  correctly scoped, and the honesty (TOST "competitive," "no quantum advantage," the
  self-disowned phase diagnostic) is a **strength** for a TQE reviewer. Do **not** rewrite the
  main text — approved as standing, save the specific trims below.
- **The unfinished work is the SUPPLEMENT** (1049 lines, still carrying the full phase apparatus).
  That, plus the small main-paper trims, is the entire finalization job.
- **Corresponding-author decisions locked:** (1) the phase-transition diagnostic is **removed
  from the abstract entirely** — it appears only in a tightened §5.3 and the supplement;
  (2) **the student executes** all edits (this is their thesis work); I set direction and review.

## Thesis (locked)
HVK is a **new hybrid quantum–classical image-reconstruction architecture** that:
- **works**,
- is **competitive** with resource-matched classical baselines — TOST equivalence at ±1 dB, *not* "beats",
- runs on **real hardware**,
- adds **capabilities** classical maps don't (entanglement-sensitive channel, interpretable Hamiltonian diagnostics).

**No quantum advantage is claimed.**

## Venue decision
- **IEEE Transactions on Quantum Engineering (TQE).**
- **Traditional (non-OA) route → free to publish (no APC).** OA is optional (~$2,045) and not taken.
- **Post the accepted version to arXiv** → also free to read.
- Honest tier: lower-Q1 / boundary. Not top-Q1 (npj QI / PRX Quantum / Quantum), which need an advantage this paper doesn't claim. That's accepted.

## The key round-2 call: kill the "phase transition"
After looking at the actual plots (`critical_temperature_traces.pdf` and `critical_temperature_susceptibility.pdf`):

- **There is no sharp transition anywhere — only smooth drift and fluctuations.**
- The `R_ES(t)` trace is a smooth, monotone descent in all 8 runs → that's the optimizer lowering energy (**convergence**), not a transition.
- The "detection" is a **within-run median+2σ threshold artifact**: only 4/8 runs fire, and the "detected" panels look identical to the "not detected" ones.
- The `M_z` order-parameter detector already failed its **on/off control**: it fires 4/4 with the Hamiltonian ON *and* 4/4 OFF (`Main2/newHVK/results/phase_transition_onoff_control/summary.json`, `fires_only_when_hamiltonian_on: false`). It measures nothing about the Hamiltonian.

**Decision:** delete every change-point / critical-epoch / "critical temperature" claim, table, figure, and threshold — in **both** the main paper and supplement, **including** the hardware/IonQ replays of it.

**Keep only two things:**
1. The **on/off negative-control table** — this is the honest disproof, and honesty is an asset here (it *demonstrates* rigor).
2. **One clean `R_ES` trace figure**, recaptioned as a plain **interpretability readout** ("the learned Heisenberg energy per unit MPS entropy decreases smoothly and consistently across seeds"), with the detection overlay removed and no "transition/temperature" language. This one supports the thesis ("interpretable Hamiltonian diagnostic").

## What the student does (from `TODO/todo.md`, §1–§6)
| § | Task | Owner |
|---|------|-------|
| **1** | Remove all transition **detection**; keep on/off table + one recaptioned `R_ES` trace | Student |
| **2** | **Shorten supplement HARD** → ≤600 lines, ≤15 tables, ≤10 figures; report before/after counts | Student |
| **3** | Reframe D4 (by-construction check), entanglement (lead capability + honest scope), `R_ES` (interpretability), "competitive"=TOST | Student |
| **4** | Hygiene — drop "Dr.", grep stray "advantage", full main↔supp number cross-check, clean compile, fold in lit-review refs | Student |
| **5** | Repo — prune result maps, update audit JSON, CI green | Student |
| **6** | Packet — cover letter (TQE + arXiv), CRediT split | Student |

## What needs YOU / supervisor (flagged, not the student's)
- **§5d** — `.git` is **1.8 GB** (duplicated binaries across `main2/` + `Main2/` + history). Decide: leave as-is, or destructive history-rewrite (BFG / git filter-repo) before public release.
- **§5e** — Mint a **Zenodo DOI** from a tagged release; add it to `CITATION.cff`.
- **§6c** — **ORCID iDs** for both authors.
- **§6d** — 2–3 **suggested reviewers** (active, appropriate, non-conflicted).
- **§6e** — Confirm TQE **article type + length limit** + the free/non-OA route.
- **§6g** — Post the **arXiv** preprint (now, or on acceptance — your call).

## Guardrails (do not undo)
- `cifar_nonlocal_advantage` stays **out** (label leak); no "quantum advantage."
- Transition detection is **cut**, not re-inflated — don't re-add it as a claim.
- Held-out CIFAR: classical wins (18.80 vs 18.12 dB) — keep it honest and visible.
- "Competitive" = TOST at ±1 dB only, never "beats."

## State when this was written
- Branch `main-checking` = `main` = `origin/main-checking` = commit `b175d8b` (all synced).
- No new simulations remain — everything above is cut / reframe / verify / package.
