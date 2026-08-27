# TODO — Round 2: Project Completion → Submission

**Thesis (locked):** HVK is a *new hybrid quantum–classical image-reconstruction
architecture* that **works**, is **competitive** with resource-matched classical
baselines (TOST equivalence at ±1 dB — *not* "beats"), runs on **real hardware**, and
adds **capabilities** classical maps don't (entanglement-sensitive channel, interpretable
Hamiltonian diagnostics). **No quantum advantage claimed.**

**Venue:** Quantum Machine Intelligence (Springer, journal 42484). Subscription route
= free to publish (no APC; OA optional ~EUR 2,890 but not required) + arXiv preprint
= free to read. NOTE: IEEE TQE is NO LONGER the target — since Jan 2024 it is gold-OA
with a mandatory ~USD 1,995 APC, so there is no free route there. Manuscript is
formatted for QMI (`overleaf_docs/paper_hvk_springer.tex`, sn-jnl class).

**No new simulations.** Science is locked (`results-core-map.md` — all rows backed except
R10's one disclosed row). Everything below is *cut / reframe / verify / package.*

**Order:** §1 → §2 → §3 → §4 → §5 → §6. §1 gates the rest. Report **done / not done / blocked** per item.

---

## §1 — Remove all transition/change-point DETECTION *(DECIDED)*
No sharp transition exists anywhere — only smooth drift + fluctuations. "Detection" is a
within-run median+2σ threshold artifact. **Delete every change-point / critical-epoch /
"critical temperature" claim, table, figure, threshold.** Keep only: (i) the on/off
negative-control table (the honest disproof), and (ii) one clean $R_{ES}$ trace reframed
as a plain interpretability readout.

**Main paper (`paper_hvk.tex`):**
> **Senior-author standing review (2026-08-13):** the main text is otherwise APPROVED — do
> NOT rewrite abstract, contributions, results, discussion, or conclusion beyond the specific
> edits below. It is internally consistent and correctly scoped. Touch only what's listed.
- [ ] **1a-abstract** *(MANDATORY, corresponding-author decision)* — **Remove the
  phase-transition/change-point material from the ABSTRACT entirely** (~lines 24–25: the two
  sentences beginning "A companion supplementary study additionally reports an exploratory
  training-dynamics change-point diagnostic..." through "...whether or not the Hamiltonian term
  is present."). The abstract must not mention the change-point diagnostic, its negative
  control, $M_z$, or $R_{ES}$ at all. Keep the existing "interpretable Hamiltonian energy
  diagnostic" phrase already present elsewhere in the abstract — that one clause is the only
  Hamiltonian-diagnostic mention the abstract carries.
- [ ] **1a-intro** — In the Introduction, cut/trim the standalone paragraph that re-summarizes
  the change-point diagnostic (~line 52, "The companion supplementary document additionally
  reports an exploratory training-dynamics change-point diagnostic..."). Reduce to at most one
  sentence, or fold into the existing supplement-pointer sentence.
- [ ] **1a** — Tighten §5.3 (`sec:phase_transition`, ~line 283) to ~3 sentences: tracked
  $M_z$/$R_{ES}$; no sharp transition, only drift+fluctuations; on/off control fires on/off
  alike → **no** transition claim. Drop the size/bond-dim sentence.
- [ ] **1b** — Merge/cut the redundant Discussion paragraph (~line 309) and the two
  phase-transition bullets in §Limitations (~lines 355–356) down to one bullet. No dangling `\ref`.

**Supplement (`supplementary_study.tex`, bulk ~lines 478–697):**
- [ ] **1c** — KEEP `tab:phase_transition_corrected` (16/24) + `tab:phase_transition_onoff`
  (4/4 vs 4/4). Tighten surrounding prose.
- [ ] **1d** — KEEP one figure: `critical_temperature_traces.pdf`, recaptioned as
  interpretability readout (energy/entropy decreasing smoothly; *not* temperature, *not*
  transition). Rename subsection off "critical temperature." Strip Eq. `critical_epoch` +
  detected/t_c language.
- [ ] **1e** — DELETE `critical_temperature_susceptibility.pdf` + its "detected @ t=" text.
- [ ] **1f** — DELETE size/bond-dim apparatus: `sec:finite_size_phase_transition`,
  `sec:bond_dimension_phase_transition` + figs (`finite_size_phase_transition.pdf`,
  `finite_size_mean_susceptibility.pdf`, `bond_dimension_phase_transition.pdf`) + tables
  (`tab:finite_size_phase_transition`, `tab:bond_dimension_phase_transition`) → one sentence.
- [ ] **1g** — DELETE hardware/IonQ replays of the detection diagnostic:
  `sec:checkpoint_hardware_finite_size`, `sec:ionq_simulator`,
  `sec:checkpoint_hardware_energy_entanglement` + their figs
  (`checkpoint_hardware_order_parameter.pdf`, `checkpoint_ionq_order_parameter.pdf`,
  `checkpoint_hardware_energy_entanglement_ratio.pdf`,
  `checkpoint_ionq_energy_entanglement_ratio.pdf`, `contrastive_order_parameter_curve.pdf`,
  bond-dim replay figs ~936/963). Real hardware story = the reconstruction pilot (stays).
- [ ] **1h** — Fix all dangling `\ref`/`\label`/`figures/*.pdf`; `git rm` unused figures.
  2-pass compile of BOTH docs clean, zero undefined refs.

## §2 — Shorten the supplement HARD (this is a priority, not cleanup)
`supplementary_study.tex` is ~1049 lines / 30 tables / 23 figures — too bloated to review.
**Hard target: cut to ≤600 lines, ≤15 tables, ≤10 figures.** §1 already removes a large
chunk (transition detection); this section finishes the job on everything else.
- [ ] **2a** — Collapse the shuffle-observables note to ONE paragraph: 0.301 vs ~16 dB
  unresolved, stated plainly; don't re-derive at length.
- [ ] **2b** — Merge redundant ablation tables — **one table per distinct finding**, no
  near-duplicate variants of the same sweep.
- [ ] **2c** — Cut every figure not referenced by a surviving claim; cut long derivations
  that a citation or one equation can replace.
- [ ] **2d** — Merge tiny subsections; a subsection that carries one number becomes a sentence.
- [ ] **2e** — Every retained item must back a *main-paper* claim or a *stated honest scope
  limit*. If it does neither, cut it.
- [ ] **2f** — Report the before/after counts (lines, tables, figures) so the cut is verifiable.
- [ ] **2g** — After cuts, re-run the number cross-check (see §4c).

## §3 — Reframe remaining diagnostics to match the thesis
- [ ] **3a** — **D4 symmetry:** present as "exactly equivariant *by construction* (error
  ~1e-16)" — a design-correctness check, not a quantum capability. Remove any "provable
  symmetry" / advantage framing.
- [ ] **3b** — **Entanglement necessity** (R²=0.9735 vs ≤0.02 controls): keep as the lead
  "capability classical maps lack," on the *constructed nonlocal* target; state plainly it
  does not transfer to natural-image reconstruction (that's the honest scope).
- [ ] **3c** — **$R_{ES}$ trace** (from §1d): position as the interpretable-Hamiltonian
  capability. One clean figure, honest caption.
- [ ] **3d** — Confirm "competitive" is always TOST-backed (±1 dB), never "beats"; confirm
  held-out CIFAR classical-wins result stays visible, not buried.

## §4 — Consistency & hygiene
- [ ] **4a** — Author line: drop "Dr." prefixes; names + affiliations final.
- [ ] **4b** — Grep both docs for stray "advantage" / "significant" not backed by a test.
- [ ] **4c** — Full number cross-check main ↔ supp (held-out 18.80/18.12, leakage
  R²=0.9735, D4 9.57e-17, hardware 25.90–31.52 dB, TOST −0.68 dB / ±1 dB). Zero mismatches.
- [ ] **4d** — Clean 2-pass compile of BOTH docs; proofread rendered pages around every
  §1–§3 edit. Zero errors, zero undefined refs.
- [ ] **4e** — Fold the fixed `literature_review.tex` references into the paper's intro/
  discussion where cited (lit-review is *not* submitted standalone).

## §5 — Reproducibility & repo release
- [ ] **5a** — `REPRODUCE.md` + `results-core-map.md`: prune rows for deleted §1 results;
  confirm every surviving table/figure maps to a script + artifact.
- [ ] **5b** — Update `submission_claim_audit.json`: remove disproven transition claims,
  keep verified ones.
- [ ] **5c** — CI green on final branch (`ruff` clean, tests pass).
- [ ] **5d** — *(Decision needed)* `.git` is 1.8 GB (duplicated binaries across `main2/`
  + `Main2/` + history). Decide: leave as-is, or history-rewrite (BFG/filter-repo) before
  the public release. Destructive — needs sign-off.
- [ ] **5e** — *(Supervisor)* Mint Zenodo DOI from a tagged release; add DOI to `CITATION.cff`.

## §6 — Submission packet
- [ ] **6a** — `cover_letter.tex` (in `overleaf_docs/`): re-target to Quantum Machine
  Intelligence (currently may still address TQE), note arXiv plan, refresh the
  headline results (now transition-free).
- [ ] **6b** — `author_contributions.md`: authors confirm CRediT split (draft exists).
- [ ] **6c** — *(Supervisor)* ORCID iDs for both authors → fill `<!-- FILL -->`.
- [ ] **6d** — *(Supervisor)* 2–3 suggested reviewers (active, appropriate, non-conflicted).
- [ ] **6e** — *(Supervisor)* Confirm QMI article type + length limit + subscription
  (free-to-publish) route. Abstract must be 150-250 words (currently 242, OK).
- [ ] **6f** — Assemble final PDF(s) + source + REPRODUCE + cover letter into the submission bundle.
- [ ] **6g** — Post preprint to arXiv (now, or on acceptance — group's call).

## Guardrails — do NOT undo
- [x] `cifar_nonlocal_advantage` stays OUT (label leak); no "quantum advantage."
- [x] Transition detection: **cut** (§1), not re-inflated. Don't re-add it as a claim.
- [x] Held-out CIFAR: classical wins (18.80 vs 18.12) — keep honest, don't hide.
- [x] "Competitive" = TOST at ±1 dB only, never "beats."
