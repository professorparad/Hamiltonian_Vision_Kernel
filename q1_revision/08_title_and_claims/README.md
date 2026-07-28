# Title and claims audit

## Reviewer ask

Shorten the title. Suggested: "Hamiltonian Vision Kernel: Symmetry-Pooled Quantum
Correlator Features with Hardware Validation." Current title presumably includes "with
Provable Symmetry" language that overstates the D4 result's current system-level
integration (see `../03_d4_end_to_end_integration/`).

## Dependency

This item should be decided **last**, alongside `../01_narrative_reframe/`, once the
experimental items resolve — in particular:
- If `../03_d4_end_to_end_integration/` ships a real end-to-end-integrated D4 result,
  "Provable Symmetry" (or similar) may be defensible again in the title after all.
- If it doesn't ship, the reviewer's suggested title (dropping the symmetry claim to
  "Symmetry-Pooled... Features" — a feature-level claim, not a system-level one) is the
  safer framing.

## Scope

Text-only change to `latex_outputs/paper_latex/paper_hvk.tex` (title, and any
abstract/intro sentences that restate the title's claims) plus a matching audit of
claim language throughout the paper (search for "provable", "with symmetry",
"physics-informed" as a performance claim vs a diagnostic description — ties into
`../04_hamiltonian_objective/`).

## Status

Not started — intentionally last in sequence.
