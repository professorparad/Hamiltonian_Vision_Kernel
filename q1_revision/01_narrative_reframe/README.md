# Narrative reframe

## Reviewer ask

Reframe the central claim as "a carefully controlled boundary study of quantum-correlator
representations" rather than a superior reconstruction architecture. Currently the paper
presents HVK as a vision architecture even though the strongest held-out evidence shows
simpler classical features perform at least as well.

## Status

Not started. Deliberately sequenced **after** the experimental items (especially
`../02_dataset_level_generalization/` and `../06_classical_baselines/`), since the
narrative should follow from final results, not be locked in before they land — the
dataset-level study in particular may change exactly how the "loses to simpler controls"
finding should be worded (a bigger, fairer generalization test could sharpen or soften
that claim).

## What to touch when this starts

- `latex_outputs/paper_latex/paper_hvk.tex`: abstract, introduction, and the closing
  paragraph of the Discussion section (currently ends with a defensive but still
  architecture-forward framing — reviewer wants the boundary-study framing to be the
  headline, not a caveat at the end).
- Cross-check against `../08_title_and_claims/` — title and abstract framing should be
  decided together, not independently.

## Open question for project owner

How much appetite is there to lead with "this is a boundary/negative-result study" in the
abstract itself (reviewer's literal suggestion), vs. keeping the current structure
(architecture framing up front, honest limitations throughout) and only sharpening the
closing framing? These read very differently to a first-pass reviewer skimming the
abstract, and it's a judgment call about the paper's identity, not a technical decision.
