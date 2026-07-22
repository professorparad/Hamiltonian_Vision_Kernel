# Paper And Figure Outputs

This folder contains LaTeX drafts, exported figures, and paper-oriented assets
from the HVK project.

It is not part of the runtime package. Treat it as the project writing area:

- `paper_latex/`: LaTeX paper draft and compiled PDF artifacts.
- `images_latex/`: figures copied or exported for paper use.

## Practical Notes

- Regenerated plots may overwrite files here.
- Keep large experimental dumps out of this folder unless they are needed for
  the paper.
- The source code that produces most experiment artifacts lives in `Main/`,
  `Main2/`, and `Baselines/`.

If a figure in the paper looks stale, rerun the corresponding experiment and
copy only the final figure here.
