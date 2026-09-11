# Manuscript editing memory

Working record of the section-by-section language pass on
`paper_hvk_springer.tex`. Scope is **language only**: code, results and physics
are settled and are not under review here. The purpose of this file is so the
pass can be resumed after a break without re-deriving the decisions.

Last updated: 2026-09-11 (after Section 4.6).

---

## 1. Agreed writing style (applies to every future edit)

1. **Indian English.** `generalisation`, `optimisation`, `regulariser`,
   `neighbour`, `behaviour`, `factorisation`, `normalised`, `centre`,
   `greyscale`.
   **Never** rewrite `-ize` inside LaTeX identifiers: `\label{}`, `\ref{}`,
   `\citep{}`, `\path{}`, file names. A bulk regex once silently renamed
   `sec:regularizer_inert` in five places. It was caught and reverted, but the
   hazard is real. Always re-grep labels after a spelling pass.

2. **Plain style. Simple sentences, commas and full stops.** One fact per
   sentence. Break long multi-clause sentences instead of re-punctuating them.
   The construction specifically rejected is a colon followed by a loaded clause.

   - BAD: "...does not transfer to a different image: on Monalisa, zero-shot
     transfer reaches only 8.31 dB, while extending training recovers 28.63 dB."
   - GOOD: "...does not transfer to a different image. On Monalisa, zero-shot
     transfer reaches 8.31 dB. Extending training to include the second image
     recovers 28.63 dB."

3. **No AI writing signatures.**
   - No trailing em-dash appending an afterthought to a complete sentence.
   - **No paired em-dashes either** (`text --- aside --- text`). This was
     initially missed. The rule is: **no `---` in prose at all.** The only
     permitted uses are numeric ranges (`$2$--$8$~dB`) and compound words
     (`feature--position`). Recast as a plain sentence, or move the aside into a
     clause after a comma.
     - BAD: "Full methodology --- exact gate sequences, job IDs, and validation
       results --- is documented in the supplement."
     - GOOD: "The supplement gives the full methodology, covering exact gate
       sequences, job IDs, and validation results."
   - No evenly weighted exhaustive lists where a clause plus citations would do.
   - No throat-clearing sentence announcing what the next sentence will do, and
     no announcing a count before delivering it ("for two reasons", "three of
     these choices warrant...").
   - No double negatives. State the fact instead. "The threshold fires equally
     with the term present and absent" beats "the control does not establish that
     the diagnostic is sensitive to the term".
   - Avoid "in the spirit of", "it is worth noting", "crucial", "comprehensive",
     "leverage", "robust" as filler.

4. **Positive framing, not defensive.** State what a result establishes, then
   scope it. Keep the limitation, but move it out of the lead. Do not disclaim
   motivations nobody has claimed.

5. **Keep "rather than".** It is the honest-register device of the paper
   ("competitive rather than superior"). Keep existing uses, add no new ones.

---

## 2. Progress

| Section | Lines | Status |
|---|---|---|
| Abstract | 39-62 | deferred to last, reconcile to the finished body |
| 1 Introduction | 76-204 | **deferred to last** (supervisor choice) |
| 2 Architecture | 205-386 | **DONE** |
| 3 Experimental Setup | 387-408 | **DONE** |
| 4.1 Six real image datasets | 413-445 | **DONE** |
| 4.2 Scope: no zero-shot transfer | 446-458 | **DONE** |
| 4.3 Competitive with baselines | 459-498 | **DONE** |
| 4.4 Hardware reconstruction pilot | 499-602 | **DONE** |
| 4.5 Noise/shot trade-off | 603-654 | **DONE** |
| 4.6 Real-hardware anchor points | 655-702 | **DONE** |
| 4.7 Exact D4 observable pooling | 703-753 | next |
| 4.8 Entanglement necessity | 754-808 | pending |
| 4.9 Training-dynamics diagnostic | 809-824 | pending |
| 5.1 Why regulariser was inert | 828-888 | pending |
| 5.2 Task-dependent scope | 889-936 | pending |
| 5.3 Topology alignment | 937-1020 | pending |
| 5.4 Validated scope, next steps | 1021-1062 | pending |
| 6 Conclusion | 1063-end | pending |

Line numbers drift as edits land. Re-derive them by grepping for the section and
subsection commands rather than trusting this table.

---

## 3. What was changed, and why

### Whole file

- **Em-dash cleanup (earlier pass).** 21 trailing-dash constructions rewritten
  across both manuscripts: 7 in the main paper (33 to 26 prose dashes) and 14 in
  the supplement (49 to 35). Only the pattern "complete sentence, dash, appended
  gloss" was touched. Paired parenthetical dashes were kept, being correct usage.
- **Indian English sweep.** 42 substitutions. A few `-ization` forms survive in
  sections not yet reached. A final sweep will catch them.

### Section 2, Architecture

- Deleted the throat-clearing sentence "Three of these design choices warrant
  explicit motivation rather than being left implicit in the pipeline above."
- Split the single 30-line "Why X" paragraph into three `\paragraph{}` blocks,
  matching the existing `\paragraph{Problem statement.}` style.
- Rewrote the grid paragraph closing, which had ended on a negation, "not because
  a chain is separately motivated for 2D image data". It now states the positive
  role of the chain as the ungridded control.
- Replaced the interleaved 1D/2D preprocessing prose with a configuration table,
  `tab:configurations`. The old text alternated between the two configurations
  six times and relied on "respective" to bind values to topologies. Arithmetic
  verified: 24+11+11=46, 12+5+5=22, 6+6+15=27, 6+6+7=19.
- Split four breathless sentences: the section opening, the three-stage pipeline,
  the ansatz figure sentence, and the MLP sentence.

### Section 3, Experimental Setup

- Rewrote "the exploratory training-dynamics material is addressed only as a
  provenance audit and is not retained as evidence". That framing contradicted
  Section 4.9 of this same paper, titled "A scoped training-dynamics diagnostic".
  It now reads as an interpretability diagnostic whose scope is set by the
  negative control given there. The limitation is unchanged, it is simply no
  longer the lead.
- Fixed a grammar error. Hardware replay is not a kind of per-image fitting
  measurement, so it is now its own sentence.

### Section 4.1

- The closing sentence led with what the result was not. It now states what the
  result establishes.
- The forward pointer to Section 4.3 was duplicated here and in 4.2. It was kept
  in 4.2, which is the subsection actually about transfer, and removed from 4.1.

### Section 4.2

- Retitled from "Scope and adaptation across images" to "Scope: models do not
  transfer zero-shot". The old title promised adaptation, whereas the content is
  a transfer failure honestly reported. Safe to rename: the title text appears
  nowhere else, and the label is referenced by name.
- Rewrote the colon construction into three plain sentences. This is the example
  quoted in rule 2 above.
- Deleted "this is how we scope every claim in this paper", a whole-paper claim
  sitting inside a narrow results subsection.
- Dropped "only" before 8.31 dB.

### Section 4.3

- Split one 30-line paragraph into four: the question, the numbers, TOST, and the
  ablation rerun.
- Removed three colon and semicolon constructions, including one sentence that
  carried four facts via colon, then "while", then semicolon.
- "a separate, equally important question" became "a separate question".
- "We report this as a rigorous complement" became "as a complement". Rigour is
  for the reader to judge.
- Every number, CI, p-value, revision marker and cross-reference was diffed
  against HEAD and confirmed preserved.

### Section 4.4, hardware reconstruction pilot

- The opening apologised for the preceding sections and praised the group for
  having been honest ("a limitation we flagged rather than left implicit"). It
  now simply states that the results above are simulator results.
- Split a 130-word sentence, the longest in the paper, into seven. It had carried
  five stacked actions plus a parenthetical plus a dash clause.
- `recognizable` to `recognisable`, twice, in prose and in a figure caption.
- Split a semicolon splice joining the Monalisa and CIFAR results.
- Removed a paired-dash construction around the methodology list.
- Rewrote the closing so the section lands on its actual claim, "This is the
  sense in which HVK runs on real quantum hardware", instead of trailing into a
  methodology pointer.
- The last paragraph ended on a double negative, "does not establish that the
  diagnostic is sensitive to the Hamiltonian term". It now states the fact, that
  the threshold fires equally with the term present and absent, which is also how
  Section 4.9 puts it. **Marked red bold for the student**, who should decide
  whether the paragraph earns its place at all. Note that `IonQBackends` is cited
  nowhere else in the main paper, so cutting the paragraph removes that entry
  from the main bibliography. That is correct behaviour, not a problem, but it
  should be a deliberate choice.

### Section 4.5, noise/shot trade-off

- Removed four colon-then-loaded-clause constructions.
- Removed a paired-dash construction listing device effects.
- The same clause appeared twice with the same verb, "what the calibration
  snapshot does not capture ... device effects a single calibration snapshot does
  not fully capture". Now stated once.
- Split into two paragraphs at the natural break, the two shot-budget findings
  and then the residual-gap analysis.
- `modeled` to `modelled`, `unmodeled` to `unmodelled`, including one instance
  that had drifted into Section 4.6.
- Paragraph 1 was already clean and was left untouched.

### Section 4.6, real-hardware anchor points

- Paragraph 1 was a single 60-word sentence with a colon, a paired dash and two
  parentheticals, and had no main verb after the colon. It is now five plain
  sentences.
- Removed the colon construction in paragraph 2 and split its 45-word "while"
  clause.
- Two consecutive sentences both opened "This is". The second now reads "It
  provides direct evidence". That repetition was introduced by the earlier
  mechanical em-dash cleanup, which is worth remembering as a hazard of bulk
  fixes.
- "explicitly quota-tracked" to "quota-tracked".

---

## 4. Open items

**Must be removed before arXiv or journal submission**

- The `xcolor` package and all colour markup (12 occurrences). Note that the
  colour command is a *switch*, not a scope. The blue switch at the top of
  Section 2 colours everything until the next switch, so more of that section is
  coloured than was actually edited. This was accepted deliberately. The braced
  form is the safe pattern.
- The `\snew{}` revision markers (5 occurrences), present for the student review
  round.
- Colour markup now stands at 19 occurrences, including two red-bold blocks
  flagged for the student: the withdrawn-table sentence in Section 4.3, and the
  closing of the last paragraph of Section 4.4.

**Deferred to a single final sweep**

- Remaining `-ization` and `-ized` forms in sections not yet reviewed.
- `nonoverlapping` versus `non-overlapping`. The file currently uses both.
- **Six paired-dash constructions remain**, down from eight. They sit in
  Sections 4.9, 5.1, 5.2 (two) and 6 (two). Each will be taken with its own
  section. To re-locate them, look for sentences containing two `---`.

**Raised, not yet decided**

- Should the main paper state seeds, step budget and optimiser? At present none
  of these appear in it, so a referee must go to the supplement. One sentence in
  Section 3 would make the paper self-contained on protocol.
- Section 4.3 paragraph 3 still carries two ideas, what TOST establishes and why
  it is reported. They were left joined because the second is short.
- The indented sub-rows in `tab:configurations` restate the 27/19 decomposition
  that the observable-dimension equation already gives as a general rule.
  Dropping them would remove the redundancy and probably return the paper to 24
  pages.

---

## 5. Working practice

- Compile after every edit and confirm 0 errors, 0 undefined citations and 0
  undefined references. The paper currently sits at 25 pages.
- The `.tex` files are **CRLF**. Edit scripts must match exact text including
  line endings, and must fail loudly rather than half-applying.
- Do not commit unless asked.
- Build artifacts (`.aux`, `.log`, `.out`, `.toc`) are removed after each
  compile. Compiled PDFs belong in `assets/`, per the existing build rule.
