# Manuscript editing memory

Working record of the section-by-section language pass on
`main_paper.tex`. Scope is **language only**: code, results and physics
are settled and are not under review here. The purpose of this file is so the
pass can be resumed in a fresh session without re-deriving the decisions.

Last updated: 2026-09-22, after Section 4.9. **Sections 2, 3 and all of 4 are
done.** Section 5 is next.

---

## 0. How to resume in a new chat

1. Read this file first.
2. Re-derive line numbers. They drift after every edit:
   `grep -n 'section{\|subsection{' overleaf_docs/main_paper.tex`
3. The working loop, one subsection at a time, is:
   supervisor says `read` -> report the issues found against the style rules in
   Section 1 below, propose a rewrite, and stop -> supervisor says `apply` ->
   apply, compile, verify numbers preserved, report.
4. **Do not commit unless asked.** The supervisor reviews diffs manually.
5. Anything where the meaning changed rather than only the wording goes into
   the text wrapped in `{\color{red}\textbf{...}}` for the student to confirm.
   Do not silently resolve a question of substance.

Editing mechanics that matter:

- The `.tex` files are **CRLF**. Bash heredocs eat backslashes and apostrophes
  and will corrupt LaTeX. The reliable method is to write a Python script with
  the Write tool, build the old and new blocks by `"\r\n".join([...])` over a
  list of `r"..."` raw strings, assert `d.count(old) == 1`, and `sys.exit` with
  a loud failure otherwise. Never half-apply.
- Compile with `pdflatex` twice, then confirm 0 errors and 0 undefined.
  Remove `.aux`, `.log`, `.out`, `.toc` afterwards.

---

## 1. Agreed writing style (applies to every future edit)

1. **Indian English.** `generalisation`, `optimisation`, `regulariser`,
   `neighbour`, `behaviour`, `factorisation`, `normalised`, `centre`,
   `greyscale`, `magnetisation`, `modelled`, `recognisable`.
   **Never** rewrite `-ize` inside LaTeX identifiers: `\label{}`, `\ref{}`,
   `\citep{}`, `\path{}`, file names. A bulk regex once silently renamed
   `sec:regularizer_inert` in five places. It was caught and reverted, but the
   hazard is real. Always re-grep labels after a spelling pass. Note that
   `\label{sec:no_generalization}` and `\label{sec:regularizer_inert}` are
   correct as they stand and must not be "fixed".

2. **Plain style. Simple sentences, commas and full stops.** One fact per
   sentence. Break long multi-clause sentences instead of re-punctuating them.
   The construction specifically rejected is a colon followed by a loaded
   clause.

   - BAD: "...does not transfer to a different image: on Monalisa, zero-shot
     transfer reaches only 8.31 dB, while extending training recovers 28.63 dB."
   - GOOD: "...does not transfer to a different image. On Monalisa, zero-shot
     transfer reaches 8.31 dB. Extending training to include the second image
     recovers 28.63 dB."

   A colon in a **section title or a table caption** is fine. The rule is about
   prose.

3. **No AI writing signatures.**
   - No trailing em-dash appending an afterthought to a complete sentence.
   - **No paired em-dashes either** (`text --- aside --- text`). This was
     initially missed and the supervisor caught it. The rule is: **no `---` in
     prose at all.** The only permitted uses are numeric ranges (`$2$--$8$~dB`),
     compound words (`feature--position`), table cells meaning "not
     applicable", and comment banners in the preamble.
     - BAD: "Full methodology --- exact gate sequences, job IDs, and validation
       results --- is documented in the supplement."
     - GOOD: "The supplement gives the full methodology, covering exact gate
       sequences, job IDs, and validation results."
   - No evenly weighted exhaustive lists where a clause plus citations would do.
   - No throat-clearing sentence announcing what the next sentence will do, and
     no announcing a count before delivering it ("for two reasons", "three of
     these choices warrant...").
   - No double negatives. State the fact instead. "The threshold fires equally
     with the term present and absent" beats "the control does not establish
     that the diagnostic is sensitive to the term".
   - Avoid "in the spirit of", "it is worth noting", "crucial", "comprehensive",
     "leverage", "robust", "settles the reading" as filler.

4. **Positive framing, not defensive.** State what a result establishes, then
   scope it. Keep the limitation, but move it out of the lead. Do not disclaim
   motivations nobody has claimed. This has been the single most frequent
   correction across the whole pass.

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
| 4.7 Exact D4 observable pooling | 703-752 | **DONE** |
| 4.8 Entanglement necessity | 753-815 | **DONE** |
| 4.9 Training-dynamics diagnostic | 816-834 | **DONE** |
| 5.1 Why regulariser was inert | 838-898 | **next** |
| 5.2 Task-dependent scope | 899-946 | pending |
| 5.3 Topology alignment | 947-1030 | pending |
| 5.4 Validated scope, next steps | 1031-1072 | pending |
| 6 Conclusion | 1073-end | pending |

Section 4 was swept as a whole after 4.9 landed and is clean: zero prose
em-dashes, zero colon-then-clause constructions, zero US spellings outside
LaTeX identifiers, across all 426 lines.

Line numbers drift as edits land. Re-derive them rather than trusting this
table.

Commits so far: `98581ca` (sections 3, 4.1-4.3, and this file),
`a76a59f` "checked sections 2, 3, 4" (the 4.4 through 4.9 work).

---

## 3. What was changed, and why

### Whole file

- **Em-dash cleanup (earlier pass).** 21 trailing-dash constructions rewritten
  across both manuscripts: 7 in the main paper and 14 in the supplement. That
  pass touched only "complete sentence, dash, appended gloss" and deliberately
  kept paired parenthetical dashes, which was later overruled. See rule 3.
- **Indian English sweep.** 42 substitutions. A few `-ization` forms may survive
  in sections not yet reached. A final sweep will catch them.

### Section 2, Architecture

- Deleted the throat-clearing sentence "Three of these design choices warrant
  explicit motivation rather than being left implicit in the pipeline above."
- Split the single 30-line "Why X" paragraph into three `\paragraph{}` blocks,
  matching the existing `\paragraph{Problem statement.}` style.
- Rewrote the grid paragraph closing, which had ended on a negation, "not
  because a chain is separately motivated for 2D image data". It now states the
  positive role of the chain as the ungridded control.
- Replaced the interleaved 1D/2D preprocessing prose with a configuration
  table, `tab:configurations`. The old text alternated between the two
  configurations six times and relied on "respective" to bind values to
  topologies. Arithmetic verified: 24+11+11=46, 12+5+5=22, 6+6+15=27, 6+6+7=19.
- Split four breathless sentences.

### Section 3, Experimental Setup

- Rewrote "the exploratory training-dynamics material is addressed only as a
  provenance audit and is not retained as evidence". That framing contradicted
  Section 4.9 of this same paper, titled "A scoped training-dynamics
  diagnostic". It now reads as an interpretability diagnostic whose scope is set
  by the negative control given there. The limitation is unchanged, it is simply
  no longer the lead.
- Fixed a grammar error. Hardware replay is not a kind of per-image fitting
  measurement, so it is now its own sentence.

### Section 4.1

- The closing sentence led with what the result was not. It now states what the
  result establishes.
- The forward pointer to Section 4.3 was duplicated here and in 4.2. It was kept
  in 4.2, which is the subsection actually about transfer.

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

- Split one 30-line paragraph into four: the question, the numbers, TOST, and
  the ablation rerun.
- Removed three colon and semicolon constructions, including one sentence that
  carried four facts via colon, then "while", then semicolon.
- "a separate, equally important question" became "a separate question".
- "We report this as a rigorous complement" became "as a complement". Rigour is
  for the reader to judge.
- **Red bold for the student.** The phrase "replaces an earlier table whose
  per-seed provenance could not be recovered" was queried by the supervisor, on
  the ground that a manuscript should speak about current data. It now reads
  that the table was withdrawn because the repository does not retain the
  per-seed records needed to reproduce it. The withdrawal itself is legitimate
  to state, because the old nine-variant table did circulate publicly via the
  GitHub Pages deploy and the presentation slides, so this is a public-record
  correction rather than drafting history.

### Section 4.4, hardware reconstruction pilot

- The opening apologised for the preceding sections and praised the group for
  having been honest ("a limitation we flagged rather than left implicit"). It
  now simply states that the results above are simulator results.
- Split a 130-word sentence, the longest in the paper, into seven.
- `recognizable` to `recognisable`, twice.
- Split a semicolon splice joining the Monalisa and CIFAR results.
- Removed a paired-dash construction around the methodology list. This is the
  one the supervisor caught, which established rule 3.
- The last paragraph was reviewed twice. The supervisor first asked whether it
  was needed at all, then decided to keep and rewrite it. It now ends on the
  section's actual claim, "This is the sense in which HVK runs on real quantum
  hardware", instead of trailing into a methodology pointer.
- Its closing double negative, "does not establish that the diagnostic is
  sensitive to the Hamiltonian term", became the plain statement that the
  threshold fires equally with the term present and absent. **Red bold for the
  student.** Note that `IonQBackends` is cited nowhere else in the main paper,
  so cutting this paragraph would remove that entry from the main bibliography.
  That is correct behaviour, not a reason to keep the paragraph.

### Section 4.5, noise/shot trade-off

- Removed four colon-then-loaded-clause constructions.
- Removed a paired-dash construction listing device effects.
- The same clause appeared twice with the same verb, "what the calibration
  snapshot does not capture ... device effects a single calibration snapshot
  does not fully capture". Now stated once.
- Split into two paragraphs, the two shot-budget findings and then the
  residual-gap analysis.
- `modeled` to `modelled`, `unmodeled` to `unmodelled`.

### Section 4.6, real-hardware anchor points

- Paragraph 1 was a single 60-word sentence with a colon, a paired dash and two
  parentheticals, and had no main verb after the colon. It is now five plain
  sentences.
- Removed the colon construction in paragraph 2 and split its 45-word "while"
  clause.
- Two consecutive sentences both opened "This is". The second now reads "It
  provides direct evidence". **That repetition was introduced by the earlier
  mechanical em-dash cleanup**, which is worth remembering as a hazard of bulk
  fixes: always read through after a mechanical pass.
- "explicitly quota-tracked" to "quota-tracked".

### Section 4.7, D4-equivariant pooling

- The subsection was one paragraph whose body was a single 95-word sentence
  carrying five facts via a colon, a "while", and a trailing em-dash.
- Removed the trailing dash and the colon construction.
- Removed the double negative "is not measurably more symmetric than a
  local/raw control". The two error values are both of order one, so the text
  now says that directly.
- "This is currently a ... rather than a ..." led with what the construction is
  not. Flipped so it leads with what it is.
- Now six plain sentences. All four numbers, the 7000 transform count and all
  three cross-references verified preserved.

### Section 4.8, entanglement necessity

- Was one paragraph of three sentences, the first two running 70 and 90 words.
  Now three paragraphs: the setup, the numbers, the mechanism.
- Removed two colon-then-loaded-clause constructions.
- Split the 90-word results sentence, which carried six models through a colon,
  a comma chain, a semicolon and a "while", into five sentences.
- Dropped "This is therefore evidence that", so the claim asserts directly.
- The `\emph{representationally necessary within the tested feature class and
  fixed linear-readout protocol}` scope wording and the closing "rather than"
  were kept word for word.
- **Red bold 1 for the student.** The opening disclaimer "deliberately
  constructed to reward pair-observable structure" was reframed as "by
  construction favourable to pair-observable structure, which is why we read it
  as a representational statement and not as a performance comparison". That
  changes the meaning rather than only the wording, so it needs confirming.
- **Red bold 2 for the student.** The `Raw-linear classical` row ($R^2=0.0133$)
  appears in `tab:hvk_pair_diagnostic` but is the only control never named in
  prose. Left unresolved on purpose, because it is the same row involved in the
  duplicate-control question already pending in block I of `todo.md`.

### Section 4.9, training-dynamics diagnostic

- Thirteen lines containing three colon-then-clause constructions and one
  paired dash, the densest concentration in the paper.
- `magnetization` to `magnetisation`.
- Split into two paragraphs, the readouts and then the negative control.
- The original left its conclusion inside a dash clause. The text now states it:
  "The threshold therefore fires whether the Hamiltonian term is present or
  absent, and it does not track that term." **This wording deliberately matches
  the red-bold block in Section 4.4**, so the two places agree. If the student
  changes one, change both.

---

## 4. Open items

**Must be removed before arXiv or journal submission**

- The `xcolor` package and **21 colour occurrences**: 6 `red`, 5 `black`,
  4 `blue`, 4 `green`, 2 `studentgreen`.
- The `\snew{}` revision markers, **5 occurrences**.
- Note that the colour command is a *switch*, not a scope. The `\color{green}`
  opened at line 461 in Section 4.3 does not close until line 607 at the end of
  4.4, so both subsections render green in full and 4.5 then opens blue. This
  is cosmetic, was accepted deliberately, and disappears when the markup is
  stripped. The braced form `{\color{red}\textbf{...}}` is the safe pattern and
  is what all six red blocks use.

**Red-bold blocks awaiting the student, four in total**

1. Section 4.3, the withdrawn nine-variant table.
2. Section 4.4, the closing of the last paragraph.
3. Section 4.8, the reframed "favourable by construction" disclaimer.
4. Section 4.8, the unnamed raw-linear classical control.

**Deferred to a single final sweep**

- Remaining `-ization` and `-ized` forms in sections not yet reviewed.
- `nonoverlapping` versus `non-overlapping`. The file currently uses both.
- **Ten prose em-dashes remain.** An earlier note in this file said four, which
  was a stale count carried forward without rechecking. The verified list, by
  line number as of this update, is:
  - Introduction: 132, 153
  - Section 5.1: 856 (one paired construction spanning 856-857)
  - Section 5.2: 932, 936, 939
  - Section 5.3: 966
  - Conclusion: 1080, 1109, 1152
  Each will be taken with its own section. To re-locate, `grep -n '\-\-\-'` and
  ignore the preamble banners at lines 3, 19, 43, 68 and the table cell at 554.

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
- Section 4.7's `tab:d4_equivariance` header (line 734) says "Std.\ error"
  where the caption and the prose describe a standard deviation over 7000 patch
  transforms. Standard error and standard deviation are different quantities.
  Flagged as a possible content question, not acted on, because it is not a
  language matter.

**Student's own open items, tracked in `todo.md`**

- Block H1, live IBM account check. Block H2, clean build.
- Block I, the duplicate-control question. `raw-linear-classical` and
  `local-observables-only` are the same 26-D feature map zero-padded to 32, yet
  three supplement tables list them separately and the text says "six of the
  eight controls" where the true count of distinct controls is five of seven.
  Asked as a question, not directed as a fix.

---

## 5. Working practice

- Compile after every edit and confirm 0 errors, 0 undefined citations and 0
  undefined references. The paper currently sits at **25 pages**.
- Verify numbers survived every rewrite by extracting them from the edited range
  and comparing against the pre-edit text, not by eye.
- The `.tex` files are **CRLF**. Edit scripts must match exact text including
  line endings, and must fail loudly rather than half-applying.
- Do not commit unless asked.
- Build artifacts (`.aux`, `.log`, `.out`, `.toc`) are removed after each
  compile. Compiled PDFs belong in `assets/`, per the existing build rule.
