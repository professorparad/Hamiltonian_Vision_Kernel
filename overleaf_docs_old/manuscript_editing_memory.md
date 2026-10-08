# Manuscript editing memory

Working record of the section-by-section language pass on `main_paper.tex` and
`supplementary.tex`. Scope is **language only**: code, results and physics
are settled and are not under review here. The purpose of this file is so the
pass can be resumed in a fresh session without re-deriving the decisions.

Last updated: 2026-09-22, after the Abstract.

`main_paper.tex`: **the language pass is complete.** Every section has been
through it: Abstract, Introduction, 2, 3, 4, 5 and 6. The paper went from 26
pages to 24 over this session.

"Complete" means the language pass only. The main paper is **not** submission
ready: the colour markup and `\snew{}` markers must still be stripped, and the
four red-bold blocks plus `todo.md` block I still need the student's answers.
Those answers should come **before** the markup is stripped, or the questions
disappear unresolved. See Section 4 below.

`supplementary.tex`: language pass **not started**. Twelve reviewer blocks have
been inserted for the student; no wording changed. Section 6 of this file has
the full survey and the agreed batch order.

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

- **Both `.tex` files are LF, not CRLF.** This entry previously said CRLF and
  that was wrong; verified 2026-09-22 with a byte count on both files
  (`main_paper.tex` 1045 LF / 0 CRLF, `supplementary.tex` 604 LF / 0 CRLF).
  Write with `newline="\n"`, or the inserted lines become mixed endings.
- Bash heredocs eat backslashes and apostrophes and will corrupt LaTeX. Inline
  `python -c "..."` is just as bad: it mangled `\color` into a regex error three
  times in one session. The reliable method is to write a Python script with the
  Write tool, build the old and new blocks by `"\n".join([...])` over a list of
  `r"..."` raw strings, assert `d.count(old) == 1`, and `sys.exit` with a loud
  failure otherwise. Never half-apply.
- When appending a block to the **start** of a paragraph, do not `lstrip()` the
  joined string. It eats the blank line that separates the paragraph from a
  preceding `\end{table}` and silently merges the two.
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
| Abstract | 43-62 | **DONE** (done last, reconciled to the finished body) |
| 1 Introduction | 76-197 | **DONE** |
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
| 5.1 Why regulariser was inert | 829-861 | **DONE** |
| 5.2 Task-dependent scope | 862-871 | **DONE** |
| 5.3 Topology alignment | 872-940 | **DONE** |
| 5.4 Validated scope | 941-953 | **DONE** |
| 6 Conclusion | 954-end | **DONE** |

The **whole of `main_paper.tex`** has now been swept and is clean: zero prose
em-dashes, zero colon-then-clause constructions, zero US spellings outside
LaTeX identifiers. The only `---` left in the file are the four preamble
banners and the single table cell at line 538 meaning "not applicable", all
of which rule 3 permits.

Line numbers drift as edits land. Re-derive them rather than trusting this
table.

Commits so far: `98581ca` (sections 3, 4.1-4.3, and this file),
`a76a59f` "checked sections 2, 3, 4" (the 4.4 through 4.9 work),
`54088d4` (rename of `paper_hvk_springer.tex` to `main_paper.tex`, plus the
reference updates in the workflow, README, tests and companion docs).
The Section 5 and 6 work is **not yet committed**.

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

### Section 5.1, why the regulariser was inert

- Was one 31-line paragraph carrying five ideas. Now two paragraphs, the finding
  and mechanism, then the read-only reading. 340 words down to about 140.
- Removed the paired dash around the contrastive variant, and four
  colon/semicolon-then-clause constructions.
- The six-line provenance narrative about the superseded draft numbers is now one
  clause. The table caption already carries the pointer to the full account.
- Removed the double negative "This does not mean Hamiltonian regularisation is
  never useful". The conditional sentence carries that content instead.
- Dropped the hedge "if anything" from the lead, which was softening a 5.93 dB
  effect the table states plainly.
- The root-cause fix and the seed-sensitive decoder-feature variant were cut from
  the main paper. Both remain in the Supplementary Material.
- The closing `sec:phase_transition` paragraph ended on a prohibition, "must not
  be read as thermodynamic transitions". It now states the positive scope.

### Section 5.2, task-dependent representational scope

- Was two breathless paragraphs, 431 words, the longest subsection in the
  Discussion. Now three paragraphs, about 210 words.
- Removed three paired dashes and four colon-then-clause constructions.
- **The whole rebuttal framing of paragraph 2 was cut.** It had existed to answer
  an accusation nobody made, naming "an uncharitable alternative explanation" and
  defending the work as "not a strawman". Rule 4 forbids disclaiming motivations
  nobody has claimed. The substantive point survives in positive form: the three
  ingredients are standard in the literature, the instantiation is typical, so the
  measured behaviour is evidence about the design pattern.
- Removed "We stress that this is not a statement about...", throat-clearing plus
  a negation lead.
- All six citations, both `\ref` targets and the closing scope caveat kept.
- The `\emph{measured pair-observable channel}` emphasis was dropped when the
  sentence was restructured.

### Section 5.3, topology alignment

- The smallest repair in the Discussion. Two paragraphs, 179 words to about 120.
- Removed the paired dash around the distinguishing-features list, and a semicolon
  splice joining the resource-cost reading to the formal-equivalence caveat.
- Dropped "Finally," which was sequencing filler and also misleading, since the
  paragraph introduces the differentiation table rather than concluding topology.
- The three hybrid-vision families were named in full in both the prose and the
  table caption two lines apart. Now named once, in the caption.
- Moved the wall-clock sentence next to the other cost claim, so the paragraph
  ends on its conclusion.
- **Raised, not acted on.** Paragraph 2 and `tab:differentiation` are about
  architectural differentiation, not topology, and sit under a topology heading.
  They would read more naturally at the end of 5.2, which `sec:representativeness`
  already labels. Moving them is structural rather than language, so it was left.

### Section 5.4, validated scope

- **Retitled** from "Validated scope and next steps" to "Validated scope". The
  subsection contained no next steps; all seven items were limitations. Next steps
  live in the Conclusion. The label `sec:limitations` is unchanged.
- Removed six colon/semicolon-then-clause constructions, the densest concentration
  left in the paper at that point.
- Every item had led with what the result is not. As a block of seven that read as
  an apology. Each now states the scope positively, then the caveat.
- `generalization` to `generalisation` twice in prose. The neighbouring
  `\ref{sec:no_generalization}` label was left alone.
- **Item 5 is now the third site** carrying the agreed negative-control wording,
  "the threshold fires whether the Hamiltonian term is present or absent". It
  replaced the double negative "has no demonstrated sensitivity to the Hamiltonian
  term itself". Sections 4.4, 4.9 and 5.4 must now be changed together.

### Section 6, Conclusion

- Was three paragraphs, 653 words, the second-largest prose block in the paper.
  Now three paragraphs, 269 words, a 60 per cent cut.
- **The Conclusion had been re-deriving the Results.** It restated
  $25.90$--$31.52$~dB, $R^2=0.9735$, $R^2\leq0.02$ and the $9.57\times10^{-17}$
  equivariance error, and re-argued the TOST tie in full. Every one of those
  numbers is established in Sections 4.3, 4.6, 4.7 and 4.8. They are replaced by
  `\ref` pointers. A conclusion asserts, it does not re-prove.
- Paragraph 2 had stated its capability list twice, four sentences apart, and then
  closed by restating its own opening clause. Said once now.
- Removed the defensive opening "If HVK ties classical baselines...", the
  rebuttal "None of these is in tension with the tie-with-classical result", and
  "We consider this carefully measured boundary a constructive contribution",
  which asked the reader to credit the work rather than stating a finding.
- Removed three paired dashes, including one in the back matter's Code
  availability statement, and the double negative "does not natively expose".
- Kept word for word: the TOST framing, "competitive with rather than superior",
  "We make no claim of quantum advantage", and the four future-work items.
- **Option A was chosen** over folding future work into 5.4. The Discussion ends
  on scope and the Conclusion ends on outlook, which is what a referee expects.

### Whole-file, this pass

- The paper went from 26 pages to **25**. The "25 pages" figure in Section 5 of
  this file had been stale since before the Section 4 work, and is now correct
  again by coincidence rather than by having been maintained.
- A `Tensor-tnetwork` typo at line 90 was introduced during the file rename, fixed,
  then reappeared when the file was regenerated, and was fixed a second time. Worth
  re-grepping after any regeneration of the manuscript from outside this loop.

### Section 1, Introduction

- Was 921 words, the largest prose block in the paper. Now about 620.
- Removed the last two prose em-dashes in the manuscript, at lines 132 and 153.
- The opening paragraph was a 26-line citation dump: six sentences carrying 29
  citations across VQE, quantum image processing, tensor networks, particle
  physics, annealing tomography, emission tomography, ghost imaging, hybrid SVMs,
  QCNNs and capsule networks. Now about 11 lines. **All 29 citations are kept**,
  grouped by theme rather than narrated one at a time. Each key was verified
  individually after the edit.
- The three-properties paragraph was one sentence chained by a colon and two
  semicolons. Now four sentences.
- "in the spirit of" appeared twice in that one sentence. Both removed, rule 3.
- `formalizes` to `formalises`.
- The Novelty subsection had three parallel semicolon splices, one per architecture
  family (`;HVK never`, `;HVK instead`, `;HVK uses`). Split into five paragraphs,
  one per family plus a closing claim.
- **Duplication with Section 5.2 resolved in favour of 5.2.** The Introduction had
  argued at length that no family combines the four ingredients and that the
  contribution is the assembly. Section 5.2 now makes that argument. The
  Introduction keeps the claim in one sentence and points to
  `sec:representativeness`. If this is ever inverted, invert it in both places.
- Contribution 5's colon construction split.
- The closing paragraph justified putting material in a companion document so the
  contributions and "the generalization question do not obscure each other". That
  framing is gone; it now simply states what the supplement reports.
  `generalization` to `generalisation`, with `\label{sec:no_generalization}` left
  untouched as rule 1 requires.
- **`\snew{five of seven distinct}` in Contribution 3 was deliberately not touched.**
  Stripping the marker is mechanical, but the number inside it is live and is the
  subject of `todo.md` block I. It needs the student's answer, not an editor's.

### Abstract

- Done last, so it could be reconciled against the finished body. All five numbers
  were cross-checked against their source sections before editing: 25.8 and 41.6
  against Section 4.1, 25.90 and 31.52 against Sections 4.4 and 4.6.
- 216 words down to 183. QMI allows 150-250, so there was no length pressure; the
  cut was for density.
- Removed the double negative "The capabilities that a purely classical
  reconstruction pipeline does not natively provide are...", which became "HVK
  adds...". This was the **third and last** site of that construction; the same
  phrase was removed from the Conclusion and the Introduction earlier in the pass.
- Split the semicolon splice after "chain and grid interaction graphs".
- Cut the null-control clause, the TOST expansion, "the CIFAR-10 image dataset" to
  "CIFAR-10", and the decoding mechanism. The equivalence result is the headline;
  beating random baselines is the weaker claim and is in Section 4.3.
- Dropped the announced count "three capabilities", rule 3.
- The word-count comment on line 43 said 230 when the true count was 216. It is now
  set to 183 and should be re-derived if the abstract changes again.

### An inconsistency found during the Abstract pass, not resolved

The Abstract and the Introduction describe the **same** TOST comparison in two
different ways, and both sit inside `\snew{}` markers:

- Abstract: competitive with "a resource-matched raw/local linear classical
  control", singular.
- Introduction, Contribution 3: equivalent to "five of seven distinct"
  resource-matched classical/ablated controls.

These need to agree before submission. The question is the same one as `todo.md`
block I, which asks whether the true count of distinct controls is five of seven
or six of eight. Raised for the student, not acted on, because it is a question of
fact about the experiments rather than of language.

---

## 4. Open items

**Must be removed before arXiv or journal submission**

- `main_paper.tex`: the `xcolor` package and **21 colour occurrences**: 6 `red`,
  5 `black`, 4 `blue`, 4 `green`, 2 `studentgreen`.
- `supplementary.tex`: the `xcolor` package, its three-line banner comment, and
  the **12 `[SUPERVISOR: ...]` blocks** listed in Section 6. Added 2026-09-22.
  Strip with a regex on `\{\\color\{red\}\\textbf\{\[SUPERVISOR:.*?\]\}\}`.
- The `\snew{}` revision markers in the main paper, **5 occurrences**.
- Note that the colour command is a *switch*, not a scope. The `\color{green}`
  opened at line 461 in Section 4.3 does not close until line 607 at the end of
  4.4, so both subsections render green in full and 4.5 then opens blue. This
  is cosmetic, was accepted deliberately, and disappears when the markup is
  stripped. The braced form `{\color{red}\textbf{...}}` is the safe pattern and
  is what all six red blocks use.

**Red-bold blocks awaiting the student, sixteen in total**

In `main_paper.tex`, four:

1. Section 4.3, the withdrawn nine-variant table.
2. Section 4.4, the closing of the last paragraph.
3. Section 4.8, the reframed "favourable by construction" disclaimer.
4. Section 4.8, the unnamed raw-linear classical control.

In `supplementary.tex`, twelve. See the table in Section 6 for the line numbers
and the issue at each site.

**Deferred to a single final sweep**

- Remaining `-ization` and `-ized` forms in sections not yet reviewed.
- `nonoverlapping` versus `non-overlapping`. The file currently uses both.
- **Prose em-dashes in `main_paper.tex`: none left.** All ten are gone. The
  last two, in the Introduction, went with the Introduction pass. To confirm,
  `grep -n '\-\-\-'` and expect only the preamble banners and the table cell
  at line 538. Note the count in this file was twice carried forward stale.
  Re-derive it rather than trusting any number written here.

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
- Both `.tex` files are **LF**. See Section 0; the earlier CRLF claim was wrong.
  Edit scripts must match exact text and must fail loudly rather than
  half-applying.
- Do not commit unless asked.
- Build artifacts (`.aux`, `.log`, `.out`, `.toc`) are removed after each
  compile. Compiled PDFs belong in `assets/`, per the existing build rule.
- `supplementary.tex` is **untracked in git** (`??`), a rename of the deleted
  `supplementary_study.tex` that was never staged. So `git checkout --` cannot
  restore it. Copy the file to the scratchpad before any scripted edit.

---

## 6. Supplementary material

`supplementary.tex`, 604 lines, 11 sections, 22 tables, 10 figures. Compiles at
**28 pages**, 0 errors, 0 undefined.

### Language status: NOT started

The only pass it has had is the early mechanical em-dash sweep (14 trailing-dash
fixes), which rule 3 later overruled. Violation density is roughly what Section 4
of the main paper looked like before its pass. Counted 2026-09-22:

- **19 prose em-dash violations** out of 21 occurrences. Line 599 is a
  bibliography title (`MedMNIST v2---A large-scale...`) and line 453 is a numeric
  range (`8--13`); both stay. Many of the rest are paired parentheticals of the
  kind the supervisor overruled, including the document's opening sentence and
  five figure captions.
- **~25 colon-then-loaded-clause constructions** out of 44 `: [a-z]` matches;
  the remainder are `\scriptsize`, list intros and captions.
- **~55 prose Indian-English substitutions**: `optimization` family (~18),
  `generalization` (~7), `memorization` (6), `regulariser` family (7), plus
  `normalization`, `standardization`, `canonicalized`, `magnetization`,
  `dequantization`, `visualize`, `summarized`, `parameterizes`, `reorganization`,
  `serialized`, `initialized`, `characterize`, `favors`, `center`.
- **Zero `-iz` forms inside `\label{}`, `\ref{}`, `\path{}` or `\texttt{}`.**
  Verified by grep. The rule-1 hazard does not apply to this file. But three
  *section titles* carry `-iz` in visible prose while their labels do not, so the
  titles change and the labels must not:
  `\subsection{Adaptation Beyond Single-Image Optimization}`,
  `\subsection{Dataset-Level Generalization...}` (label **is**
  `sec:dataset_level_generalization`), and
  `\subsection{The Physics-Informed Regularizer's Conditional Value}`.
- Rule 4 is the worst category here, worse than in the main paper. The file
  repeatedly asks the reader to credit the authors' honesty.

### Proposed batch order, agreed but not yet run

A: Indian English sweep (prose + three titles, labels untouched).
B: Sections 1-2, lines 22-40. C: 3.1-3.3, lines 43-158. D: 3.4-3.7, lines
177-283 excluding 283. E: 3.8-3.9, lines 288-375. F: Sections 4-6, lines
376-453. G: Sections 7-11, lines 455-end.

### Structural note

The prose is written as **very long single physical lines**. Lines 283, 319, 375,
401, 445, 453 and 532 are each one paragraph of 200-400 words on one line; line
283 alone is ~370 words. One line is one whole argument, so re-derived line
numbers are coarse and `sed -n` on a range is not a useful preview.

### Reviewer annotation pass, 2026-09-22 (DONE)

At supervisor request, **12 `{\color{red}\textbf{[SUPERVISOR: ...]}}` blocks**
were inserted for the student. **No language was changed.** `xcolor` was not
loaded in this file and was added, under a removable banner comment. Verified:
stripping the 12 blocks and the banner returns the file to the original byte for
byte apart from the single space each block adds; all 1304 numbers preserved; all
labels, refs, paths and citations identical.

Current block sites, by line:

| Line | Issue |
|---|---|
| 28 | "a rigorous complement", the phrase already cut from main 4.3 |
| 42 | whether main Section 3 should state seeds/budget/optimiser |
| 158 | **count disagreement**, "five of seven" here vs "six of eight" in main |
| 264 | "does not mean ... never useful", the double negative cut from main 5.1 |
| 287 | provenance narrative, **held back** pending the main 4.3 red block |
| 323 | fourth negative-control site |
| 348 | same, in the `tab:phase_transition_onoff` caption |
| 373 | "does not natively expose", the double negative cut from the Conclusion |
| 382 | "we report this honestly" |
| 386 | "we report this budget explicitly rather than silently shrinking scope" |
| 405 | heading "and we say so" |
| 449 | Std. error vs standard deviation, carried from main 4.7 |

### Three cross-document agreements this pass established

1. **The negative-control wording is now a FOUR-site invariant, not three.**
   Main paper 4.4, 4.9, 5.4 carry "the threshold fires whether the Hamiltonian
   term is present or absent". Supplement line 323 and the line 348 caption state
   the same fact in different words. Change all four together.
2. Supplement line 264 currently **contradicts** the agreed main-paper 5.1
   wording, and line 373 contradicts the agreed Conclusion wording.
3. Supplement line 158 says "five of the seven distinct controls", the corrected
   count from block I of `todo.md`. The main paper still says "six of the eight".
   The two documents disagree on a number. Flagged, not changed.
