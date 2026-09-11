# Student punch-list — QMI/Springer submission bundle

Only open items live here. Completed work is recorded in `EDITS.md` and git history.
(Block H — the live IBM check and clean build — is done: H1 closed as a documented
negative result, the 25-job re-execution recorded in `RESULTS_MAP.md` §F4, H2 build
clean. See `EDITS.md`.)

---

## I. QUESTION FROM SUPERVISOR (assigned 2026-09-11)

The original hardware job IDs no longer resolve (§F2); the §F4 re-execution jobs do.
You chose to keep §F4 as a separate ledger rather than update the paper's tables, and
wrote that the new numbers "must not be swapped into the manuscript tables." Before we
freeze the manuscript, walk me through that decision:

  I1. Given the original job IDs are permanently unrecoverable, what is the provenance
      story for the paper's *printed* hardware numbers if a referee asks to verify them?
      Is "reproducible from the retained `.npy` artifacts" enough in your judgment, or is
      it a weakness we should address?

  I2. If we adopted the §F4 numbers as the paper's results, the §4.6 anchor analysis
      would have to be rewritten — several of its claims reverse under the new values:
        - "reproduces almost exactly" (Monalisa marrakesh vs fez): 25.94/25.90 becomes
          23.28/26.53 — a 3.25 dB gap, no longer "almost exactly";
        - "2.59 dB below the fez pilot" (cat kingston 256sh): 28.93 becomes 33.67, now
          *above* the pilot — the paragraph's logic inverts;
        - the 256→1024-shot "recovers to within 0.28 dB" story reverses (33.67 → 33.04,
          a slight drop with more shots).
      Did you weigh this, and is it the main reason you left the tables alone?

  I3. Your recommendation, as the person who ran both campaigns: (A) keep the printed
      numbers and cite §F4 only as provenance evidence, or (B) move fully to the §F4
      numbers and rewrite the §4.6 analysis to match? Give the reason.

Do NOT edit the manuscript for this yet — this is a decision to settle first.

---

## NOT your items — supervisor (Siddhartha) will handle

These are `[FILL]`s in the cover letter — judgment calls, do NOT fill them:
  - Suggested reviewers (2–3 names) — `cover_letter.tex` line ~50.
  - Siddhartha Patra's ORCID — `cover_letter.tex` line ~68 (Sparsho's is filled).
  - arXiv preprint timing (post now vs. on acceptance) and the Zenodo DOI.
