# Edit log — QMI/Springer submission bundle

Running record of every edit to the manuscripts in this folder, newest first.
Target venue: **Quantum Machine Intelligence** (Springer, journal 42484). Free to
publish via the subscription route (no APC); arXiv preprint for free-to-read.

**Folder layout:** source files (`*.tex`, `sn-bibliography.bib`, `sn-jnl.cls`,
`sn-basic.bst`) at root; figures and all compiled/aux outputs under `assets/`
(`assets/figures/`, `assets/*.pdf`, etc.). Figures resolve via
`\graphicspath{{assets/}}` in both manuscript preambles, so `\includegraphics`
paths stay `figures/...`.

**Build rule:** compile from the folder root (NOT `-outdir=assets`, which breaks
bibtex's search for the `.bst`/`.bib`), then move outputs into `assets/`:

    cd overleaf_docs
    python ../latex_outputs/compile_tex.py paper_hvk_springer.tex
    for e in pdf aux bbl blg fdb_latexmk fls log out; do mv -f paper_hvk_springer.$e assets/; done

**Current state:** `paper_hvk_springer.pdf` = 24 pp, 0 undefined citations,
abstract 242 words (QMI wants 150-250). `supplementary_study.pdf` = 27 pp, 0
undefined citations, self-contained `thebibliography` (does not use the `.bib`).

---

## 2026-09-05

### H1 re-tested with a live instance, then the whole hardware campaign re-executed
**Why this went further than H1 asked.** H1 asked for the ledger's job identifiers to be
confirmed against the service. They cannot be, and now that is established rather than
inferred: an `open` plan Qiskit Runtime instance was created in the active account
(TCG CREST, `059312687e3b4f8484a4d6cd7c311a3d`) and a live IBM Cloud API key issued
against it, which closes the two gaps that made the 2026-09-04 attempt inconclusive --
there is now both a valid key *and* an instance. `IBM_Cloud/verify_hardware_jobs.py`
still reports **all 15 IBM identifiers as `RuntimeJobNotFound`** ("25 jobs, 15 needing
attention"). The accounts API confirms the cause directly: the IBMid
`25ph05023@iitbbs.ac.in` reaches exactly `IIT bhubaneswar [CANCELED]` and
`TCG CREST [ACTIVE]`. No credential can revive those jobs -- a key works only in the
account that created it, a `CANCELED` account cannot issue one, and the submitting
instance no longer exists. §F2 of `RESULTS_MAP.md` records this.

**What was then run, at the supervisor's request, to obtain records that do resolve.**
The full hardware campaign was re-executed under the new instance:
`python IBM_Cloud/run_provenance_campaign.py --stage all` (new driver). **25 of 25 jobs
completed and every identifier retrieves** -- 15 IBM (`ibm_fez`, `ibm_marrakesh`,
`ibm_kingston`; verified `DONE`) and 10 IonQ (`ionq_simulator`). QPU cost: **196 s of the
600 s monthly allowance**, 404 s left. The instance CRN is serialized next to every job
id, which is the one field the 2026-07 campaign failed to record; the new ledger lives in
`IBM_Cloud/outputs/provenance_campaign/` (`ledger.json`, `.md`, `.tex`) and is summarised
in the new §F4 of `RESULTS_MAP.md`.

**Neither manuscript was edited, and the new numbers must not be substituted into the old
tables.** A job identifier and a PSNR are a matched pair. Re-execution on hardware
recalibrated months later returns different values -- Monalisa/`ibm_fez` gives 26.533 dB
where Table 3 prints 25.896 -- so pairing a new identifier with an old PSNR would produce
a table that contradicts itself as soon as anyone retrieves the job, which (unlike
before) they now can. `ledger.tex` therefore holds a *separate* table with each id beside
the value its own job returned and the printed value in its own column.

**One substantive observation for the supervisor.** Across the nine reconstruction jobs
the re-run deltas span **-2.66 dB to +4.74 dB** on identical circuits replayed from
identical checkpoints, seven of nine higher than printed. That is device calibration
drift between campaigns, and it puts a number on a variation the manuscripts currently
leave implicit: the single-point hardware PSNRs in Sections 4.4 and 4.6 carry roughly
+/-2-3 dB of device-and-day spread. Worth one sentence in Section 4.6; it supports the
robustness argument rather than weakening it.

**Code changes.** `IBM_Cloud/run_hvk_hardware_reconstruction.py` and
`run_hardware_robustness_simulator_sweep.py`: the HVK1D checkpoint path said `Main2/` but
the checkpoint is under `main2/` -- harmless on Windows, fatal on this case-sensitive
filesystem, and it blocked every replay until fixed. New:
`IBM_Cloud/run_provenance_campaign.py` (drives the 25 jobs, records the CRN, emits the
ledger) and `IBM_Cloud/write_f4_section.py` (regenerates §F4 from the ledger so the two
cannot drift). No retained artifact under `IBM_Cloud/outputs/` was modified; all new
output is confined to `outputs/provenance_campaign/`.

**Environment.** This ran in the repository venv,
`Hamiltonian_Vision_Kernel/.venv` -- `qiskit` 2.4.1, `qiskit-ibm-runtime` 0.47.0,
`qiskit-ionq` 1.1.1 (installed for this campaign) -- not the
`/home/adminpc/Desktop/HVK/.venv` with `qiskit` 2.5.2 / runtime 0.49.0 used on
2026-09-04. Different environment; noted so the log does not imply continuity.

---

## 2026-09-04

### H1 closed -- as a negative result: the account that ran the jobs no longer exists
**What was run.** `IBM_Cloud/check_ibm_credentials.py` (new, see below) and
`IBM_Cloud/verify_hardware_jobs.py`, against the student's live IBM Cloud API key. The
repository venv (`/home/adminpc/Desktop/HVK/.venv`) held no `qiskit` at all, which is the
second reason the earlier attempt stopped; `qiskit` 2.5.2 and `qiskit-ibm-runtime` 0.49.0
were installed first.

**Finding -- the credential was never the problem.** The key is valid: IBM Cloud issued
an access token from it. It simply reaches no Qiskit Runtime instance, so no job can be
retrieved. Four facts fix the diagnosis:

- the jobs were submitted on `channel="ibm_quantum_platform"`
  (`IBM_Cloud/run_hvk_hardware_reconstruction.py:309`), so they ran under an IBM Cloud
  instance, not on the retired `quantum.ibm.com` platform;
- the account holding that instance -- an IIT Bhubaneswar trial,
  `12586c7ca151474297be30db083c3bcb` -- is now `CANCELED`;
- the one remaining active account, `059312687e3b4f8484a4d6cd7c311a3d`, authenticates but
  holds no instance whatsoever (resource controller returns `rows_count: 0`);
- an IBM Cloud API key cannot be re-scoped to a different account (IAM refuses with
  `BXNIM0413E`), and job history is scoped to the instance that submitted the job, so a
  freshly created instance would not reach these jobs either.

Service-side confirmation of the hardware jobs is therefore permanently out of reach.
There is no key, from any account the student can log into, that would close it.

**No result changes.** Every hardware number in both manuscripts is backed by its
retained artifact and recomputes from it. What is lost is the *second*, service-side line
of evidence -- a limit on provenance, not on the results -- and it is now disclosed
rather than parked as an open to-do.

**Recorded in `RESULTS_MAP.md`.** The §F2 header states the finding with the evidence
above. The 15 IBM ledger rows now read `not retrievable (account closed)` and the 10 IonQ
rows `IonQ — not IBM-retrievable`; the instance/CRN line is marked unrecoverable, with a
note that future campaigns should serialize `service.active_account()["instance"]` into
the output JSON next to the job ID. The status label `backed (device-side pending)`
became `backed (device-side unverifiable)` in all nine places it appeared, including the
coverage tally, since "pending" implies a wait that will never end.

**Tooling left behind.** `IBM_Cloud/check_ibm_credentials.py` reports four things
separately -- key present, IBM Cloud accepts it, which instances/CRNs it reaches, and
whether one ledger job actually retrieves -- and on failure names every IBM Cloud account
the IBMid can reach together with its state, so a dead key is never mistaken for a missing
job. `verify_hardware_jobs.py` now exits with a readable message instead of a traceback
when the account will not open, and accepts `--instance <CRN>`.

**Open, for the supervisor (logged as H2 in `todo.md`).** Whether the manuscripts should
carry a one-line note that the cited job IDs are no longer retrievable from the service
because the submitting account was closed. A referee who tries a job ID will get nothing
back. Recommendation: disclose it rather than be asked. No `.tex` file was touched for
this.

---

### H2 and I done; H1 remains credential-gated
**H2 -- clean local build confirmed.** Rebuilt from a clean aux/log state using the
available TeX Live toolchain (`latexmk -pdf`) because the repository helper expects
MiKTeX's `mpm`, which is not installed on this Linux machine. Final PDFs were moved
into `overleaf_docs/assets/`: `paper_hvk_springer.pdf` = 24 pp and
`supplementary_study.pdf` = 27 pp. Final logs contain 0 undefined citations and 0
undefined references.

**I1/I2 -- collapsed raw/local control presentation.** The retained code and artifacts
support reading `local-observables-only` and `raw-linear-classical` as the same retained
raw/local linear map in this bundle, not as two distinct controls. The duplicate rows in
the held-out CIFAR, paired-test, and TOST tables were merged; the main-paper and
supplement counts now say five of seven distinct controls are TOST-equivalent; the
multi-dataset paragraph now refers to the same raw/local control; and the methods
section clarifies that resource matching is at readout-facing 32-D width, with narrower
descriptors zero-padded before the readout.

**Still open -- H1.** *(Superseded later the same day -- see "H1 closed" above. Retained
for the record.)* The live IBM account check still requires the student's IBM
Quantum credentials/account access. A local attempt on this machine stopped before
account retrieval because `qiskit_ibm_runtime` is not installed:

    python IBM_Cloud/verify_hardware_jobs.py --write-map

Do not mark H1 complete until `RESULTS_MAP.md` is updated with the retrieved
instance/CRN and per-job account-verification status.

---

## 2026-09-01

### G1-G3 done (item G, student) --- one of them changes what the supplement claims
**G1 -- confirmed a miscitation, replaced with the primary source.** The `.bib` entry
`Fei2021` was S.-M. Fei, "Compressed Sensing Based on Tensor Network Machine Learning",
*Phys Sci & Biophys J* 5(1):000174 (2021). Retrieved and read: it is a **3-page Mini
Review** in a Medwin Publishers journal, and its own reference [9] is
Ran, Sun, Fei, Su, Lewenstein, *Tensor network compressed sensing with unsupervised
machine learning*, **Phys. Rev. Research 2, 033293 (2020)** (arXiv:1907.10290) --- i.e. it
is a short summary, by one co-author, of exactly the paper the supervisor identified.
The citation sits in the intro's tensor-network list, where the primary source is what
belongs. `Fei2021` is deleted and replaced by `Ran2020TNCS` (author list, title, volume,
article number, year and DOI verified against arXiv and the journal listing); the single
`\citep` in `paper_hvk_springer.tex` now points at it. Paper rebuilds with 0 undefined
citations and renders as "Ran et al. 2020".

**G2 -- the answer is no, and the supplement now says so.** The question was whether the
11.73 / 11.57 dB in `tab:topology_real_circuit` is meaningfully above the random-VQC
floor at the same 90-step budget. The artifact had no such control, so one was run:
`Main2/newHVK/run_topology_random_vqc_control.py`, identical protocol (same two training
images, same overlapping 8x8 patches at stride 4, same 90 steps, same per-topology
learning rate, same three held-out images and seeds) with the observable vector replaced
by resampled noise --- the study's own `random-vqc`, which costs no QNode calls.

Result: the floor is **12.74 +/- 2.21 dB (HVK1D)** and **12.64 +/- 2.12 dB (HVK2D)**. The
trained circuits sit about **1 dB below their own floor** (-1.01 and -1.08 dB on the
means), negative in all six seed-level comparisons (two-sided Wilcoxon p = 0.031). At 90
steps neither model has trained far enough for its latent to beat an input-independent
predictor. A paragraph in Section 4.1 now states this plainly: the absolute PSNRs in that
table are not reconstruction quality, only the 1D-vs-2D difference is meaningful, and
even that compares two runs measured before either clears the floor --- the full-scope
surrogate sweep, not this subset, is what carries the topology conclusion. Artifact:
`Main2/newHVK/results/topology_comparison/random_vqc_control.json`.

**G3 -- decided (b): explicitly single-seed, no physical claim.** The chi non-monotonicity
paragraph in Section 3.6 previously ended "bond dimension is therefore a coupled
optimization hyperparameter here, not a monotonic compression-fidelity knob" --- a
mechanism inferred from one seed. Rewritten to attach no mechanism: it is a single-seed
observation at one budget on one image, the seed-level variance on the neighbouring q=4
axis (+/-1.64 dB) is of the same order as the chi gaps themselves, the
chi=4-over-parameterizes-a-low-entanglement-target reading is named as interesting but
explicitly not tested, and confirming it would need the same sweep over several seeds at
a matched budget (roughly 12 real-circuit runs, not run). The interrupted follow-up
reached only chi=1 and chi=2 at a shorter budget and cannot settle it either.

**Also fixed while here:** eight driver scripts hardcoded
`REPO_ROOT = Path(r"c:\Users\HP\Desktop\HVK\Hamiltonian_Vision_Kernel")`, so every
command RESULTS_MAP.md cites from them only ran on one Windows machine. All eight now
derive the repo root from `__file__`.

---


### F1 done, F3 closed, F2 half done (item F, student)
**F1 — `overleaf_docs/RESULTS_MAP.md` written.** One row per Springer table and figure:
driver script, exact command, key parameters, retained artifact, status. All 8 tables and
6 figures of the paper and all 15 tables and 8 figures of the supplement are covered, with
numbering taken from the compiled PDFs' `.aux` `\newlabel` entries rather than guessed, and
a `Prior ID` column carrying the old `R1`-`R14` IDs so it reads side by side with
`TODO/results-core-map.md` (which is untouched, and was restored after being deleted --
F1 and F3 both refer to it). Two items are `descriptive` (`tab:variants`,
`tab:differentiation` -- definitions, nothing to reproduce); one is `partial`
(`tab:hamiltonian_controls` / `tab:hamiltonian_reproducibility`, whose gap the supplement
already discloses at length); everything else is `backed`, with the artifact value quoted
against the printed one where it was recomputed.

**F3 -- closed, no artifact needed.** R10's open row was `contrastive+no-energy` (33.33 dB).
That value is not in either Springer manuscript: `33.33` appears nowhere in
`paper_hvk_springer.tex` or `supplementary_study.tex`. What survives is §3.7's account of
the *withdrawal*, which quotes `32.24 -> 33.30` and contrastive `32.84` precisely to say
they are superseded and that three of the old table's six rows are not independently
reproducible. Those are cited as history, not as results. Nothing to produce, nothing to
cut; the row reopens only if a withdrawn value is ever promoted back into a claim.

**F2 -- the map is ready, the account check is not done.** Every job identifier the
manuscripts cite is now in one ledger in RESULTS_MAP.md -- 25 jobs (5 reconstruction
pilot, 4 repeated-execution anchors, 16 archived replays) with backend, shots, circuit
count and PSNR read back from the retained JSON, all marked `pending` in an `Account`
column. `IBM_Cloud/verify_hardware_jobs.py` was written to close that column: it retrieves
each job, compares backend/shots/status against what the manuscripts claim, prints the
account's instance/CRN, and with `--write-map` rewrites the column in place.

It has **not been run against the service**: this machine has no IBM Quantum credentials
(no `~/.qiskit/qiskit-ibm.json`, no `IBM_QUANTUM_TOKEN`), so nothing below the `Account`
column has been confirmed by anything except local artifacts. Whoever holds the account
runs `python IBM_Cloud/verify_hardware_jobs.py --write-map` and records the instance/CRN
line. The 16 IonQ rows are not IBM jobs and are skipped by design -- they need the IonQ
console if provenance for them is wanted. The job-ID enumeration and the `--write-map`
rewrite were both exercised offline (`--list`, and a dry run against a copy of the map).

---

## 2026-08-31

### Manuscript folders consolidated: `overleaf_docs/` is now the only copy
`latex_outputs/paper_latex/` (IEEEtran `paper_hvk.tex`, the Aug-20 `paper_hvk_sn.tex`
Springer snapshot, an Aug-13 supplement, a cover letter still addressed to IEEE TQE, and
a 73-file figure archive) and `submission_bundle/` (the abandoned TQE bundle, IEEE
sources) were deleted; both remain in git history. Every duplicate was older than its
`overleaf_docs/` counterpart, so there is now one source per document.

Live references retargeted rather than left dangling:
- `.github/workflows/pages.yml` stages `overleaf_docs/assets/paper_hvk_springer.pdf`
  (the Pages site would otherwise fail on a missing file).
- The nine `IBM_Cloud/plot_*`/`generate_ansatz_figures.py` scripts now write figures to
  `overleaf_docs/assets/figures/`.
- `tests/test_submission_claim_audit.py` reads the Springer manuscript for both its
  headline-string guard (now `R^2=0.9735`, the Springer rounding, instead of the IEEE
  file's `0.974`) and its no-transition-claims scan; `submission_claim_audit.json`
  updated to match. 19 tests pass.
- `README.md`, `REPRODUCE.md`, `latex_outputs/README.md`, `project_artifacts/results.md`,
  `Main_new/README.md`, `latex_outputs/compile_tex.py` docstring.

`author_contributions.pdf` and `literature_review.pdf` had no build under
`overleaf_docs/`; both were compiled into `assets/` so every source here has its PDF.
`latex_outputs/` now holds only `compile_tex.py`, `images_latex/` and its README.

## 2026-08-27

### Physics review pass (both manuscripts + .bib) — 6 fixes applied, 3 sent to student
Supervisor high-effort read. Applied directly (text/reference only, no numbers change):
- **A1 (supplement):** renamed the "global order parameter" $M_z(t)$ to "global
  magnetization summary" and added a sentence stating we avoid "order parameter"
  since there is no symmetry breaking/transition. Fixes a term-of-art contradiction
  with the paper's own no-transition proof.
- **A2 (supplement):** added that $S$ (MPS bond entropy) is constant in $t$, so
  $R_{ES}(t)$ tracks the learned energy $H(t)$ up to a fixed scale — pre-empts
  "why divide by a constant?".
- **A3 (main, entanglement-necessity):** added the explicit mechanism — a linear
  readout of $\{\langle O_i\rangle\}$ cannot form $\langle O_i\rangle\langle O_j\rangle$,
  while the entangling measurement supplies $\langle O_iO_j\rangle$ directly; scoped
  the claim to the linear protocol (a nonlinear classical readout could recover it).
- **A4 (main, §dequant):** added a clause distinguishing the circuit's entangling
  power (strongly-entangling ansatz, genuinely entangled) from whether the measured
  pair-observable channel carries task-relevant info (a property of the target).
- **A5 (main, §regularizer_inert):** promoted from supplement — the learned couplings
  enter only the energy term and never reach the decoder, so the Hamiltonian is a
  read-only observable, not an inductive bias; that is the precise sense of "diagnostic."
- **B1 (.bib):** corrected Chakraborty2018 author names to the verified
  Sanjay Chakraborty, Sudhindu Bikash Mandal, Soharab Hossain Shaikh (were garbled).
  Verified against the Springer record; bbl now renders "Mandal SB, Shaikh SH".
Both recompiled: main 24 pp, supplement 26 pp, 0 undefined citations.
Assigned to student as todo.md block G (need code/data/citation access): **G1** Fei2021
likely-miscitation (→ Ran et al. PRResearch 2, 033293, 2020?), **G2** topology 11.7 dB
vs random floor, **G3** bond-dim non-monotonicity (single-seed → confirm or scope).

### Main paper: Tier-1 quality enhancements (`paper_hvk_springer.tex`)
Senior-author critical-read pass; three fixes:
1. **Data-availability contradiction fixed.** The Statements block still said the
   audit covers "the held-out comparison the tested HVK map **loses**" --- a
   deficit-register phrase that survived the framing pass. Now: "the
   resource-matched held-out comparison, on which the tested HVK map is competitive
   with rather than superior to the classical controls." Matches the body register.
2. **Author-contributions formatting fixed.** Was a broken `description` list with an
   empty `\item[Author contributions]` label followed by dangling name items. Now one
   clean CRediT paragraph (both authors), + "Both authors reviewed and approved."
3. **Formal problem statement added to Sec. 2.** New `\paragraph{Problem statement.}`
   with an explicit objective: patches $x_p$, MPS features, the Pauli-correlator map
   $\Phi(x_p)\in\mathbb{R}^d$ ($d=27$/$19$), decoder $D_\theta$ with Fourier encoding
   $\pi(p)$, and the reconstruction loss $\mathcal{L}(\theta)=\sum_p\|x_p-\hat{x}_p\|^2
   + \lambda E(\Phi(x_p))$ (Eq. objective). Notation $\Phi$ is consistent with the D4
   section. Raises technical polish; the loss was previously only implicit.
Recompiled: 24 pp, 0 undefined citations (the added problem-statement paragraph +
display equation restored the page the Discussion trim had reclaimed). (Considered
but deferred to the student's pass: SSIM precision normalization, negative-$R^2$
footnote.)

### Assigned to student: re-prepare results-code map for the Springer manuscript
Added task block F to `overleaf_docs/todo.md`:
- **F1** — produce `overleaf_docs/RESULTS_MAP.md` re-targeting every row from the old
  `paper_hvk.tex` §IV/§V labels to the current Springer section/table/figure IDs.
  (Do not overwrite `TODO/results-core-map.md`.)
- **F2** — real-hardware cross-check against the IBM Quantum account (only the student
  has access): confirm each job ID exists, backend, shots, PSNR, and record the
  instance/CRN + job IDs in the map. Supervisor verified all headline NUMBERS already
  survive unchanged in the Springer paper (no drift), so this is re-mapping +
  HW-provenance, not a re-run.
- **F3** — close R10 (`contrastive+no-energy`, 33.33 dB, no artifact).
- Reference punch-list items A1/A2/B1/B2 in that file marked DONE (verified by
  supervisor; no bib edits needed).

### Cover letter re-targeted to QMI (`cover_letter.tex`)
- Addressee "Editor-in-Chief, IEEE TQE" -> "Editor-in-Chief, *Quantum Machine
  Intelligence* (Springer)"; "Article in IEEE TQE" -> "research article in QMI".
- Scope paragraph rewritten from "fits TQE's scope (quantum-circuit design,
  hybrid learning, hardware-validated evaluation)" to QMI's scope (quantum machine
  learning, hybrid classical-quantum architectures, tensor-network methods for
  structured data, hardware-validated).
- Register fix: "still beating the two weakest controls by a wide margin" ->
  "substantially outperforms the two null controls" (same rule as the manuscripts).
- Closing: "IEEE's author-posting policy ... traditional (non-open-access) route"
  -> "Springer Nature's self-archiving policy ... standard subscription route".
- Compiles: 2 pp. STILL TO FILL (supervisor): suggested reviewers (line ~50),
  Siddhartha Patra ORCID (line ~68).

### Reference punch-list (student's `todo.md`) — all verified, no bib edits needed
- **West et al. 2024 erratum:** `sn-bibliography.bib` already has both `West2024`
  (PRX Quantum 5, 030320) and `West2024Erratum` (PRX Quantum 6, 020902, 2025, DOI
  10.1103/g4zk-79l9); both are cited in the paper. Resolved.
- **Krizhevsky 2009 co-author:** single-author `{Krizhevsky, Alex}` is correct —
  Hinton is advisor, not a listed author on the 2009 CIFAR tech report. No change.
- **Xu2021 pages:** `13569--13578` matches the official CVF open-access proceedings
  and is used consistently in both docs. No change.
- **Wolberg1993:** present in the supplement's own `thebibliography` with correct
  UCI attribution + DOI 10.24432/C5DW2B. No change. (It is NOT in the `.bib` because
  the supplement uses a self-contained bibliography, not `sn-bibliography.bib`.)

### Supplement — positive-framing sweep (`supplementary_study.tex`)
- **Table caption (was ~line 128):** "does not beat local/raw controls, although it
  remains far above random-VQC" -> "is competitive with local/raw controls (TOST
  equivalence, +/-1 dB) and substantially outperforms the random-VQC null control."
  Removes the banned verb "beat" and the deficit-first register.
- **Sec. Held-Out (was ~line 157):** "HVK2D exceeds both by a wide, practically
  meaningful margin" -> "HVK2D substantially outperforms both." Drops the
  null-control brag phrasing, consistent with the main paper.
- **LEFT as-is (correct register):** the meta-discussion of "beats"/"loses to" as
  terminology (quoted), and "ties ... rather than beating it" describing the
  internal energy-loss ablation (not an HVK-vs-classical claim).

### Main paper — Discussion trim (`paper_hvk_springer.tex`)
- Merged `\subsection{HVK as a representative instance of a broader design pattern}`
  (~40 lines) into `\subsection{Task-dependent representational scope}` as one
  appended paragraph; re-homed `\label{sec:representativeness}` (cross-referenced in
  Secs. 1 and 6) onto the merged subsection. Removed triple repetition + hedging.
- Demoted `\subsection{Architectural differentiation matrix}` (a 6-line pointer) to
  a lead-in paragraph on the table; unreferenced `\label{sec:differentiation}`
  removed. Discussion 8 -> 6 subsections. PDF 24 -> 23 pp.

### Main paper — positive-framing pass (`paper_hvk_springer.tex`)
- **Abstract:** led the held-out result with the positive TOST equivalence finding
  ("statistically competitive with resource-matched controls") instead of "does not
  exceed classical controls (18.12 vs 18.80)"; exact numbers moved to the body (Sec.
  5.3). Still 242 words.
- **Contribution 3:** "beating the two weakest controls by a wide margin" ->
  "substantially outperforms the two null controls (strict classical random features
  and a random-VQC)."
- **Sec. 5.3:** dropped "by a wide, practically meaningful margin" brag.
- Register rule (locked): *competitive* => resource-matched baselines (TOST only);
  *outperforms* => null controls only; never "beats." Body honesty (-0.68 dB, CI,
  Wilcoxon p) unchanged.

### Folder reorganization
- Created `assets/`; moved figures to `assets/figures/` and compiled outputs to
  `assets/`. Added `\graphicspath{{assets/}}` to both manuscript preambles.
