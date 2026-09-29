# PCAE paper — rewrites of the red blocks (2026-09-26)

Positive-tone, paper-ready replacements for every red block in the pasted Overleaf
version, up to where the paste was cut off (start of the D4 section). Each entry says
what the repo shows, then gives LaTeX to paste in place of the red block. Drop the
`{\color{red}...}` wrapper when you paste.

---

## R1. Rename note (after `\maketitle`)

Not paper text — a to-do. Delete the red block once done:

1. Regenerate figures whose labels say HVK/HVK1D/HVK2D (at least `hvk1d_ansatz.pdf`,
   `hvk2d_ansatz.pdf`; check the hardware and shot-sweep figures too).
2. Rename the model in `supplementary_study.tex`, including its title.
3. Update the 3 places the paper cites the supplement title (Intro "A companion
   document…", §Competitive "the companion document…", and one further on).
   Suggested new supplement title:
   *Supplementary Material for the Pauli-Correlator Autoencoder: Resource-Matched
   Ablations, Validation, and Reproducibility*.

## R2. Eq. (obs_dim) block

The equation is correct (12+15=27, 12+7=19, 12+5=17). Keep it; just remove the red colour.

## R3. Loss equation (Eq. objective)

Repo: supplement uses patch means for both terms; code uses `torch.mean(energies)`
(`Main_new2/src/training.py:81`). Replace the equation and the red note with:

```latex
\begin{equation}
  \label{eq:objective}
  \mathcal{L}(\theta)=\frac{1}{P}\sum_{p=1}^{P}\big\|x_p-\hat{x}_p\big\|_2^{2}
  \;+\;\lambda\,\frac{1}{P}\sum_{p=1}^{P}E\big(\Phi(x_p)\big),
\end{equation}
```

(If the reconstruction term in code is a per-pixel MSE, write `\frac{1}{P}\sum_p
\mathrm{MSE}(x_p,\hat{x}_p)` instead — match supplement eq. `\mathcal{L}_{\text{rec}}`.)

## R4. "Please confirm" on the D4 / 4×4 patch grid

Confirmed: supplement line 479 defines the D4 action as the permutation action on the
4×4 patch grid. Delete the red note; keep the green/add text as is.

## R5. MPS preprocessing details (three questions)

Repo (`Main_new/src/tensornetworks/mps_features.py`): the patch is flattened row-major,
ℓ2-normalised, reshaped to `[2]*n` (each site = one bit of the pixel index), converted
with quimb `from_dense`, compressed to χ=4 and renormalised. All features are full
contractions ⟨ψ|O|ψ⟩ or Schmidt spectra, so they do not depend on the gauge.

**Warning — (3) is a real gap.** I ran it: an all-zero 8×8 patch crashes
(`LinAlgError: Array must not contain infs or NaNs`). The code adds 1e-8 to the norm,
but the zero vector stays zero and quimb fails. Either no all-zero patch occurred in
the reported runs, or a different code path handled them. Check this before writing
(3) — MNIST backgrounds are the likely case. Suggested text assumes a fix/skip is
confirmed:

```latex
Each patch is flattened in row-major order into a vector of $2^{n}$ values
($n=6$ for $8\times8$ patches, $n=12$ for $64\times64$ patches) and reshaped
into an $n$-site tensor of local dimension~2, so that site~$k$ indexes the
$k$-th bit of the pixel position. The MPS is obtained by sequential SVD and
truncated to $\chi=4$. All features are computed as full contractions
$\langle\psi|O|\psi\rangle$ or from Schmidt spectra, so they are independent
of the MPS gauge and no particular canonical form is required.
[All-zero patches: <state what the code does once verified>.]
```

## R6. Exact expectation values in training

Confirmed: training devices are `lightning.gpu` / `lightning.qubit` / `default.qubit`
with no `shots` argument (`Main_new2/src/model.py:20-30`), i.e. analytic expectations.
Delete the red note; keep the add text.

## R7. Held-out map is 32-D + quadratic control definition

Repo: supplement Table `tab:resource_capacity` — PCAE2D real-CIFAR map is 32-D, 6 pair
channels / 14 CNOTs. Quadratic control (`main2/newHVK/run_newhvk_suite.py:954`): 18
local features, 9 products of adjacent local-feature pairs, the sines of those products,
and 8 positional features, selected to 32-D. Replace the red note with (and add the
same two sentences to the supplement):

```latex
The held-out PCAE2D map is the 32-D variant of Table~S-resource\_capacity,
with six pair channels and 14 CNOTs, rather than the 19-D map of
Table~\ref{tab:configurations}. The quadratic classical control augments the
18 local features with the nine products of adjacent local-feature pairs and
their sines, together with the positional features, selected to the same
32-D width.
```

## R8. The three optional statistics sentences

All three are correct and they strengthen the section. Recommend adding them (and to
the supplement):

```latex
With five seeds, $p=0.0625$ is the smallest attainable two-sided Wilcoxon
$p$-value, so the $t$-based TOST is the informative test at this sample
size. The $t$-based 90\% interval is deliberately more conservative than the
bootstrap interval, which tends to be narrow at $n=5$. For scale, a 1~dB
margin corresponds to a factor of about $1.26$ in MSE.
```

## R9. When was the ±1 dB margin fixed?

Repo: `run_tost_equivalence.py`, its results and the "pre-declared" wording were all
added in the **same commit** (`a4b15c6`, 2026-08-01). No earlier dated commit fixes the
margin, so "pre-declared" can't be supported. Recommend: remove "pre-declared" from
the abstract, contribution 3 and the conclusion, and justify the margin with the MSE
factor, not the observed 3–7 dB gaps (supplement line 132). Positive wording:

```latex
The margin is fixed at $\pm1$~dB, a change of about $1.26\times$ in MSE,
which is below the level at which reconstruction differences are visually
apparent at these PSNR values.
```

(Only keep the "visually apparent" clause if you can cite or show it.)

## R10. Multiple comparisons (Holm)

Verified: with Holm over 7 tests, four equivalences survive (ZZ-only, quadratic,
shuffled-pair, and no-entanglement at 0.008×4=0.032). Raw/local (0.034×3=0.10) does not.
Recommendation for the PI discussion: **option (b)** — name the no-entanglement control
in the abstract. It is the more interesting comparison anyway (same circuit, no
entanglement), and it survives the correction. Abstract sentence:

```latex
... establishes that PCAE2D is statistically \emph{competitive} with four
resource-matched controls, including a no-entanglement circuit, after Holm
correction for multiple comparisons.
```

Main-text replacement for the red note + the "no correction" add sentence:

```latex
Under a Holm correction across the seven comparisons, four equivalences
remain (no entanglement, $ZZ$-only, quadratic classical, shuffled pair
observables).
```

Contribution 3 then reads "four of seven distinct … (five without correction)".

## R11. "It replaces an earlier table that we withdrew…"

```latex
It supersedes an earlier table, and every row is now backed by retained
per-seed records.
```

## R12. Unmitigated runs: SamplerV2 settings, calibration, qubits, shots

Repo: all hardware scripts call `SamplerV2(mode=backend)` with **no options**, after
`transpile(..., optimization_level=1)`. SamplerV2 defaults keep dynamical decoupling and
twirling off, and SamplerV2 has no resilience/mitigation layer. So "unmitigated" holds.
Shots per image: Monalisa 16 patches × 3 bases × 256 = 12,288; each CIFAR image
16 × 2 × 256 = 8,192.

**Physical qubits / calibration (checked 2026-09-27):** the paper's own 9 IBM job IDs
return `RuntimeJobNotFound`. They were submitted from an account that no longer exists
(RESULTS_MAP §F2), so their layouts and calibration times are **unrecoverable**
(`IBM_Cloud/outputs/job_layouts.md`). The 2026-09-05 provenance re-campaign re-ran the
same circuits, and those jobs *are* retrievable: layouts, run times, calibration times
and readout errors are in `IBM_Cloud/outputs/job_layouts_ledger.md`. Those are new runs
with different PSNRs, so they must not be presented as the paper's runs. Suggested text:

```latex
Circuits were transpiled at optimisation level~1 and run through
\texttt{SamplerV2} with default options, so dynamical decoupling, Pauli
twirling and readout mitigation were all disabled. Each image used
256~shots per basis, i.e.\ $12{,}288$ shots for \textit{Monalisa}
($16\times3$ bases) and $8{,}192$ for each CIFAR-10 image ($16\times2$).
The transpiler selected a six-qubit layout per circuit. A documented
re-execution of the same circuits, with its physical qubits, calibration
timestamps and readout errors, is given in the Supplementary Material.
```

## R13. "It stays out of the main text because…"

```latex
It is reported in full in the Supplementary Material as an interpretability
readout, alongside the training-dynamics traces it supports: it is a
single-patch, single-seed probe, and the threshold it tracks appears with
and without the Hamiltonian term (Section~\ref{sec:phase_transition}).
```

## R14. Wrong section reference

The training-dynamics diagnostic is §`sec:phase_transition`. Change
"serves the training-dynamics diagnostic of Section~\ref{sec:regularizer_inert}" to
`Section~\ref{sec:phase_transition}` and delete the red note.
