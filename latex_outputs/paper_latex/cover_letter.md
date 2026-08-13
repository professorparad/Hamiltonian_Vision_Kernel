# Cover Letter

**To:** Editor-in-Chief, IEEE Transactions on Quantum Engineering (TQE)

**Re:** Submission of the manuscript "Hamiltonian Vision Kernel: Symmetry, Entangling
Correlators, and Real-Hardware Replay in Hybrid Quantum Image Reconstruction"

Dear Editor,

We are submitting the enclosed manuscript, "Hamiltonian Vision Kernel: Symmetry,
Entangling Correlators, and Real-Hardware Replay in Hybrid Quantum Image
Reconstruction," for consideration as an Article in IEEE Transactions on Quantum
Engineering.

The manuscript introduces the Hamiltonian Vision Kernel (HVK), a hybrid
quantum-classical framework for image reconstruction that combines matrix-product-state
patch features, a variational quantum circuit exposing a Pauli-correlator latent space,
Fourier positional encoding, a learnable Hamiltonian energy diagnostic, and a classical
decoder. We treat HVK not as a claim of quantum advantage but as a testbed for studying
how symmetry, topology, and Hamiltonian structure manifest in a trainable hybrid vision
model, and we report both the framework's structural properties and its behavior on
real quantum hardware. We believe this fits TQE's scope directly: it sits at the
intersection of quantum-circuit design, hybrid classical-quantum learning, and
hardware-validated evaluation, and it takes deliberate care to characterize where the
approach's evidence is strong and where it is exploratory rather than confirmatory.

Four results anchor the manuscript:

1. The pooled HVK2D observable map achieves $D_4$ group-equivariance by construction,
   with a measured numerical equivariance error of $9.57 \times 10^{-17}$.
2. On a leakage-audited diagnostic task requiring distant-patch correlations, the
   entangling observable channel reaches mean $R^2 = 0.974$ across five seeds, against
   $R^2 \leq 0.02$ for non-entangling controls -- evidence that the model's entangling
   structure, not incidental capacity, is responsible for this capability.
3. Five images are reconstructed from observables measured on real IBM Quantum
   hardware, without retraining the decoder, reaching 25.90-31.52 dB PSNR -- a scoped
   hardware feasibility pilot, reported with complete job metadata and simulator-transfer
   checks rather than as a general performance claim.
4. Exploratory training-dynamics diagnostics (a magnetization-style change-point
   statistic and an energy-to-entanglement ratio) are tracked across qubit count and
   MPS bond dimension in simulation and separately replayed on real hardware via
   gate-for-gate checkpoint reconstruction; we are explicit throughout that these are
   descriptive, small-sample observations, not inferential evidence of a genuine
   thermodynamic phase transition.

A companion supplementary study accompanies the manuscript with resource-matched
held-out controls, leakage audits, statistical significance testing, and complete
hardware execution methodology (backend names, job IDs, shot counts). Code, trained
checkpoints, and raw result artifacts backing every reported number are maintained in a
public repository (see `REPRODUCE.md` in the submission materials for a one-to-one
table/figure-to-script map).

This manuscript is not under review at any other journal, and all authors have approved
this submission.

<!-- FILL: Conflict-of-interest statement. State any relevant financial/professional
     relationships, or "The authors declare no conflicts of interest." This must come
     from the authors/supervisor, not be assumed. -->

<!-- FILL: Suggested reviewers (2-3 names, affiliations, emails). Needs domain
     knowledge of who is active and appropriate in variational quantum algorithms /
     quantum machine learning / tensor-network methods for images -- a judgment call
     for the authors and supervisor, not something to guess. -->

We thank the editors and reviewers for their time and consideration.

Sincerely,

Sparsho Chakraborty<br>
School of Basic Sciences, Department of Physics, Indian Institute of Technology
Bhubaneswar, Odisha, India<br>
sparshochakraborty123@gmail.com, 25ph05023@iitbbs.ac.in<br>
ORCID: 0009-0004-1667-7208

Siddhartha Patra<br>
Centre for Quantum Engineering, Research and Education (CQuERE), TCG CREST, Kolkata,
India<br>
siddhartha.patra@tcgcrest.org<br>
<!-- FILL: ORCID (supervisor to supply) -->
