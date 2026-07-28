# Hardware study strengthening

## Reviewer ask

Repeated jobs, multiple calibration periods, error mitigation, uncertainty intervals,
hardware-aware training, and matched classical/noisy-simulator controls. Currently:
"five per-image-trained reconstructions, one principal trial, 256 shots, no mitigation,
no hardware-level statistical benchmark... establishes executability, not effectiveness."

## What already exists

More than the review implies at a glance: `sec:hardware_anchors` already has repeated
execution on a second/third backend (`ibm_marrakesh`, `ibm_kingston`) at two shot
budgets; `sec:hardware_robustness` already has a calibrated-noise-simulator sweep
(`FakeFez`, 4 shot budgets, 3 repeated executions) as a zero-quota complement. What's
still missing, matching the reviewer's specific list: error mitigation (no
zero-noise-extrapolation / readout-error mitigation used anywhere), and uncertainty
intervals reported as such (the anchor-points table reports point estimates, not CIs).

## Blocking constraint

This is the one item in this tracker that is **not** purely a local-compute decision —
it needs real IBM Quantum Runtime time. The paper already documents the account's free-
tier quota usage (`~35s used of 600s/month` as of the last recorded reading). Before any
code is written here:

## Open question for project owner

1. Is there current IBM Quantum quota available (free-tier resets monthly, or is there a
   paid allocation), and how much are you willing to spend on this?
2. Error mitigation (e.g. Qiskit Runtime's built-in resilience levels) can likely be
   turned on with a small code change to the existing `run_on_ibm` call path
   (`IBM_Cloud/run_ibm_hvk_probe.py`) — worth doing cheaply even before a full repeated-
   jobs study, since it's close to free in engineering time.

## Status

Not started — blocked on the quota question above.
