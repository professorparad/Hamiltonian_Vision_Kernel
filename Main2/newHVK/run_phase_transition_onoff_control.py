"""On/off control for the phase-transition change-point diagnostic (Task C).

paper_hvk.tex (Section~\\ref{sec:phase_transition}) reports a change-point detection
rate of 16/24 across six datasets using the corrected, noise-free, evaluation-mode
order-parameter trace, with detection decided by a median + 2*std threshold on the
trace's step-to-step change magnitude. That section (and q1_revision/07_phase_transition_
scope/README.md) is heavily caveated as "descriptive, not inferential" -- but the one
control that would rule out "this just fires on every run regardless" as a threshold
artifact was never built (see experiments/todo.md Task 5 and TODO/todo.md's guardrail
note on phase transition).

This script builds that control. It reuses the exact same protocol/codebase that
produced the paper's existing phase-transition numbers --
Main/src/quantum/quantum_model.py::QuantumModel + Main/src/decoder/patch_decoder.py,
the same feature/positional pipeline, and the same median+2*std detect_change_point
rule as Main2/newHVK/run_finite_size_phase_transition.py and
Main2/newHVK/run_qubit_energy_phase_transition.py (NOT Main_new -- deliberately kept on
the codebase that generated the paper's 16/24 number, for an apples-to-apples protocol
match) -- and runs it on matched seeds/datasets for two conditions:

  (a) Hamiltonian-ON  : standard QuantumModel (learned VQC + Heisenberg energy loss).
  (b) Hamiltonian-OFF : QuantumModel(use_classical_replacement=True) -- a classical
                         Linear+tanh map with energy identically 0.0 for every patch,
                         every epoch, by construction (see quantum_model.py forward()).

If the detector only fires when the Hamiltonian is ON, that's a real positive
validation of the diagnostic. If it ALSO fires when OFF (energy == 0 throughout),
that confirms the guardrail's suspicion that median+2*std on a small trace is a
threshold artifact, not physics -- and this script reports that outcome plainly
either way; it does not suppress a negative result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "Main"
BENCH_DIR = ROOT / "Baselines" / "cifar10_comparisons"
for p in (MAIN_DIR, BENCH_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.preprocessing.patching import extract_patches  # noqa: E402
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding  # noqa: E402
from src.tensornetworks.mps_features import extract_mps_features  # noqa: E402
from src.quantum.quantum_model import QuantumModel  # noqa: E402
from src.decoder.patch_decoder import PatchDecoder  # noqa: E402
from src.training.training import resolve_device  # noqa: E402

PATCH_SIZE = 8
PATCH_STRIDE = 8
N_SITES = 6  # classical MPS site count for feature extraction; independent of qubit_count
POSITIONAL_DIM = 4
IMAGE_SIZE = 32
EPOCHS = 200
LR = 0.004
QUBIT_COUNT = 6
DEFAULT_SEEDS = [0, 1]

WORKSPACE = ROOT / "Main2" / "newHVK"
OUTPUT_DIR = WORKSPACE / "results" / "phase_transition_onoff_control"
MONALISA_PATH = MAIN_DIR / "data" / "monalisa.jpg"
CIFAR_DIR = BENCH_DIR / "datasets" / "images"


def load_monalisa() -> np.ndarray:
    import cv2

    img = cv2.imread(str(MONALISA_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(MONALISA_PATH)
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_cifar_image(stem_prefix: str) -> np.ndarray:
    import cv2

    matches = sorted(CIFAR_DIR.glob(f"{stem_prefix}*.png"))
    if not matches:
        raise FileNotFoundError(f"No CIFAR image matching '{stem_prefix}*' in {CIFAR_DIR}")
    img = cv2.imread(str(matches[0]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_images() -> list[tuple[str, np.ndarray]]:
    return [("monalisa", load_monalisa()), ("cifar_cat", load_cifar_image("0000_cat"))]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_change_point(trace: list[float]) -> dict:
    """Same median + 2*std threshold-on-step-change rule used throughout this project
    (paper_hvk.tex's Eq. for the critical epoch; identical to
    run_qubit_energy_phase_transition.py::detect_change_point and
    Main_new/src/training/order_parameters.py::detect_phase_transition)."""
    diffs = [0.0] + [abs(trace[i] - trace[i - 1]) for i in range(1, len(trace))]
    diffs_arr = np.array(diffs)
    threshold = float(np.median(diffs_arr) + 2 * np.std(diffs_arr))
    idx = int(np.argmax(diffs_arr))
    peak = float(diffs_arr[idx])
    detected = bool(peak > threshold and peak > 0)
    return {
        "critical_epoch": idx if detected else -1,
        "max_change": peak,
        "threshold": threshold,
        "detected": detected,
        "final_value": trace[-1],
    }


def train_with_tracking(
    image: np.ndarray, device: torch.device, epochs: int, seed: int, hamiltonian_on: bool
) -> dict:
    set_seed(seed)
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)
    features_t, positions, targets = features_t.to(device), positions.to(device), targets.to(device)

    model = QuantumModel(
        feature_dim=features_t.shape[1],
        positional_dim=POSITIONAL_DIM,
        qubit_count=QUBIT_COUNT,
        use_classical_replacement=not hamiltonian_on,
    ).to(device)
    decoder = PatchDecoder(
        observable_dim=model.observable_dim, positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE
    ).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    order_trace: list[float] = []
    energy_trace: list[float] = []
    for _ in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()

        # Noise-free evaluation-mode pass: training-mode forward deliberately injects
        # observable noise (QuantumModel.forward), which is why the original per-epoch
        # traces were withdrawn (see paper_hvk.tex sec:phase_transition).
        model.eval()
        with torch.no_grad():
            eval_obs, eval_energies = model(features_t, positions)
        order_trace.append(float(eval_obs[:, :QUBIT_COUNT].mean().item()))
        energy_trace.append(float(eval_energies.mean().item()))

    return {
        "hamiltonian_on": hamiltonian_on,
        "order_trace": order_trace,
        "energy_trace": energy_trace,
        "mean_energy_over_training": float(np.mean(energy_trace)),
        "energy_identically_zero": bool(np.allclose(energy_trace, 0.0)),
        "order_transition": detect_change_point(order_trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Resume from results/phase_transition_onoff_control/raw_traces.json, skipping (image, seed, on/off) combos already recorded there.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    images = load_images()
    print(f"Images: {[name for name, _ in images]}  Seeds: {args.seeds}  Epochs: {args.epochs}", flush=True)

    raw_traces_path = OUTPUT_DIR / "raw_traces.json"
    all_results: list[dict] = []
    if args.skip_existing and raw_traces_path.exists():
        all_results = json.loads(raw_traces_path.read_text(encoding="utf-8"))
        print(f"Resuming: {len(all_results)} runs already recorded in {raw_traces_path}", flush=True)
    done = {(r["image"], r["seed"], r["hamiltonian_on"]) for r in all_results}

    for image_name, image in images:
        for seed in args.seeds:
            for hamiltonian_on in (True, False):
                label = "ON" if hamiltonian_on else "OFF"
                if (image_name, seed, hamiltonian_on) in done:
                    print(f"SKIP (already done) image={image_name} seed={seed} Hamiltonian={label}", flush=True)
                    continue
                print(f"\n=== image={image_name} seed={seed} Hamiltonian={label} ===", flush=True)
                t0 = time.time()
                result = train_with_tracking(image, device, args.epochs, seed, hamiltonian_on)
                elapsed = time.time() - t0
                record = {
                    "image": image_name,
                    "seed": seed,
                    "hamiltonian_on": hamiltonian_on,
                    "elapsed_seconds": elapsed,
                    **result,
                }
                all_results.append(record)
                # Incremental save: write after every single run, not just at the end, so a
                # kill/interrupt mid-sweep does not lose already-completed (and expensive)
                # runs -- this is the fix for the earlier version of this script, which only
                # wrote raw_traces.json/summary.json after the full loop and lost 3 completed
                # (image, seed) pairs' full traces when the process was killed mid-run.
                raw_traces_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
                ot = result["order_transition"]
                print(
                    f"  Hamiltonian={label}: detected={ot['detected']} critical_epoch={ot['critical_epoch']} "
                    f"max_change={ot['max_change']:.6f} threshold={ot['threshold']:.6f} "
                    f"mean_energy={result['mean_energy_over_training']:.6f} "
                    f"energy_identically_zero={result['energy_identically_zero']} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    (OUTPUT_DIR / "raw_traces.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    on_runs = [r for r in all_results if r["hamiltonian_on"]]
    off_runs = [r for r in all_results if not r["hamiltonian_on"]]
    n_on_fired = sum(1 for r in on_runs if r["order_transition"]["detected"])
    n_off_fired = sum(1 for r in off_runs if r["order_transition"]["detected"])

    fires_only_when_on = (n_on_fired > 0) and (n_off_fired == 0)
    if n_on_fired > 0 and n_off_fired == 0:
        guardrail_result = (
            "PASS: the change-point detector fires only when the Hamiltonian is on, and does NOT fire on the "
            "classical-replacement (Hamiltonian-off, energy identically 0) control -- a real positive validation "
            "of the diagnostic, ruling out 'fires on every run regardless' as a threshold artifact."
        )
    elif n_on_fired > 0 and n_off_fired > 0:
        guardrail_result = (
            "FAIL: the classical-replacement (Hamiltonian-off, energy identically 0) control ALSO triggers "
            "detection under the same median+2*std threshold. This confirms the guardrail's worry in "
            "experiments/todo.md Task 5 -- a median+2*std threshold on a small change-magnitude trace can be a "
            "threshold artifact, not physics. Reported honestly, not suppressed."
        )
    elif n_on_fired == 0 and n_off_fired == 0:
        guardrail_result = (
            "INCONCLUSIVE: neither the Hamiltonian-on nor the Hamiltonian-off runs triggered detection under "
            "this configuration (epochs/seeds/images). This does not validate the diagnostic either way -- it "
            "means this particular run set never crossed its own median+2*std threshold in either condition. "
            "Reported honestly rather than forced into a PASS or FAIL."
        )
    else:  # n_on_fired == 0 and n_off_fired > 0
        guardrail_result = (
            "FAIL (inverted): the classical-replacement (Hamiltonian-off, energy identically 0) control "
            "triggered detection while the Hamiltonian-on runs did not. This is the opposite of what a working "
            "diagnostic should show and is reported honestly."
        )

    summary = {
        "protocol": (
            "median + 2*std threshold on |delta M_z(t)| (same rule as paper_hvk.tex sec:phase_transition / "
            "run_qubit_energy_phase_transition.py::detect_change_point / "
            "Main_new/src/training/order_parameters.py::detect_phase_transition)"
        ),
        "epochs": args.epochs,
        "seeds": args.seeds,
        "images": [name for name, _ in images],
        "n_hamiltonian_on_runs": len(on_runs),
        "n_hamiltonian_on_detected": n_on_fired,
        "n_hamiltonian_off_runs": len(off_runs),
        "n_hamiltonian_off_detected": n_off_fired,
        "fires_only_when_hamiltonian_on": fires_only_when_on,
        "guardrail_result": guardrail_result,
        "per_run": [
            {
                "image": r["image"],
                "seed": r["seed"],
                "hamiltonian_on": r["hamiltonian_on"],
                "detected": r["order_transition"]["detected"],
                "critical_epoch": r["order_transition"]["critical_epoch"],
                "max_change": r["order_transition"]["max_change"],
                "threshold": r["order_transition"]["threshold"],
                "energy_identically_zero": r["energy_identically_zero"],
            }
            for r in all_results
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINAL RESULT ===")
    print(f"Hamiltonian ON : {n_on_fired}/{len(on_runs)} runs detected a change point")
    print(f"Hamiltonian OFF: {n_off_fired}/{len(off_runs)} runs detected a change point")
    print(guardrail_result)
    print(f"\nWrote {OUTPUT_DIR / 'raw_traces.json'}")
    print(f"Wrote {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
