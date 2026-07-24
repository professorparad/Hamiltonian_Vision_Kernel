"""Finite-size scaling of the order-parameter phase-transition diagnostic.

Extends the corrected diagnostic in run_phase_transition_multi_dataset.py (same
post-update evaluation-mode order-parameter tracking, same median-plus-two-std
critical-epoch detection rule) across HVK1D qubit counts N in {4, 6}. HVK2D is
not used here: its fixed 2x3 grid topology has no qubit_count parameter, unlike
HVK1D's QuantumModel, which already trains end-to-end for N=4 and N=6 (see
Main2/newHVK/run_scaling_study.py's qubit_sweep results). N=8 is excluded: a
prior 90-step probe at N=8 did not complete within a 1-hour cap on this
CPU-only circuit simulator.

Same epistemic status as the rest of this project's phase-transition work:
results are tagged "exploratory_pending_statistical_review" and reported
honestly regardless of whether a size-dependent shift in the critical epoch
is actually observed.
"""
from __future__ import annotations

import argparse
import json
import sys
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

from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.tensornetworks.mps_features import extract_mps_features
from src.quantum.quantum_model import QuantumModel
from src.decoder.patch_decoder import PatchDecoder
from src.training.training import resolve_device

PATCH_SIZE = 8
PATCH_STRIDE = 8
N_SITES = 6  # classical MPS site count for feature extraction; independent of qubit_count
POSITIONAL_DIM = 4
IMAGE_SIZE = 32
EPOCHS = 200
LR = 0.004
QUBIT_COUNTS = [4, 6]
SEEDS = [0, 1]

WORKSPACE = ROOT / "Main2" / "newHVK"
OUTPUT_DIR = WORKSPACE / "results" / "finite_size_phase_transition"


def load_cifar_images(n: int) -> list[np.ndarray]:
    import cv2

    image_dir = BENCH_DIR / "datasets" / "images"
    paths = sorted(image_dir.glob("*.png"))[:n]
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        images.append(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))
    return images


def resize_to_32(image: np.ndarray) -> np.ndarray:
    import cv2

    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_torchvision_images(name: str, n: int) -> list[np.ndarray]:
    from torchvision import datasets

    root = WORKSPACE / "datasets"
    if name == "mnist":
        dataset = datasets.MNIST(root=str(root), train=True, download=True)
    else:
        raise ValueError(name)
    data = dataset.data.numpy() if hasattr(dataset.data, "numpy") else np.asarray(dataset.data)
    return [resize_to_32(data[i]) for i in range(n)]


DATASET_LOADERS = {
    "cifar10": lambda n: load_cifar_images(n),
    "mnist": lambda n: load_torchvision_images("mnist", n),
}


def order_parameter_from_observables(observables: torch.Tensor, qubit_count: int) -> float:
    z = observables[:, :qubit_count]
    return float(z.mean().item())


def detect_phase_transition(order_trace: list[float]) -> dict:
    susceptibility = [0.0] + [abs(order_trace[i] - order_trace[i - 1]) for i in range(1, len(order_trace))]
    susceptibility_arr = np.array(susceptibility)
    threshold = float(np.median(susceptibility_arr) + 2 * np.std(susceptibility_arr))
    critical_epoch = int(np.argmax(susceptibility_arr))
    max_susceptibility = float(susceptibility_arr[critical_epoch])
    detected = bool(max_susceptibility > threshold and max_susceptibility > 0)
    return {
        "critical_epoch": critical_epoch if detected else -1,
        "max_susceptibility": max_susceptibility,
        "threshold": threshold,
        "detected": detected,
        "final_order_parameter": order_trace[-1],
        "order_parameter_jump": max_susceptibility,
    }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_with_tracking(image: np.ndarray, device: torch.device, epochs: int, seed: int, qubit_count: int) -> dict:
    set_seed(seed)
    patches, raw_positions = extract_patches(image, patch_size=PATCH_SIZE, stride=PATCH_STRIDE)
    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    features_t, positions, targets = features_t.to(device), positions.to(device), targets.to(device)

    model = QuantumModel(feature_dim=features_t.shape[1], positional_dim=POSITIONAL_DIM, qubit_count=qubit_count).to(device)
    decoder = PatchDecoder(observable_dim=model.observable_dim, positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(decoder.parameters()), lr=LR)

    order_trace = []
    for step in range(epochs):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(features_t, positions)
        output = decoder(observables, positions)
        loss = torch.mean((output - targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
        # Record after the update in evaluation mode: training mode deliberately
        # injects observable noise and is unsuitable for a dynamics trace.
        model.eval()
        with torch.no_grad():
            eval_observables, _ = model(features_t, positions)
        order_trace.append(order_parameter_from_observables(eval_observables, qubit_count))

    transition = detect_phase_transition(order_trace)
    transition["order_trace"] = order_trace
    return transition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit-counts", nargs="+", type=int, default=QUBIT_COUNTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LOADERS.keys()))
    parser.add_argument("--images-per-dataset", type=int, default=1)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    results_path = OUTPUT_DIR / "finite_size_scaling.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else []
    already_done = {
        (r["qubit_count"], r["dataset"], r["image_index"], r["seed"]) for r in all_results
    }

    for qubit_count in args.qubit_counts:
        for dataset_name in args.datasets:
            images = DATASET_LOADERS[dataset_name](args.images_per_dataset)
            for img_idx, image in enumerate(images):
                for seed in args.seeds:
                    key = (qubit_count, dataset_name, img_idx, seed)
                    if key in already_done:
                        print(f"skip (already done): N={qubit_count} dataset={dataset_name} image={img_idx} seed={seed}", flush=True)
                        continue
                    print(f"\n=== N={qubit_count} dataset={dataset_name} image={img_idx} seed={seed} ===", flush=True)
                    try:
                        result = train_with_tracking(image, device, args.epochs, seed, qubit_count)
                    except Exception as exc:
                        print(f"  SKIPPED due to error: {exc}", flush=True)
                        continue
                    record = {
                        "qubit_count": qubit_count,
                        "dataset": dataset_name,
                        "image_index": img_idx,
                        "seed": seed,
                        "trace_mode": "post_update_eval_mode",
                        "evidentiary_status": "exploratory_pending_statistical_review",
                        **result,
                    }
                    all_results.append(record)
                    results_path.write_text(json.dumps(all_results, indent=2))
                    print(
                        f"  critical_epoch={result['critical_epoch']} "
                        f"max_susceptibility={result['max_susceptibility']:.4f} "
                        f"final_order_parameter={result['final_order_parameter']:.4f}",
                        flush=True,
                    )

    summary = {}
    for qc in sorted({r["qubit_count"] for r in all_results}):
        rows = [r for r in all_results if r["qubit_count"] == qc]
        critical_epochs = [r["critical_epoch"] for r in rows if r["detected"]]
        summary[str(qc)] = {
            "n": len(rows),
            "n_detected": len(critical_epochs),
            "mean_critical_epoch": float(np.mean(critical_epochs)) if critical_epochs else None,
            "std_critical_epoch": float(np.std(critical_epochs)) if critical_epochs else None,
        }
    print("\n=== Summary ===")
    for qc, s in summary.items():
        print(f"N={qc}: n={s['n']} detected={s['n_detected']} mean_t_c={s['mean_critical_epoch']} std={s['std_critical_epoch']}")
    (OUTPUT_DIR / "finite_size_scaling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
