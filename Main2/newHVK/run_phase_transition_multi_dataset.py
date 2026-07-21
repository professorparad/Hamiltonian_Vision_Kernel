"""Record deterministic HVK2D training-dynamics traces on six real datasets.

After every optimizer update, this runner performs a separate evaluation-mode
forward pass and records its noise-free global order parameter. Runs are
repeated over explicit seeds. Legacy ``*_order_traces.json`` files were recorded
in training mode and are retained only for provenance; new evidence is written
to distinct ``*_eval_order_traces.json`` files so the two cannot be mixed.
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

from src.preprocessing.positional_encoding import sinusoidal_positional_encoding
from src.training.training import resolve_device
from Main2.src.model import Quantum2DGridModel, PatchDecoder as PatchDecoder2D
from src.tensornetworks.mps_features import extract_mps_features

PATCH_SIZE = 8
PATCH_STRIDE = 8
N_SITES = 6
POSITIONAL_DIM = 4
IMAGE_SIZE = 32
EPOCHS = 200
LR = 0.004

WORKSPACE = ROOT / "Main2" / "newHVK"
OUTPUT_DIR = WORKSPACE / "results" / "phase_transition_multi_dataset"


def extract_patches(image: np.ndarray, patch_size: int, stride: int):
    height, width = image.shape
    patches, positions = [], []
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            patches.append(image[i : i + patch_size, j : j + patch_size])
            positions.append([i / height, j / width])
    return np.array(patches, dtype=np.float32), np.array(positions, dtype=np.float32)


def resize_to_32(image: np.ndarray) -> np.ndarray:
    import cv2

    img = np.asarray(image, dtype=np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_cifar_images(n: int) -> list[np.ndarray]:
    import cv2

    image_dir = BENCH_DIR / "datasets" / "images"
    paths = sorted(image_dir.glob("*.png"))[:n]
    images = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        images.append(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))
    return images


def load_torchvision_images(name: str, n: int) -> list[np.ndarray]:
    from torchvision import datasets

    root = WORKSPACE / "datasets"
    if name == "mnist":
        dataset = datasets.MNIST(root=str(root), train=True, download=True)
    elif name == "fashion-mnist":
        dataset = datasets.FashionMNIST(root=str(root), train=True, download=True)
    else:
        raise ValueError(name)
    data = dataset.data.numpy() if hasattr(dataset.data, "numpy") else np.asarray(dataset.data)
    return [resize_to_32(data[i]) for i in range(n)]


def load_medmnist_images(name: str, n: int) -> list[np.ndarray]:
    import medmnist
    from medmnist import INFO

    info = INFO[name]
    dataset_class = getattr(medmnist, info["python_class"])
    root = WORKSPACE / "datasets"
    dataset = dataset_class(split="train", root=str(root), download=True, as_rgb=False, size=28)
    imgs = np.asarray(dataset.imgs)
    if imgs.ndim == 4:
        imgs = imgs.mean(axis=-1)
    return [resize_to_32(imgs[i]) for i in range(n)]


DATASET_LOADERS = {
    "cifar10": lambda n: load_cifar_images(n),
    "mnist": lambda n: load_torchvision_images("mnist", n),
    "fashion-mnist": lambda n: load_torchvision_images("fashion-mnist", n),
    "pathmnist": lambda n: load_medmnist_images("pathmnist", n),
    "bloodmnist": lambda n: load_medmnist_images("bloodmnist", n),
    "pneumoniamnist": lambda n: load_medmnist_images("pneumoniamnist", n),
}


def order_parameter_from_observables(observables: torch.Tensor) -> float:
    z = observables[:, :6]
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


def train_with_tracking(image: np.ndarray, device: torch.device, epochs: int, seed: int) -> dict:
    set_seed(seed)
    patches, raw_positions = extract_patches(image, PATCH_SIZE, PATCH_STRIDE)
    # Guard against exact-zero patches (common in MNIST/medmnist backgrounds), which
    # produce a zero-norm MPS and crash the internal SVD with inf/NaN.
    safe_patches = patches + 1e-4
    features = np.array([extract_mps_features(p, n_sites=N_SITES, bond_dim=4) for p in safe_patches])
    features_t = torch.tensor(features, dtype=torch.float32)
    features_t = (features_t - features_t.mean(dim=0)) / (features_t.std(dim=0, unbiased=False) + 1e-8)
    positions = sinusoidal_positional_encoding(raw_positions, d_model=POSITIONAL_DIM)
    targets = torch.tensor(patches, dtype=torch.float32).unsqueeze(1)

    features_t, positions, targets = features_t.to(device), positions.to(device), targets.to(device)

    model = Quantum2DGridModel(feature_dim=features_t.shape[1], positional_dim=POSITIONAL_DIM).to(device)
    decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
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
        order_trace.append(order_parameter_from_observables(eval_observables))

    transition = detect_phase_transition(order_trace)
    transition["order_trace"] = order_trace
    return transition


def summarize_saved_traces() -> list[dict]:
    """Rebuild the excluded legacy aggregate for provenance only."""
    summary = []
    for trace_path in sorted(OUTPUT_DIR.glob("*_order_traces.json")):
        if trace_path.name.endswith("_eval_order_traces.json"):
            continue
        dataset = trace_path.name.removesuffix("_order_traces.json")
        traces = json.loads(trace_path.read_text())
        per_image = []
        for trace in traces:
            result = detect_phase_transition(trace)
            per_image.append(result)
        critical_epochs = [r["critical_epoch"] for r in per_image if r["detected"]]
        max_susceptibilities = [r["max_susceptibility"] for r in per_image]
        final_orders = [r["final_order_parameter"] for r in per_image]
        summary.append(
            {
                "dataset": dataset,
                "trace_mode": "legacy_training_mode_with_observable_noise",
                "evidentiary_status": "excluded",
                "n_images": len(per_image),
                "n_detected": len(critical_epochs),
                "mean_critical_epoch": float(np.mean(critical_epochs)) if critical_epochs else None,
                "mean_max_susceptibility": float(np.mean(max_susceptibilities)),
                "std_max_susceptibility": float(np.std(max_susceptibilities)),
                "mean_final_order_parameter": float(np.mean(final_orders)),
                "std_final_order_parameter": float(np.std(final_orders)),
                "per_image": per_image,
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-per-dataset", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LOADERS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Recompute summary.json from all retained order-trace files without training.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.rebuild_summary:
        summary = summarize_saved_traces()
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"Rebuilt {OUTPUT_DIR / 'summary.json'} from {len(summary)} trace files.")
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    existing_summary_path = OUTPUT_DIR / "summary.json"
    existing_rows = json.loads(existing_summary_path.read_text()) if existing_summary_path.exists() else []
    summary_by_dataset = {row["dataset"]: row for row in existing_rows}
    for name in args.datasets:
        print(f"\n=== {name} ===", flush=True)
        images = DATASET_LOADERS[name](args.images_per_dataset)
        per_image = []
        trace_records = []
        for idx, image in enumerate(images):
            for seed in args.seeds:
                try:
                    result = train_with_tracking(image, device, args.epochs, seed)
                except Exception as exc:
                    print(f"  image {idx}, seed {seed}: SKIPPED due to error: {exc}")
                    continue
                per_image.append(result)
                trace_records.append({"image_index": idx, "seed": seed, "order_trace": result["order_trace"]})
                print(
                    f"  image {idx}, seed {seed}: critical_epoch={result['critical_epoch']} "
                    f"max_susceptibility={result['max_susceptibility']:.4f} "
                    f"final_order_parameter={result['final_order_parameter']:.4f}",
                    flush=True,
                )
        if not per_image:
            print(f"  WARNING: no images succeeded for {name}, skipping dataset in summary.")
            continue
        critical_epochs = [r["critical_epoch"] for r in per_image if r["detected"]]
        max_susceptibilities = [r["max_susceptibility"] for r in per_image]
        final_orders = [r["final_order_parameter"] for r in per_image]
        summary_by_dataset[name] = {
                "dataset": name,
                "trace_mode": "post_update_eval_mode",
                "evidentiary_status": "exploratory_pending_statistical_review",
                "seeds": args.seeds,
                "n_images": len(per_image),
                "n_detected": len(critical_epochs),
                "mean_critical_epoch": float(np.mean(critical_epochs)) if critical_epochs else None,
                "mean_max_susceptibility": float(np.mean(max_susceptibilities)),
                "std_max_susceptibility": float(np.std(max_susceptibilities)),
                "mean_final_order_parameter": float(np.mean(final_orders)),
                "std_final_order_parameter": float(np.std(final_orders)),
                "per_image": [{k: v for k, v in r.items() if k != "order_trace"} for r in per_image],
            }
        (OUTPUT_DIR / f"{name}_eval_order_traces.json").write_text(
            json.dumps(trace_records, indent=2)
        )

    summary = [summary_by_dataset[key] for key in sorted(summary_by_dataset)]
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Summary ===")
    for row in summary:
        print(
            f"{row['dataset']}: n={row['n_images']} detected={row['n_detected']} "
            f"mean_t_c={row['mean_critical_epoch']} "
            f"mean_X_max={row['mean_max_susceptibility']:.4f}+/-{row['std_max_susceptibility']:.4f} "
            f"mean_M_final={row['mean_final_order_parameter']:.4f}"
        )
    print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
