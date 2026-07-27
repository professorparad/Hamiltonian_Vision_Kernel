"""Qubit-count sweep of the order-parameter / energy / R_ES change-point diagnostic.

Extends run_finite_size_phase_transition.py's corrected (noise-free,
post-optimizer-step, evaluation-mode) protocol -- the replacement for the
withdrawn training-mode traces (see paper_hvk.tex Section "A Magnetization-
Style Change-Point Diagnostic") -- along three axes at once:

1. Qubit count N in {2, 4, 6, 8} (the prior finite-size study covered only
   N in {4, 6}; N=8 was previously skipped over concern it wouldn't finish
   in an hour, but a timing probe on this machine shows N=8 takes ~6
   minutes for 200 epochs, so it is included here).
2. Five images: Monalisa plus the same four CIFAR-10 images used by the
   real-hardware reconstruction pilot (cat, ship/hydrofoil, ship/sea-boat,
   airplane), instead of one CIFAR image.
3. The energy-to-entanglement ratio R_ES(t) = H(t)/S (run_critical_temperature.py's
   diagnostic), tracked alongside the order parameter at N=6 -- the only
   qubit count where the classical MPS bond entropies and the quantum
   circuit's bonds correspond 1:1 -- to check whether R_ES's own detected
   change epoch coincides with the order parameter's.

This is the same descriptive change-point formalism used throughout this
project (median-plus-two-std threshold on the trace's absolute step-to-step
change): a training-dynamics diagnostic, not a physical thermodynamic phase
transition, a critical exponent, or an infinite-system claim. Single seed
pair per (image, N); tagged exploratory_pending_statistical_review like the
rest of this project's phase-transition work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
QUBIT_COUNTS = [2, 4, 6, 8]
SEEDS = [0, 1]
R_ES_QUBIT_COUNT = 6  # only N where classical MPS bonds == quantum circuit bonds, 1:1
ENTROPY_SLICE = slice(17, 22)  # matches run_critical_temperature.py's n_sites=6 feature layout

WORKSPACE = ROOT / "Main2" / "newHVK"
OUTPUT_DIR = WORKSPACE / "results" / "qubit_energy_phase_transition"
MONALISA_PATH = MAIN_DIR / "data" / "monalisa.jpg"
CIFAR_DIR = BENCH_DIR / "datasets" / "images"


def load_monalisa() -> np.ndarray:
    import cv2

    img = cv2.imread(str(MONALISA_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(MONALISA_PATH)
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)


def load_cifar_images(n: int) -> list[tuple[str, np.ndarray]]:
    import cv2

    paths = sorted(CIFAR_DIR.glob("*.png"))[:n]
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        out.append((p.stem, cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)))
    return out


def load_images() -> list[tuple[str, np.ndarray]]:
    return [("monalisa", load_monalisa())] + load_cifar_images(4)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_change_point(trace: list[float]) -> dict:
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

    s_total = None
    if qubit_count == R_ES_QUBIT_COUNT:
        bond_entropies = np.maximum(features[:, ENTROPY_SLICE].mean(axis=0), 1e-6)
        s_total = float(bond_entropies.sum())

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

        # Noise-free evaluation-mode pass: training-mode forward deliberately
        # injects observable noise (QuantumModel.forward), which is why the
        # original per-epoch traces were withdrawn (see module docstring).
        model.eval()
        with torch.no_grad():
            eval_obs, eval_energies = model(features_t, positions)
        order_trace.append(float(eval_obs[:, :qubit_count].mean().item()))
        energy_trace.append(float(eval_energies.mean().item()))

    result: dict = {
        "order_trace": order_trace,
        "energy_trace": energy_trace,
        "order_transition": detect_change_point(order_trace),
    }
    if s_total is not None:
        r_es_trace = [e / s_total for e in energy_trace]
        result["r_es_trace"] = r_es_trace
        result["s_total"] = s_total
        result["r_es_transition"] = detect_change_point(r_es_trace)
    return result


def save_trace_plot(result: dict, image_name: str, qubit_count: int, seed: int, output_path: Path) -> None:
    order = np.array(result["order_trace"])
    energy = np.array(result["energy_trace"])
    epochs = np.arange(len(order))
    energy_scale = np.max(np.abs(energy)) or 1.0
    order_transition = result["order_transition"]
    susceptibility = np.array(
        [0.0] + [abs(order[i] - order[i - 1]) for i in range(1, len(order))]
    )

    has_r_es = "r_es_trace" in result
    n_panels = 3 if has_r_es else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.6))

    axes[0].plot(epochs, order, label="order parameter")
    axes[0].plot(epochs, energy / energy_scale, label="energy (scaled)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Order Parameter and Energy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, susceptibility, color="tab:orange", label="susceptibility")
    if order_transition["detected"]:
        axes[1].axvline(
            order_transition["critical_epoch"], color="crimson", linestyle="--",
            label=f"critical epoch={order_transition['critical_epoch']}",
        )
    axes[1].axhline(order_transition["threshold"], color="purple", linestyle=":", linewidth=1.2, label="detection threshold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Susceptibility")
    axes[1].set_title("Phase Transition Signal (order parameter)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    if has_r_es:
        r_es = np.array(result["r_es_trace"])
        r_es_transition = result["r_es_transition"]
        axes[2].plot(epochs, r_es, color="tab:purple", label=r"$R_{ES}(t)$")
        if r_es_transition["detected"]:
            axes[2].axvline(
                r_es_transition["critical_epoch"], color="crimson", linestyle="--",
                label=f"critical epoch={r_es_transition['critical_epoch']}",
            )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel(r"$R_{ES}(t) = H(t)/S$")
        axes[2].set_title("Phase Transition Signal (R_ES / \"temperature\")")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"{image_name}, N={qubit_count}, seed={seed} (noise-free evaluation-mode trace)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubit-counts", nargs="+", type=int, default=QUBIT_COUNTS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"Using device: {device}", flush=True)

    images = load_images()
    print(f"Images: {[name for name, _ in images]}", flush=True)

    results_path = OUTPUT_DIR / "qubit_energy_scaling.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else []
    already_done = {(r["image"], r["qubit_count"], r["seed"]) for r in all_results}

    for image_name, image in images:
        for qubit_count in args.qubit_counts:
            for seed in args.seeds:
                key = (image_name, qubit_count, seed)
                if key in already_done:
                    print(f"skip (already done): image={image_name} N={qubit_count} seed={seed}", flush=True)
                    continue
                print(f"\n=== image={image_name} N={qubit_count} seed={seed} ===", flush=True)
                try:
                    result = train_with_tracking(image, device, args.epochs, seed, qubit_count)
                except Exception as exc:
                    print(f"  SKIPPED due to error: {exc}", flush=True)
                    continue
                record = {
                    "image": image_name,
                    "qubit_count": qubit_count,
                    "seed": seed,
                    "trace_mode": "post_update_eval_mode",
                    "evidentiary_status": "exploratory_pending_statistical_review",
                    **result,
                }
                all_results.append(record)
                results_path.write_text(json.dumps(all_results, indent=2))
                ot = result["order_transition"]
                msg = f"  order: critical_epoch={ot['critical_epoch']} max_change={ot['max_change']:.4f}"
                if "r_es_transition" in result:
                    rt = result["r_es_transition"]
                    msg += f" | R_ES: critical_epoch={rt['critical_epoch']} max_change={rt['max_change']:.4f}"
                print(msg, flush=True)

                if seed == args.seeds[0]:
                    plot_path = OUTPUT_DIR / f"{image_name}_N{qubit_count}_seed{seed}_phase_transition.png"
                    save_trace_plot(result, image_name, qubit_count, seed, plot_path)
                    print(f"  plot: {plot_path}", flush=True)

    # Summary: critical-epoch stats per (image, N), plus order/R_ES coincidence at N=6.
    summary: dict = {}
    images_seen = sorted({r["image"] for r in all_results})
    for image_name in images_seen:
        summary[image_name] = {}
        for qc in sorted({r["qubit_count"] for r in all_results if r["image"] == image_name}):
            rows = [r for r in all_results if r["image"] == image_name and r["qubit_count"] == qc]
            order_epochs = [r["order_transition"]["critical_epoch"] for r in rows if r["order_transition"]["detected"]]
            entry = {
                "n": len(rows),
                "n_detected_order": len(order_epochs),
                "mean_critical_epoch_order": float(np.mean(order_epochs)) if order_epochs else None,
                "std_critical_epoch_order": float(np.std(order_epochs)) if order_epochs else None,
            }
            if qc == R_ES_QUBIT_COUNT:
                r_es_epochs = [r["r_es_transition"]["critical_epoch"] for r in rows if r.get("r_es_transition", {}).get("detected")]
                entry["n_detected_r_es"] = len(r_es_epochs)
                entry["mean_critical_epoch_r_es"] = float(np.mean(r_es_epochs)) if r_es_epochs else None
                entry["std_critical_epoch_r_es"] = float(np.std(r_es_epochs)) if r_es_epochs else None
                if entry["mean_critical_epoch_order"] is not None and entry["mean_critical_epoch_r_es"] is not None:
                    entry["order_r_es_epoch_gap"] = abs(entry["mean_critical_epoch_order"] - entry["mean_critical_epoch_r_es"])
            summary[image_name][str(qc)] = entry

    print("\n=== Summary ===")
    for image_name, per_n in summary.items():
        for qc, s in per_n.items():
            line = f"image={image_name} N={qc}: n={s['n']} order_detected={s['n_detected_order']} mean_t_c_order={s['mean_critical_epoch_order']}"
            if "mean_critical_epoch_r_es" in s:
                line += f" | R_ES_detected={s['n_detected_r_es']} mean_t_c_r_es={s['mean_critical_epoch_r_es']} gap={s.get('order_r_es_epoch_gap')}"
            print(line)
    (OUTPUT_DIR / "qubit_energy_scaling_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
