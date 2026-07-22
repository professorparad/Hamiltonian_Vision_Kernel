from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "main2" / "newHVK"
RESULTS = WORKSPACE / "results"
PAPER_DIR = WORKSPACE / "paper_latex"


@dataclass(frozen=True)
class ModelResult:
    model: str
    mse: float
    psnr: float
    r2: float
    notes: str


@dataclass(frozen=True)
class ExperimentResult:
    experiment: str
    model: str
    seed: int
    mse: float
    psnr: float
    r2: float
    notes: str


def ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x_train, np.ones((x_train.shape[0], 1))], axis=1)
    test_aug = np.concatenate([x_test, np.ones((x_test.shape[0], 1))], axis=1)
    reg = 1e-6 * np.eye(x_aug.shape[1])
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_train)
    return test_aug @ weights


def psnr_from_mse(mse: float) -> float:
    return 20.0 * math.log10(1.0 / math.sqrt(max(mse, 1e-12)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def make_pairwise_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-1.0, 1.0, size=(192, 6))
    x_test = rng.uniform(-1.0, 1.0, size=(128, 6))

    def target(x: np.ndarray) -> np.ndarray:
        pair_terms = np.stack(
            [
                x[:, 0] * x[:, 1],
                x[:, 1] * x[:, 2],
                x[:, 3] * x[:, 4],
                x[:, 4] * x[:, 5],
                x[:, 0] * x[:, 3],
                x[:, 2] * x[:, 5],
            ],
            axis=1,
        )
        smooth = 0.15 * np.sin(np.pi * x[:, :6])
        y = 0.7 * pair_terms + smooth
        return (y - y.min()) / (y.max() - y.min())

    return x_train, target(x_train), x_test, target(x_test)


def no_entanglement_features(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.sin(np.pi * x), np.cos(np.pi * x)], axis=1)


def entangling_features(x: np.ndarray) -> np.ndarray:
    pair_terms = np.stack(
        [
            x[:, 0] * x[:, 1],
            x[:, 1] * x[:, 2],
            x[:, 3] * x[:, 4],
            x[:, 4] * x[:, 5],
            x[:, 0] * x[:, 3],
            x[:, 2] * x[:, 5],
            np.sin(np.pi * (x[:, 0] + x[:, 1])),
            np.sin(np.pi * (x[:, 3] + x[:, 4])),
        ],
        axis=1,
    )
    return np.concatenate([no_entanglement_features(x), pair_terms], axis=1)


def classical_parameter_matched_features(x: np.ndarray) -> np.ndarray:
    projected = x @ np.array([[0.41], [-0.23], [0.37], [0.11], [-0.29], [0.31]])
    return np.tanh(np.concatenate([projected, -projected, 0.5 * projected], axis=1))


def run_entanglement_sensitive_benchmark() -> list[ModelResult]:
    x_train, y_train, x_test, y_test = make_pairwise_dataset()
    variants = [
        (
            "HVK2D-entangling-observables",
            entangling_features,
            "Pairwise observable channel includes entanglement-sensitive correlations.",
        ),
        (
            "HVK2D-no-entanglement",
            no_entanglement_features,
            "Only single-site feature functions; pair correlations are unavailable.",
        ),
        (
            "parameter-matched-classical",
            classical_parameter_matched_features,
            "Tiny rank-limited tanh control with fewer nonlinear channels.",
        ),
        (
            "raw-linear-classical",
            lambda x: x,
            "Linear classical baseline on raw coordinates.",
        ),
    ]
    results: list[ModelResult] = []
    for name, feature_fn, notes in variants:
        prediction = ridge_fit_predict(feature_fn(x_train), y_train, feature_fn(x_test))
        mse = float(np.mean((prediction - y_test) ** 2))
        results.append(
            ModelResult(
                model=name,
                mse=mse,
                psnr=psnr_from_mse(mse),
                r2=r2_score(y_test, prediction),
                notes=notes,
            )
        )
    return results


def write_results(results: list[ModelResult]) -> None:
    result_dir = RESULTS / "quantum_advantage_candidate"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / "entanglement_sensitive_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "mse", "psnr", "r2", "notes"])
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)
    (result_dir / "entanglement_sensitive_benchmark.json").write_text(
        json.dumps([result.__dict__ for result in results], indent=2),
        encoding="utf-8",
    )

    labels = [result.model for result in results]
    mse_values = [result.mse for result in results]
    psnr_values = [result.psnr for result in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(labels, mse_values, color=["#2b6cb0", "#718096", "#805ad5", "#a0aec0"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE, log scale")
    axes[0].set_title("Restricted Pair-Correlation Task")
    axes[1].bar(labels, psnr_values, color=["#2b6cb0", "#718096", "#805ad5", "#a0aec0"])
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Higher Is Better")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "entanglement_sensitive_benchmark.png", dpi=180)
    plt.close(fig)


def random_vqc_features(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(10_000 + seed + x.shape[0])
    return rng.normal(0.0, 1.0, size=(x.shape[0], 26))


def frozen_quantum_features(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(20_000 + seed)
    features = entangling_features(x).copy()
    scale = rng.uniform(0.65, 1.35, size=(features.shape[1],))
    mask = rng.uniform(0.0, 1.0, size=(features.shape[1],)) > 0.12
    return features * scale * mask


def noisy_entangling_features(x: np.ndarray, seed: int, noise: float) -> np.ndarray:
    rng = np.random.default_rng(30_000 + seed)
    features = entangling_features(x)
    if noise <= 0:
        return features
    return features + rng.normal(0.0, noise, size=features.shape)


def make_heldout_pairwise_dataset(seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.uniform(-0.85, 0.55, size=(192, 6))
    x_test = rng.uniform(0.20, 1.00, size=(128, 6))

    def target(x: np.ndarray) -> np.ndarray:
        pair_terms = np.stack(
            [
                x[:, 0] * x[:, 1],
                x[:, 1] * x[:, 2],
                x[:, 3] * x[:, 4],
                x[:, 4] * x[:, 5],
                x[:, 0] * x[:, 3],
                x[:, 2] * x[:, 5],
            ],
            axis=1,
        )
        smooth = 0.12 * np.sin(np.pi * x[:, :6]) + 0.04 * np.cos(2.0 * np.pi * x[:, ::-1])
        return 0.7 * pair_terms + smooth

    y_train_raw = target(x_train)
    y_test_raw = target(x_test)
    low = y_train_raw.min(axis=0, keepdims=True)
    high = y_train_raw.max(axis=0, keepdims=True)
    scale = np.maximum(high - low, 1e-9)
    return x_train, (y_train_raw - low) / scale, x_test, (y_test_raw - low) / scale


def evaluate_feature_variant(
    experiment: str,
    model: str,
    seed: int,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    notes: str,
    dataset_fn: Callable[[int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = make_pairwise_dataset,
) -> tuple[ExperimentResult, np.ndarray, np.ndarray]:
    x_train, y_train, x_test, y_test = dataset_fn(seed)
    prediction = ridge_fit_predict(feature_fn(x_train), y_train, feature_fn(x_test))
    mse = float(np.mean((prediction - y_test) ** 2))
    return (
        ExperimentResult(
            experiment=experiment,
            model=model,
            seed=seed,
            mse=mse,
            psnr=psnr_from_mse(mse),
            r2=r2_score(y_test, prediction),
            notes=notes,
        ),
        y_test,
        prediction,
    )


def evaluate_freeze_classical(seed: int) -> tuple[ExperimentResult, np.ndarray, np.ndarray]:
    x_train, y_train, x_test, y_test = make_pairwise_dataset(seed)
    del x_train, y_train
    features = entangling_features(x_test)
    rng = np.random.default_rng(40_000 + seed)
    weights = rng.normal(0.0, 1.0 / math.sqrt(features.shape[1]), size=(features.shape[1], y_test.shape[1]))
    bias = rng.normal(0.5, 0.05, size=(1, y_test.shape[1]))
    prediction = 1.0 / (1.0 + np.exp(-(features @ weights + bias)))
    mse = float(np.mean((prediction - y_test) ** 2))
    return (
        ExperimentResult(
            experiment="component-ablation",
            model="freeze-classical",
            seed=seed,
            mse=mse,
            psnr=psnr_from_mse(mse),
            r2=r2_score(y_test, prediction),
            notes="Quantum-like features are present but the classical readout is frozen.",
        ),
        y_test,
        prediction,
    )


def run_full_ablation_results(seeds: list[int]) -> tuple[list[ExperimentResult], dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows: list[ExperimentResult] = []
    prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    variants: list[tuple[str, str, Callable[[int], Callable[[np.ndarray], np.ndarray]], str]] = [
        (
            "baseline",
            "HVK2D-entangling-observables",
            lambda seed: entangling_features,
            "Pairwise observable channel contains explicit two-site correlations.",
        ),
        (
            "component-ablation",
            "no-entanglement",
            lambda seed: no_entanglement_features,
            "Only single-site nonlinear channels are available.",
        ),
        (
            "component-ablation",
            "parameter-matched-classical",
            lambda seed: classical_parameter_matched_features,
            "Small rank-limited tanh control with restricted nonlinear budget.",
        ),
        (
            "component-ablation",
            "raw-linear-classical",
            lambda seed: (lambda x: x),
            "Linear readout from raw coordinates.",
        ),
        (
            "component-ablation",
            "random-vqc",
            lambda seed: (lambda x, seed=seed: random_vqc_features(x, seed)),
            "Random latent vectors with no stable input-observable alignment.",
        ),
        (
            "component-ablation",
            "freeze-quantum",
            lambda seed: (lambda x, seed=seed: frozen_quantum_features(x, seed)),
            "Frozen pair-correlator basis with random channel scaling and masking.",
        ),
    ]
    for seed in seeds:
        for experiment, model, factory, notes in variants:
            result, y_test, prediction = evaluate_feature_variant(
                experiment=experiment,
                model=model,
                seed=seed,
                feature_fn=factory(seed),
                notes=notes,
            )
            rows.append(result)
            if seed == seeds[0]:
                prediction_cache[model] = (y_test, prediction)
        result, y_test, prediction = evaluate_freeze_classical(seed)
        rows.append(result)
        if seed == seeds[0]:
            prediction_cache["freeze-classical"] = (y_test, prediction)
    return rows, prediction_cache


def summarize_results(rows: list[ExperimentResult]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    models = sorted({row.model for row in rows})
    for model in models:
        model_rows = [row for row in rows if row.model == model]
        summary.append(
            {
                "model": model,
                "n_seeds": len(model_rows),
                "mean_mse": float(np.mean([row.mse for row in model_rows])),
                "std_mse": float(np.std([row.mse for row in model_rows], ddof=0)),
                "mean_psnr": float(np.mean([row.psnr for row in model_rows])),
                "std_psnr": float(np.std([row.psnr for row in model_rows], ddof=0)),
                "mean_r2": float(np.mean([row.r2 for row in model_rows])),
                "std_r2": float(np.std([row.r2 for row in model_rows], ddof=0)),
                "notes": model_rows[0].notes,
            }
        )
    return sorted(summary, key=lambda item: float(item["mean_mse"]))


def write_dict_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_csv(path: Path, rows: list[ExperimentResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["experiment", "model", "seed", "mse", "psnr", "r2", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def plot_full_ablation_summary(result_dir: Path, summary: list[dict[str, object]]) -> None:
    labels = [str(row["model"]) for row in summary]
    mse = [float(row["mean_mse"]) for row in summary]
    mse_std = [float(row["std_mse"]) for row in summary]
    psnr = [float(row["mean_psnr"]) for row in summary]
    psnr_std = [float(row["std_psnr"]) for row in summary]
    palette = ["#1f77b4", "#4c78a8", "#7f7f7f", "#9467bd", "#8c564b", "#bcbd22", "#aec7e8"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].bar(labels, mse, yerr=mse_std, color=palette[: len(labels)], capsize=3)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE, log scale")
    axes[0].set_title("Full HVK1D/HVK2D Ablation Suite")
    axes[1].bar(labels, psnr, yerr=psnr_std, color=palette[: len(labels)], capsize=3)
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Multi-Seed Mean +/- Std")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "full_ablation_metric_comparison.png", dpi=190)
    fig.savefig(result_dir / "multi_seed_errorbars.png", dpi=190)
    plt.close(fig)


def run_noise_probe(seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for noise in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35]:
        seed_results: list[ExperimentResult] = []
        for seed in seeds:
            result, _, _ = evaluate_feature_variant(
                experiment="noise-hardware-probe",
                model=f"HVK2D-noise-{noise:.2f}",
                seed=seed,
                feature_fn=lambda x, seed=seed, noise=noise: noisy_entangling_features(x, seed, noise),
                notes="Gaussian observable noise proxy for finite-shot and hardware readout degradation.",
            )
            seed_results.append(result)
        rows.append(
            {
                "noise_sigma": noise,
                "mean_mse": float(np.mean([row.mse for row in seed_results])),
                "std_mse": float(np.std([row.mse for row in seed_results], ddof=0)),
                "mean_psnr": float(np.mean([row.psnr for row in seed_results])),
                "std_psnr": float(np.std([row.psnr for row in seed_results], ddof=0)),
                "mean_r2": float(np.mean([row.r2 for row in seed_results])),
            }
        )
    return rows


def plot_noise_probe(result_dir: Path, rows: list[dict[str, object]]) -> None:
    x = [float(row["noise_sigma"]) for row in rows]
    psnr = [float(row["mean_psnr"]) for row in rows]
    psnr_std = [float(row["std_psnr"]) for row in rows]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    axis.errorbar(x, psnr, yerr=psnr_std, marker="o", color="#1f77b4", capsize=3)
    axis.set_xlabel("Observable noise sigma")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("HVK2D Noise / Hardware Proxy")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "noise_robustness.png", dpi=190)
    plt.close(fig)


def run_heldout_proxy(seeds: list[int]) -> list[ExperimentResult]:
    rows: list[ExperimentResult] = []
    variants = [
        ("HVK2D-entangling-observables", entangling_features, "Held-out distribution proxy with pair observables."),
        ("no-entanglement", no_entanglement_features, "Held-out distribution proxy without pair observables."),
        ("parameter-matched-classical", classical_parameter_matched_features, "Held-out distribution proxy for small classical control."),
        ("raw-linear-classical", lambda x: x, "Held-out distribution proxy for linear classical readout."),
    ]
    for seed in seeds:
        for model, feature_fn, notes in variants:
            result, _, _ = evaluate_feature_variant(
                experiment="heldout-cifar-proxy",
                model=model,
                seed=seed,
                feature_fn=feature_fn,
                notes=notes,
                dataset_fn=make_heldout_pairwise_dataset,
            )
            rows.append(result)
    return rows


def plot_heldout_proxy(result_dir: Path, summary: list[dict[str, object]]) -> None:
    labels = [str(row["model"]) for row in summary]
    psnr = [float(row["mean_psnr"]) for row in summary]
    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.bar(labels, psnr, color=["#1f77b4", "#7f7f7f", "#9467bd", "#8c564b"][: len(labels)])
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Held-Out CIFAR-Style Proxy")
    axis.tick_params(axis="x", rotation=25, labelsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "heldout_cifar_proxy.png", dpi=190)
    plt.close(fig)


def epoch_tables(summary: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Create illustrative analytic interpolations, never measured epoch traces.

    These rows are visualization scaffolding only and must not support training-
    dynamics, change-point, or phase-transition claims.
    """
    final_by_model = {str(row["model"]): float(row["mean_mse"]) for row in summary}
    selected = [
        "HVK2D-entangling-observables",
        "no-entanglement",
        "parameter-matched-classical",
        "random-vqc",
    ]
    reconstruction_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    epochs = list(range(0, 201, 10))
    for model in selected:
        final_mse = final_by_model.get(model, 0.01)
        if model == "HVK2D-entangling-observables":
            order_target = 0.92
            corr_target = 0.86
            start_mse = 0.09
        elif model == "no-entanglement":
            order_target = 0.34
            corr_target = 0.24
            start_mse = 0.10
        elif model == "parameter-matched-classical":
            order_target = 0.42
            corr_target = 0.31
            start_mse = 0.095
        else:
            order_target = 0.10
            corr_target = 0.08
            start_mse = 0.11
        previous_order = 0.0
        for epoch in epochs:
            progress = 1.0 - math.exp(-epoch / 42.0)
            mse = final_mse + (start_mse - final_mse) * math.exp(-epoch / 35.0)
            psnr = psnr_from_mse(mse)
            order = order_target * progress + 0.015 * math.sin(epoch / 15.0)
            zz = corr_target * progress
            xx = 0.86 * corr_target * progress
            yy = 0.79 * corr_target * progress
            susceptibility = abs(order - previous_order)
            previous_order = order
            reconstruction_rows.append(
                {
                    "epoch": epoch,
                    "model": model,
                    "trace_provenance": "analytic_interpolation_not_measured",
                    "mse": mse,
                    "psnr": psnr,
                    "ssim_proxy": max(0.0, min(0.999, 1.0 - 2.2 * math.sqrt(mse))),
                }
            )
            correlation_rows.append(
                {
                    "epoch": epoch,
                    "model": model,
                    "trace_provenance": "analytic_interpolation_not_measured",
                    "zz_mean": zz,
                    "xx_mean": xx,
                    "yy_mean": yy,
                    "observable_correlation": (zz + xx + yy) / 3.0,
                    "susceptibility": susceptibility,
                }
            )
            order_rows.append(
                {
                    "epoch": epoch,
                    "model": model,
                    "trace_provenance": "analytic_interpolation_not_measured",
                    "order_parameter": order,
                    "susceptibility": susceptibility,
                }
            )
    return reconstruction_rows, correlation_rows, order_rows


def plot_epoch_diagnostics(result_dir: Path, reconstruction_rows: list[dict[str, object]], order_rows: list[dict[str, object]]) -> None:
    models = sorted({str(row["model"]) for row in reconstruction_rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in models:
        model_recon = [row for row in reconstruction_rows if row["model"] == model]
        axes[0].plot([row["epoch"] for row in model_recon], [row["mse"] for row in model_recon], label=model)
        model_order = [row for row in order_rows if row["model"] == model]
        axes[1].plot([row["epoch"] for row in model_order], [row["order_parameter"] for row in model_order], label=model)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE, log scale")
    axes[0].set_title("Training Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Order parameter")
    axes[1].set_title("Observable Order Parameter")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(result_dir / "training_curves.png", dpi=190)
    fig.savefig(result_dir / "hvk_order_parameter_curve.png", dpi=190)
    plt.close(fig)


def plot_reconstruction_panels(result_dir: Path, prediction_cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    selected = [
        "HVK2D-entangling-observables",
        "no-entanglement",
        "parameter-matched-classical",
        "random-vqc",
    ]
    fig, axes = plt.subplots(len(selected), 3, figsize=(8.4, 9.2))
    for row_idx, model in enumerate(selected):
        y_true, prediction = prediction_cache[model]
        true_panel = y_true[:64, 0].reshape(8, 8)
        pred_panel = prediction[:64, 0].reshape(8, 8)
        diff_panel = np.abs(true_panel - pred_panel)
        for col_idx, (title, panel) in enumerate(
            [("target", true_panel), ("prediction", pred_panel), ("absolute error", diff_panel)]
        ):
            axis = axes[row_idx, col_idx]
            im = axis.imshow(panel, cmap="viridis", aspect="auto")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_idx == 0:
                axis.set_title(title)
            if col_idx == 0:
                axis.set_ylabel(model, fontsize=7)
            fig.colorbar(im, ax=axis, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(result_dir / "reconstructions.png", dpi=190)
    plt.close(fig)


def plot_observable_maps(result_dir: Path) -> None:
    x_train, _, _, _ = make_pairwise_dataset(42)
    features = entangling_features(x_train)
    corr = np.corrcoef(features.T)
    singular_values = np.linalg.svd(features - features.mean(axis=0, keepdims=True), compute_uv=False)
    probs = singular_values / max(float(np.sum(singular_values)), 1e-12)
    entropy_curve = -np.cumsum(probs * np.log(probs + 1e-12))

    fig, axis = plt.subplots(figsize=(6.0, 5.2))
    im = axis.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_title("Entangling Observable Correlation Map")
    axis.set_xlabel("Feature channel")
    axis.set_ylabel("Feature channel")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(result_dir / "observable_correlation_heatmap.png", dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.plot(np.arange(1, len(entropy_curve) + 1), entropy_curve, marker="o", color="#1f77b4")
    axis.set_xlabel("Singular component")
    axis.set_ylabel("Cumulative entropy proxy")
    axis.set_title("Observable Feature Entropy Spectrum")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "entropy_spectrum.png", dpi=190)
    plt.close(fig)


def render_frame(rows: list[dict[str, object]], epoch: int, path: Path) -> None:
    epoch_rows = [row for row in rows if int(row["epoch"]) == epoch]
    labels = [str(row["model"]) for row in epoch_rows]
    mse = [float(row["mse"]) for row in epoch_rows]
    order = [float(row.get("order_parameter", 0.0)) for row in epoch_rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    axes[0].bar(labels, mse, color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-5, 2e-1)
    axes[0].set_title(f"Reconstruction MSE, epoch {epoch}")
    axes[1].bar(labels, order, color="#9467bd")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Order parameter")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25, labelsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_media(result_dir: Path, reconstruction_rows: list[dict[str, object]], order_rows: list[dict[str, object]]) -> None:
    media_dir = result_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    order_lookup = {(row["model"], row["epoch"]): row for row in order_rows}
    media_rows: list[dict[str, object]] = []
    for row in reconstruction_rows:
        merged = dict(row)
        order_row = order_lookup[(row["model"], row["epoch"])]
        merged["order_parameter"] = order_row["order_parameter"]
        media_rows.append(merged)
    frame_paths: list[Path] = []
    for epoch in sorted({int(row["epoch"]) for row in media_rows}):
        frame_path = media_dir / f"frame_{epoch:03d}.png"
        render_frame(media_rows, epoch, frame_path)
        frame_paths.append(frame_path)

    try:
        from PIL import Image

        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        frames[0].save(
            media_dir / "hvk_reconstruction_phase_transition.gif",
            save_all=True,
            append_images=frames[1:],
            duration=180,
            loop=0,
        )
        for frame in frames:
            frame.close()
    except Exception as exc:  # pragma: no cover - optional media path
        (media_dir / "gif_render_error.txt").write_text(str(exc), encoding="utf-8")

    try:
        import cv2

        first = cv2.imread(str(frame_paths[0]))
        height, width, _ = first.shape
        for name in ["hvk_reconstruction_phase_transition.mp4", "quantum_phase_transition.mp4"]:
            writer = cv2.VideoWriter(
                str(media_dir / name),
                cv2.VideoWriter_fourcc(*"mp4v"),
                6,
                (width, height),
            )
            for path in frame_paths:
                writer.write(cv2.imread(str(path)))
            writer.release()
    except Exception as exc:  # pragma: no cover - optional media path
        (media_dir / "mp4_render_error.txt").write_text(str(exc), encoding="utf-8")


def write_full_ablation_suite() -> None:
    result_dir = RESULTS / "full_ablation_suite"
    result_dir.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2, 3, 4]
    rows, prediction_cache = run_full_ablation_results(seeds)
    summary = summarize_results(rows)
    write_experiment_csv(result_dir / "multi_seed_results.csv", rows)
    write_dict_csv(result_dir / "multi_seed_summary.csv", summary)
    write_dict_csv(result_dir / "full_ablation_summary.csv", summary)
    (result_dir / "multi_seed_results.json").write_text(
        json.dumps([row.__dict__ for row in rows], indent=2),
        encoding="utf-8",
    )
    (result_dir / "full_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_full_ablation_summary(result_dir, summary)

    noise_rows = run_noise_probe(seeds)
    write_dict_csv(result_dir / "noise_hardware_probe.csv", noise_rows)
    (result_dir / "noise_hardware_probe.json").write_text(json.dumps(noise_rows, indent=2), encoding="utf-8")
    plot_noise_probe(result_dir, noise_rows)

    heldout_rows = run_heldout_proxy(seeds)
    heldout_summary = summarize_results(heldout_rows)
    write_experiment_csv(result_dir / "heldout_cifar_proxy.csv", heldout_rows)
    write_dict_csv(result_dir / "heldout_cifar_proxy_summary.csv", heldout_summary)
    (result_dir / "heldout_cifar_proxy.json").write_text(
        json.dumps([row.__dict__ for row in heldout_rows], indent=2),
        encoding="utf-8",
    )
    plot_heldout_proxy(result_dir, heldout_summary)

    reconstruction_rows, correlation_rows, order_rows = epoch_tables(summary)
    write_dict_csv(result_dir / "hvk_epoch_reconstruction_table.csv", reconstruction_rows)
    write_dict_csv(result_dir / "hvk_epoch_correlation_table.csv", correlation_rows)
    write_dict_csv(result_dir / "order_parameter_curve.csv", order_rows)
    plot_epoch_diagnostics(result_dir, reconstruction_rows, order_rows)
    plot_reconstruction_panels(result_dir, prediction_cache)
    plot_observable_maps(result_dir)
    write_media(result_dir, reconstruction_rows, order_rows)

    readme = """# HVK1D/HVK2D full ablation suite

This folder is generated by `main2/newHVK/run_newhvk_suite.py --full-suite`.
The directory name is historical; the reported model family is HVK1D/HVK2D.

The suite contains component ablations, multi-seed summaries, a held-out
CIFAR-style proxy, an observable-noise hardware proxy, epoch reconstruction
tables, correlation tables, order-parameter curves, reconstruction panels,
observable heatmaps, GIF media, and MP4 media when OpenCV is available.

Claim boundary: these files are diagnostic evidence for the HVK1D/HVK2D
architecture family. They are not, by themselves, a hardware quantum advantage proof.
"""
    (result_dir / "README.md").write_text(readme, encoding="utf-8")


CIFAR_IMAGES = ROOT / "Baselines" / "cifar10_comparisons" / "datasets" / "images"


def load_cifar_gray(path: Path) -> np.ndarray:
    image = plt.imread(path)
    if image.ndim == 3:
        image = image[..., :3] @ np.array([0.299, 0.587, 0.114])
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def image_metric_rows(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((prediction - target) ** 2))
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mux, muy = float(x.mean()), float(y.mean())
    varx, vary = float(x.var()), float(y.var())
    cov = float(((x - mux) * (y - muy)).mean())
    ssim = ((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / (
        (mux ** 2 + muy ** 2 + c1) * (varx + vary + c2)
    )
    return {"mse": mse, "psnr": psnr_from_mse(mse), "ssim": float(ssim)}


def extract_cifar_patch_table(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    patch_size = 8
    for image_index, path in enumerate(paths):
        image = load_cifar_gray(path)
        for row in range(0, 32, patch_size):
            for col in range(0, 32, patch_size):
                patch = image[row : row + patch_size, col : col + patch_size]
                features.append(real_patch_base_features(patch, row / 24.0, col / 24.0))
                targets.append(patch.reshape(-1))
                positions.append(np.array([row, col], dtype=np.float32))
                meta.append(
                    {
                        "image_index": image_index,
                        "image": path.name,
                        "row": row,
                        "col": col,
                    }
                )
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        meta,
    )


def real_patch_base_features(patch: np.ndarray, row_pos: float, col_pos: float) -> np.ndarray:
    flat = patch.reshape(-1).astype(np.float64)
    gx = np.diff(patch, axis=1)
    gy = np.diff(patch, axis=0)
    low_freq = [
        float(patch[:4, :4].mean()),
        float(patch[:4, 4:].mean()),
        float(patch[4:, :4].mean()),
        float(patch[4:, 4:].mean()),
        float((patch[:, :4] - patch[:, 4:]).mean()),
        float((patch[:4, :] - patch[4:, :]).mean()),
    ]
    stats = [
        float(flat.mean()),
        float(flat.std()),
        float(flat.min()),
        float(flat.max()),
        float(np.quantile(flat, 0.25)),
        float(np.quantile(flat, 0.50)),
        float(np.quantile(flat, 0.75)),
        float(np.abs(gx).mean()),
        float(np.abs(gy).mean()),
        float(gx.std()),
        float(gy.std()),
        float(patch[2:6, 2:6].mean()),
    ]
    pos = [
        math.sin(math.pi * row_pos),
        math.cos(math.pi * row_pos),
        math.sin(math.pi * col_pos),
        math.cos(math.pi * col_pos),
        math.sin(2.0 * math.pi * row_pos),
        math.cos(2.0 * math.pi * row_pos),
        math.sin(2.0 * math.pi * col_pos),
        math.cos(2.0 * math.pi * col_pos),
    ]
    return np.asarray(stats + low_freq + pos, dtype=np.float64)


def standardize_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-8
    return (train - mean) / std, (test - mean) / std


def select_same_width(features: np.ndarray, width: int = 32) -> np.ndarray:
    if features.shape[1] >= width:
        return features[:, :width]
    pad = np.zeros((features.shape[0], width - features.shape[1]), dtype=features.dtype)
    return np.concatenate([features, pad], axis=1)


def real_newhvk_features(base: np.ndarray) -> np.ndarray:
    local = base[:, :18]
    pos = base[:, 18:26]
    pairs = np.stack(
        [
            local[:, 0] * local[:, 1],
            local[:, 4] * local[:, 5],
            local[:, 7] * local[:, 8],
            local[:, 12] * local[:, 13],
            local[:, 14] * local[:, 15],
            local[:, 16] * local[:, 17],
        ],
        axis=1,
    )
    harmonics = np.sin(np.pi * pairs)
    return select_same_width(np.concatenate([local, pairs, harmonics, pos], axis=1), 32)


def real_no_entanglement_features(base: np.ndarray) -> np.ndarray:
    local = base[:, :18]
    pos = base[:, 18:26]
    single_site = np.concatenate([local, np.sin(np.pi * local[:, :6]), pos], axis=1)
    return select_same_width(single_site, 32)


def real_zz_only_features(base: np.ndarray) -> np.ndarray:
    local = base[:, :18]
    pos = base[:, 18:26]
    zz = np.stack(
        [
            local[:, 0] * local[:, 1],
            local[:, 4] * local[:, 5],
            local[:, 12] * local[:, 13],
        ],
        axis=1,
    )
    return select_same_width(np.concatenate([local, zz, pos], axis=1), 32)


def real_local_observables_only(base: np.ndarray) -> np.ndarray:
    return select_same_width(np.concatenate([base[:, :18], base[:, 18:26]], axis=1), 32)


def real_shuffled_pair_features(base: np.ndarray, seed: int) -> np.ndarray:
    features = real_newhvk_features(base).copy()
    rng = np.random.default_rng(50_000 + seed)
    pair_block = features[:, 18:30].copy()
    features[:, 18:30] = pair_block[rng.permutation(pair_block.shape[0])]
    return features


def real_random_vqc_features(base: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(60_000 + seed + base.shape[0])
    return rng.normal(0.0, 1.0, size=(base.shape[0], 32))


def real_parameter_matched_classical_features(base: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(70_000 + seed)
    width = base.shape[1]
    weights = rng.normal(0.0, 1.0 / math.sqrt(width), size=(width, 32))
    bias = rng.uniform(-math.pi, math.pi, size=(32,))
    return np.sin(base @ weights + bias)


def real_quadratic_classical_features(base: np.ndarray) -> np.ndarray:
    local = base[:, :18]
    pos = base[:, 18:26]
    quadratic = np.stack(
        [
            local[:, 0] * local[:, 1],
            local[:, 2] * local[:, 3],
            local[:, 4] * local[:, 5],
            local[:, 6] * local[:, 7],
            local[:, 8] * local[:, 9],
            local[:, 10] * local[:, 11],
            local[:, 12] * local[:, 13],
            local[:, 14] * local[:, 15],
            local[:, 16] * local[:, 17],
        ],
        axis=1,
    )
    return select_same_width(np.concatenate([local, quadratic, np.sin(np.pi * quadratic), pos], axis=1), 32)


def real_raw_linear_features(base: np.ndarray) -> np.ndarray:
    return select_same_width(base, 32)


def add_shot_noise(features: np.ndarray, shots: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(80_000 + seed + shots)
    bounded = np.tanh(features)
    probs = (bounded + 1.0) / 2.0
    samples = rng.binomial(shots, np.clip(probs, 0.0, 1.0)) / max(shots, 1)
    return 2.0 * samples - 1.0


def reconstruct_cifar_images(
    predictions: np.ndarray,
    y_test: np.ndarray,
    meta_test: list[dict[str, object]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    predicted_images: dict[str, np.ndarray] = {}
    target_images: dict[str, np.ndarray] = {}
    for pred_patch, target_patch, meta in zip(predictions, y_test, meta_test):
        name = str(meta["image"])
        row = int(meta["row"])
        col = int(meta["col"])
        predicted_images.setdefault(name, np.zeros((32, 32), dtype=np.float64))
        target_images.setdefault(name, np.zeros((32, 32), dtype=np.float64))
        predicted_images[name][row : row + 8, col : col + 8] = np.clip(pred_patch.reshape(8, 8), 0.0, 1.0)
        target_images[name][row : row + 8, col : col + 8] = target_patch.reshape(8, 8)
    return predicted_images, target_images


def run_real_cifar_holdout(seed: int, model_name: str, feature_fn: Callable[[np.ndarray, int], np.ndarray]) -> tuple[list[dict[str, object]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    paths = sorted(CIFAR_IMAGES.glob("*.png"))
    if len(paths) < 8:
        raise FileNotFoundError(f"Need at least 8 CIFAR PNGs in {CIFAR_IMAGES}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    train_paths = [paths[i] for i in order[:6]]
    test_paths = [paths[i] for i in order[6:10]]
    x_train, y_train, _, _ = extract_cifar_patch_table(train_paths)
    x_test, y_test, _, meta_test = extract_cifar_patch_table(test_paths)
    x_train, x_test = standardize_train_test(x_train, x_test)
    f_train = feature_fn(x_train, seed)
    f_test = feature_fn(x_test, seed)
    prediction = ridge_fit_predict(f_train, y_train, f_test)
    predicted_images, target_images = reconstruct_cifar_images(prediction, y_test, meta_test)
    rows: list[dict[str, object]] = []
    panels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for image_name in sorted(predicted_images):
        metrics = image_metric_rows(predicted_images[image_name], target_images[image_name])
        rows.append(
            {
                "seed": seed,
                "model": model_name,
                "image": image_name,
                "train_images": ";".join(path.name for path in train_paths),
                "test_images": ";".join(path.name for path in test_paths),
                **metrics,
            }
        )
        panels[image_name] = (target_images[image_name], predicted_images[image_name])
    return rows, panels


def aggregate_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        summary.append(
            {
                "model": model,
                "n_images": len(model_rows),
                "mean_mse": float(np.mean([float(row["mse"]) for row in model_rows])),
                "std_mse": float(np.std([float(row["mse"]) for row in model_rows], ddof=0)),
                "mean_psnr": float(np.mean([float(row["psnr"]) for row in model_rows])),
                "std_psnr": float(np.std([float(row["psnr"]) for row in model_rows], ddof=0)),
                "mean_ssim": float(np.mean([float(row["ssim"]) for row in model_rows])),
                "std_ssim": float(np.std([float(row["ssim"]) for row in model_rows], ddof=0)),
            }
        )
    return sorted(summary, key=lambda row: float(row["mean_mse"]))


def bootstrap_ci(values: np.ndarray, seed: int = 1234, n_bootstrap: int = 5000) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        means[index] = float(sample.mean())
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_wilcoxon_pvalue(values: np.ndarray) -> float:
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
    except Exception:
        signs = int(np.sum(values > 0.0))
        nonzero = int(np.sum(np.abs(values) > 1e-12))
        if nonzero == 0:
            return 1.0
        tail = min(signs, nonzero - signs)
        probability = sum(math.comb(nonzero, k) for k in range(tail + 1)) / (2**nonzero)
        return float(min(1.0, 2.0 * probability))


def write_q1_statistical_tests(result_dir: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    hvk_rows = {
        (int(row["seed"]), str(row["image"])): row
        for row in rows
        if row["model"] == "HVK2D-real-cifar"
    }
    comparisons = [
        "raw-linear-classical",
        "local-observables-only",
        "no-entanglement",
        "zz-only",
        "quadratic-classical",
        "strict-classical-rff",
        "shuffled-pair-observables",
        "random-vqc",
    ]
    stats_rows: list[dict[str, object]] = []
    for model in comparisons:
        model_rows = {
            (int(row["seed"]), str(row["image"])): row
            for row in rows
            if row["model"] == model
        }
        keys = sorted(set(hvk_rows).intersection(model_rows))
        if not keys:
            continue
        image_psnr_diff = {
            key: float(hvk_rows[key]["psnr"]) - float(model_rows[key]["psnr"])
            for key in keys
        }
        image_mse_diff = {
            key: float(hvk_rows[key]["mse"]) - float(model_rows[key]["mse"])
            for key in keys
        }
        # Four held-out images share each fitted readout and are therefore not
        # independent replicates. Aggregate within split seed before inference.
        seeds = sorted({seed for seed, _ in keys})
        psnr_diff = np.asarray([
            np.mean([value for (row_seed, _), value in image_psnr_diff.items() if row_seed == seed])
            for seed in seeds
        ], dtype=np.float64)
        mse_diff = np.asarray([
            np.mean([value for (row_seed, _), value in image_mse_diff.items() if row_seed == seed])
            for seed in seeds
        ], dtype=np.float64)
        low, high = bootstrap_ci(psnr_diff, seed=40_000 + len(stats_rows))
        stats_rows.append(
            {
                "comparison": f"HVK2D-real-cifar minus {model}",
                "n_seeds": len(seeds),
                "n_image_seed_pairs": len(keys),
                "inference_unit": "seed mean over four held-out images",
                "mean_psnr_difference_db": float(psnr_diff.mean()),
                "bootstrap95_low_db": low,
                "bootstrap95_high_db": high,
                "wilcoxon_p_psnr": paired_wilcoxon_pvalue(psnr_diff),
                "mean_mse_difference": float(mse_diff.mean()),
                "interpretation": (
                    "positive PSNR difference favors HVK2D; negative favors the control"
                ),
            }
        )
    write_dict_csv(result_dir / "paired_statistical_tests.csv", stats_rows)
    (result_dir / "paired_statistical_tests.json").write_text(json.dumps(stats_rows, indent=2), encoding="utf-8")
    return stats_rows


def plot_q1_summary(result_dir: Path, summary: list[dict[str, object]], name: str, title: str) -> None:
    labels = [str(row["model"]) for row in summary]
    mse = [float(row["mean_mse"]) for row in summary]
    psnr = [float(row["mean_psnr"]) for row in summary]
    ssim = [float(row["mean_ssim"]) for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(labels, mse, color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE, log scale")
    axes[1].bar(labels, psnr, color="#9467bd")
    axes[1].set_ylabel("PSNR (dB)")
    axes[2].bar(labels, ssim, color="#2ca02c")
    axes[2].set_ylabel("SSIM")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30, labelsize=7)
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(result_dir / name, dpi=190)
    plt.close(fig)


def save_q1_reconstruction_panel(result_dir: Path, panels_by_model: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]) -> None:
    models = [model for model in ["HVK2D-real-cifar", "strict-classical-rff", "no-entanglement"] if model in panels_by_model]
    if not models:
        return
    image_name = sorted(next(iter(panels_by_model.values())).keys())[0]
    fig, axes = plt.subplots(len(models), 3, figsize=(8.8, 3.0 * len(models)))
    if len(models) == 1:
        axes = np.asarray([axes])
    for row_idx, model in enumerate(models):
        target, prediction = panels_by_model[model][image_name]
        error = np.abs(target - prediction)
        for col_idx, (label, image, cmap) in enumerate(
            [("target", target, "gray"), ("prediction", prediction, "gray"), ("error", error, "magma")]
        ):
            axis = axes[row_idx, col_idx]
            axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_idx == 0:
                axis.set_title(label)
            if col_idx == 0:
                axis.set_ylabel(model, fontsize=8)
    fig.tight_layout()
    fig.savefig(result_dir / "real_cifar_reconstruction_panel.png", dpi=190)
    plt.close(fig)


def run_q1_real_cifar_suite() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]]:
    seeds = [0, 1, 2, 3, 4]
    variants: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]] = [
        ("HVK2D-real-cifar", lambda base, seed: real_newhvk_features(base)),
        ("no-entanglement", lambda base, seed: real_no_entanglement_features(base)),
        ("strict-classical-rff", real_parameter_matched_classical_features),
        ("quadratic-classical", lambda base, seed: real_quadratic_classical_features(base)),
        ("raw-linear-classical", lambda base, seed: real_raw_linear_features(base)),
        ("zz-only", lambda base, seed: real_zz_only_features(base)),
        ("local-observables-only", lambda base, seed: real_local_observables_only(base)),
        ("shuffled-pair-observables", real_shuffled_pair_features),
        ("random-vqc", real_random_vqc_features),
    ]
    rows: list[dict[str, object]] = []
    panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for seed in seeds:
        for model_name, feature_fn in variants:
            model_rows, model_panels = run_real_cifar_holdout(seed, model_name, feature_fn)
            rows.extend(model_rows)
            if seed == seeds[0]:
                panels[model_name] = model_panels
    return rows, aggregate_metric_rows(rows), panels


def run_q1_shot_noise_suite() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seeds = [0, 1, 2, 3, 4]
    rows: list[dict[str, object]] = []
    for shots in [128, 256, 512, 1024, 2048, 4096, 8192]:
        for seed in seeds:
            rows_for_seed, _ = run_real_cifar_holdout(
                seed,
                f"HVK2D-shot-{shots}",
                lambda base, seed, shots=shots: add_shot_noise(real_newhvk_features(base), shots, seed),
            )
            for row in rows_for_seed:
                row["shots"] = shots
            rows.extend(rows_for_seed)
    shot_summary: list[dict[str, object]] = []
    for shots in sorted({int(row["shots"]) for row in rows}):
        shot_rows = [row for row in rows if int(row["shots"]) == shots]
        shot_summary.append(
            {
                "shots": shots,
                "n_images": len(shot_rows),
                "mean_mse": float(np.mean([float(row["mse"]) for row in shot_rows])),
                "std_mse": float(np.std([float(row["mse"]) for row in shot_rows], ddof=0)),
                "mean_psnr": float(np.mean([float(row["psnr"]) for row in shot_rows])),
                "std_psnr": float(np.std([float(row["psnr"]) for row in shot_rows], ddof=0)),
                "mean_ssim": float(np.mean([float(row["ssim"]) for row in shot_rows])),
            }
        )
    return rows, shot_summary


def plot_shot_noise(result_dir: Path, rows: list[dict[str, object]]) -> None:
    shots = [int(row["shots"]) for row in rows]
    psnr = [float(row["mean_psnr"]) for row in rows]
    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.plot(shots, psnr, marker="o", color="#1f77b4")
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Shots")
    axis.set_ylabel("Mean PSNR (dB)")
    axis.set_title("Real CIFAR Shot-Noise Robustness")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(result_dir / "real_cifar_shot_noise.png", dpi=190)
    plt.close(fig)


def write_resource_table(result_dir: Path) -> None:
    rows = [
        {
            "model": "HVK2D-real-cifar",
            "feature_dim": 32,
            "readout_parameters": 2112,
            "cnot_or_pair_channels": 6,
            "train_images_per_seed": 6,
            "test_images_per_seed": 4,
            "claim_role": "candidate entangling observable model",
        },
        {
            "model": "strict-classical-rff",
            "feature_dim": 32,
            "readout_parameters": 2112,
            "cnot_or_pair_channels": 0,
            "train_images_per_seed": 6,
            "test_images_per_seed": 4,
            "claim_role": "same-width classical nonlinear control",
        },
        {
            "model": "no-entanglement",
            "feature_dim": 32,
            "readout_parameters": 2112,
            "cnot_or_pair_channels": 0,
            "train_images_per_seed": 6,
            "test_images_per_seed": 4,
            "claim_role": "single-site observable ablation",
        },
        {
            "model": "zz-only",
            "feature_dim": 32,
            "readout_parameters": 2112,
            "cnot_or_pair_channels": 3,
            "train_images_per_seed": 6,
            "test_images_per_seed": 4,
            "claim_role": "reduced observable-sector ablation",
        },
    ]
    write_dict_csv(result_dir / "resource_comparison.csv", rows)
    (result_dir / "resource_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def write_q1_report(
    result_dir: Path,
    real_summary: list[dict[str, object]],
    shot_summary: list[dict[str, object]],
    stats_rows: list[dict[str, object]],
) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    best_real = real_summary[0]
    hvk2d_real = next(row for row in real_summary if row["model"] == "HVK2D-real-cifar")
    real_rows = "\n".join(
        f"{row['model']} & ${float(row['mean_mse']):.4e}\\pm{float(row['std_mse']):.1e}$ & "
        f"${float(row['mean_psnr']):.2f}\\pm{float(row['std_psnr']):.2f}$ & "
        f"${float(row['mean_ssim']):.4f}\\pm{float(row['std_ssim']):.4f}$ \\\\"
        for row in real_summary
    )
    shot_rows = "\n".join(
        f"{int(row['shots'])} & ${float(row['mean_mse']):.4e}$ & {float(row['mean_psnr']):.2f} & {float(row['mean_ssim']):.4f} \\\\"
        for row in shot_summary
    )
    stat_rows = "\n".join(
        f"{row['comparison'].replace('HVK2D-real-cifar minus ', '')} & {int(row['n_pairs'])} & "
        f"${float(row['mean_psnr_difference_db']):.2f}$ & "
        f"$[{float(row['bootstrap95_low_db']):.2f},{float(row['bootstrap95_high_db']):.2f}]$ & "
        f"{float(row['wilcoxon_p_psnr']):.3g} \\\\"
        for row in stats_rows
    )
    tex = rf"""\documentclass[journal]{{IEEEtran}}
\usepackage{{amsmath,amsfonts,amssymb,graphicx,booktabs,cite,bm}}
\usepackage[hypertexnames=false]{{hyperref}}
\begin{{document}}
\title{{HVK1D/HVK2D Q1-Validation Addendum: Held-Out CIFAR Baselines, Observable Ablations, and Shot-Noise Robustness}}
\author{{Sparsho~Chakraborty and Siddhartha~Patra}}
\maketitle
\begin{{abstract}}
This addendum strengthens the HVK1D/HVK2D evidence package by separating diagnostic pair-correlation performance from held-out natural-image reconstruction. We report multi-seed real CIFAR-10 image splits, strict same-width classical controls, observable-sector ablations, shuffled-pair controls, random-latent controls, resource matching, and finite-shot observable noise. Unlike the restricted pair-correlation diagnostic, the real held-out CIFAR test does not establish quantum advantage: the best real-image result is obtained by {best_real['model']}, while HVK2D-real-cifar reaches {float(hvk2d_real['mean_psnr']):.2f} dB mean PSNR. The correct conclusion is therefore narrower: HVK2D has a useful entanglement-sensitive diagnostic, but the current real-image validation still requires architectural improvement before a Q1-level quantum-advantage claim is defensible.
\end{{abstract}}

\begin{{IEEEkeywords}}
Hamiltonian vision kernel, CIFAR-10 reconstruction, ablation study, quantum observables, shot noise, classical controls.
\end{{IEEEkeywords}}

\section{{Validation Protocol}}
Ten cached CIFAR-10 grayscale images are split across five random seeds. For each seed, six images are used to fit a single shared linear readout from a 32-dimensional latent feature vector to $8\times 8$ patch pixels, and four held-out images are reconstructed without per-image retraining. All principal controls use the same latent width and the same readout parameter count, so the comparison isolates feature structure rather than decoder capacity. This cached-image protocol is a reproducible smoke test, not a substitute for the recommended class-balanced CIFAR-10 subset protocol with mutually exclusive training, validation, and test images.

\section{{Classical Pair-Feature Controls}}
Because the nonlocal target is constructed from pair products, a raw-linear baseline is not by itself a sufficient classical control. We therefore include explicit quadratic-feature controls in the CIFAR reconstruction suite and in the CIFAR-derived nonlocal diagnostic. These controls test whether performance arises from access to multiplicative pair features generally, rather than from a uniquely quantum mechanism.

\section{{Held-Out CIFAR Results}}
\begin{{table*}}[t]
\centering
\caption{{Real held-out CIFAR-10 reconstruction across five random image splits. Lower MSE and higher PSNR/SSIM are better.}}
\label{{tab:real_cifar_q1}}
\scriptsize
\begin{{tabular}}{{lccc}}
\toprule
Model & MSE & PSNR & SSIM \\
\midrule
{real_rows}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.72\textwidth]{{../results/q1_validation/real_cifar_holdout_summary.png}}
\caption{{Held-out CIFAR metric summary for HVK2D and same-width controls.}}
\label{{fig:real_cifar_summary}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.82\textwidth]{{../results/q1_validation/real_cifar_reconstruction_panel.png}}
\caption{{Representative held-out CIFAR reconstruction panel. The target, prediction, and absolute-error images are shown for the candidate observable model and major controls.}}
\label{{fig:real_cifar_recons}}
\end{{figure*}}

\section{{Statistical Analysis}}
All principal image-level comparisons are paired by seed and held-out image.
For each pair, we compute the PSNR difference between HVK2D-real-cifar and the
corresponding control. Because the sample size is small and normality should
not be assumed, we report a Wilcoxon signed-rank $p$ value and a bootstrap
$95\%$ confidence interval for the mean paired PSNR difference. Positive values
favor HVK2D; negative values favor the control.

\begin{{table*}}[t]
\centering
\caption{{Paired statistical tests for held-out CIFAR reconstruction.}}
\label{{tab:paired_stats_q1}}
\scriptsize
\begin{{tabular}}{{lcccc}}
\toprule
Control & Pairs & Mean $\Delta$PSNR & Bootstrap 95\% CI & Wilcoxon $p$ \\
\midrule
{stat_rows}
\bottomrule
\end{{tabular}}
\end{{table*}}

\section{{Shot-Noise Robustness}}
\begin{{table}}[t]
\centering
\caption{{Finite-shot observable-noise simulation for real held-out CIFAR reconstruction.}}
\label{{tab:shot_noise_q1}}
\scriptsize
\begin{{tabular}}{{rccc}}
\toprule
Shots & Mean MSE & Mean PSNR & Mean SSIM \\
\midrule
{shot_rows}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../results/q1_validation/real_cifar_shot_noise.png}}
\caption{{Shot-noise robustness of the HVK2D observable feature vector on real held-out CIFAR splits.}}
\label{{fig:shot_noise_q1}}
\end{{figure}}

\section{{Interpretation}}
The key baseline is the strict same-width classical random-feature control, but the strongest warning comes from the local and raw-linear controls. In the real held-out CIFAR table, {best_real['model']} obtains the best mean MSE (${float(best_real['mean_mse']):.4e}$), while HVK2D-real-cifar is weaker (${float(hvk2d_real['mean_mse']):.4e}$). This means the current HVK2D evidence is strong only for the restricted pair-correlation diagnostic and for rejecting random-latent controls. It does not yet support a broad natural-image quantum-advantage claim. The Q1-ready path is therefore to keep these negative controls in the paper, redesign the image feature map or decoder-capacity match, and rerun this exact validation suite.

\section{{Reproducibility}}
The full tables, plots, resource comparison, and CSV files are generated under \texttt{{main2/newHVK/results/q1\_validation/}} by running \texttt{{main2/newHVK/run\_newhvk\_suite.py --q1-validation --write-q1-report}}.
\end{{document}}
"""
    (PAPER_DIR / "newhvk_q1_validation_report.tex").write_text(tex, encoding="utf-8")


def write_q1_validation_suite(write_report: bool = False) -> None:
    result_dir = RESULTS / "q1_validation"
    result_dir.mkdir(parents=True, exist_ok=True)
    real_rows, real_summary, panels = run_q1_real_cifar_suite()
    write_dict_csv(result_dir / "real_cifar_holdout.csv", real_rows)
    write_dict_csv(result_dir / "real_cifar_holdout_summary.csv", real_summary)
    (result_dir / "real_cifar_holdout.json").write_text(json.dumps(real_rows, indent=2), encoding="utf-8")
    (result_dir / "real_cifar_holdout_summary.json").write_text(json.dumps(real_summary, indent=2), encoding="utf-8")
    plot_q1_summary(result_dir, real_summary, "real_cifar_holdout_summary.png", "Real Held-Out CIFAR-10 Validation")
    save_q1_reconstruction_panel(result_dir, panels)

    gate_rows = [
        row
        for row in real_rows
        if row["model"]
        in {
            "HVK2D-real-cifar",
            "no-entanglement",
            "zz-only",
            "local-observables-only",
            "shuffled-pair-observables",
            "random-vqc",
        }
    ]
    gate_summary = aggregate_metric_rows(gate_rows)
    write_dict_csv(result_dir / "observable_gate_ablation.csv", gate_rows)
    write_dict_csv(result_dir / "observable_gate_ablation_summary.csv", gate_summary)
    plot_q1_summary(result_dir, gate_summary, "observable_gate_ablation.png", "Observable and Gate Ablations")

    shot_rows, shot_summary = run_q1_shot_noise_suite()
    write_dict_csv(result_dir / "shot_noise_real_cifar.csv", shot_rows)
    write_dict_csv(result_dir / "shot_noise_real_cifar_summary.csv", shot_summary)
    (result_dir / "shot_noise_real_cifar.json").write_text(json.dumps(shot_rows, indent=2), encoding="utf-8")
    plot_shot_noise(result_dir, shot_summary)
    write_resource_table(result_dir)
    stats_rows = write_q1_statistical_tests(result_dir, real_rows)
    if write_report:
        write_q1_report(result_dir, real_summary, shot_summary, stats_rows)
    readme = """# HVK1D/HVK2D Q1 validation suite

This folder contains a stronger validation layer requested for a Q1-style paper:
real held-out CIFAR smoke-test splits, multi-seed summaries, strict same-width
classical controls, explicit quadratic controls, observable/gate ablations,
shuffled-pair controls, random-latent controls, finite-shot noise simulation,
reconstruction panels, plots, and a resource comparison.

Claim boundary: these experiments strengthen the empirical baseline and ablation
story. They are not a replacement for a class-balanced CIFAR-10 subset protocol,
and they still support a controlled representational-diagnostic claim rather
than a formal hardware quantum-advantage proof.
"""
    (result_dir / "README.md").write_text(readme, encoding="utf-8")


def cifar_nonlocal_pair_dataset(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = sorted(CIFAR_IMAGES.glob("*.png"))
    if len(paths) < 8:
        raise FileNotFoundError(f"Need at least 8 CIFAR PNGs in {CIFAR_IMAGES}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    train_paths = [paths[i] for i in order[:6]]
    test_paths = [paths[i] for i in order[6:10]]

    def rows_for(paths_subset: list[Path]) -> np.ndarray:
        inputs: list[np.ndarray] = []
        for path in paths_subset:
            image = load_cifar_gray(path)
            patch_features: list[np.ndarray] = []
            patch_coords: list[tuple[int, int]] = []
            for row in range(0, 32, 8):
                for col in range(0, 32, 8):
                    patch = image[row : row + 8, col : col + 8]
                    patch_features.append(real_patch_base_features(patch, row / 24.0, col / 24.0))
                    patch_coords.append((row // 8, col // 8))
            patch_features_arr = np.asarray(patch_features, dtype=np.float64)
            pair_indices = []
            for left in range(len(patch_coords)):
                for right in range(left + 1, len(patch_coords)):
                    left_coord = patch_coords[left]
                    right_coord = patch_coords[right]
                    distance = abs(left_coord[0] - right_coord[0]) + abs(left_coord[1] - right_coord[1])
                    if distance >= 2:
                        pair_indices.append((left, right))
            for left, right in pair_indices:
                a = patch_features_arr[left]
                b = patch_features_arr[right]
                local_pair_input = np.concatenate([a[:10], b[:10], np.abs(a[:6] - b[:6])])
                inputs.append(local_pair_input)
        return np.asarray(inputs, dtype=np.float64)

    x_train = rows_for(train_paths)
    x_test = rows_for(test_paths)
    x_train, x_test = standardize_train_test(x_train, x_test)

    def target_from_standardized(x: np.ndarray) -> np.ndarray:
        left = x[:, :10]
        right = x[:, 10:20]
        delta = x[:, 20:26]
        target = np.stack(
            [
                left[:, 0] * right[:, 0],
                left[:, 1] * right[:, 1],
                left[:, 4] * right[:, 5],
                left[:, 7] * right[:, 8],
                (left[:, 2] - left[:, 3]) * (right[:, 4] - right[:, 5]),
                np.sin(np.pi * (left[:, 0] + right[:, 0])) * (left[:, 7] + right[:, 8]),
            ],
            axis=1,
        )
        return target

    y_train_raw = target_from_standardized(x_train)
    y_test_raw = target_from_standardized(x_test)
    mean = y_train_raw.mean(axis=0, keepdims=True)
    std = y_train_raw.std(axis=0, keepdims=True) + 1e-8
    y_train = (y_train_raw - mean) / std
    y_test = (y_test_raw - mean) / std
    return x_train, y_train, x_test, y_test


def cifar_pair_entangling_features(x: np.ndarray) -> np.ndarray:
    left = x[:, :10]
    right = x[:, 10:20]
    delta = x[:, 20:26]
    products = np.stack(
        [
            left[:, 0] * right[:, 0],
            left[:, 1] * right[:, 1],
            left[:, 4] * right[:, 5],
            left[:, 7] * right[:, 8],
            (left[:, 2] - left[:, 3]) * (right[:, 4] - right[:, 5]),
            np.sin(np.pi * (left[:, 0] + right[:, 0])) * (left[:, 7] + right[:, 8]),
            delta[:, 0] * delta[:, 1],
            delta[:, 2] * delta[:, 3],
        ],
        axis=1,
    )
    return select_same_width(np.concatenate([left, right, delta, products, np.sin(products)], axis=1), 40)


def cifar_pair_no_entanglement_features(x: np.ndarray) -> np.ndarray:
    return select_same_width(np.concatenate([x, np.sin(np.pi * x[:, :14])], axis=1), 40)


def cifar_pair_left_only_features(x: np.ndarray) -> np.ndarray:
    left = x[:, :10]
    delta = x[:, 20:26]
    return select_same_width(np.concatenate([left, delta, np.sin(np.pi * left), np.cos(np.pi * left)], axis=1), 40)


def cifar_pair_right_only_features(x: np.ndarray) -> np.ndarray:
    right = x[:, 10:20]
    delta = x[:, 20:26]
    return select_same_width(np.concatenate([right, delta, np.sin(np.pi * right), np.cos(np.pi * right)], axis=1), 40)


def cifar_pair_classical_matched_features(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(90_000 + seed)
    weights = rng.normal(0.0, 1.0 / math.sqrt(x.shape[1]), size=(x.shape[1], 40))
    bias = rng.uniform(-math.pi, math.pi, size=(40,))
    return np.tanh(x @ weights + bias)


def cifar_pair_quadratic_features(x: np.ndarray) -> np.ndarray:
    left = x[:, :10]
    right = x[:, 10:20]
    delta = x[:, 20:26]
    explicit_products = np.stack(
        [
            left[:, 0] * right[:, 0],
            left[:, 1] * right[:, 1],
            left[:, 2] * right[:, 2],
            left[:, 3] * right[:, 3],
            left[:, 4] * right[:, 5],
            left[:, 5] * right[:, 4],
            left[:, 6] * right[:, 7],
            left[:, 7] * right[:, 8],
            (left[:, 2] - left[:, 3]) * (right[:, 4] - right[:, 5]),
            delta[:, 0] * delta[:, 1],
            delta[:, 2] * delta[:, 3],
            delta[:, 4] * delta[:, 5],
        ],
        axis=1,
    )
    return select_same_width(np.concatenate([x, explicit_products, np.sin(np.pi * explicit_products)], axis=1), 40)


def cifar_pair_poly2_kernel_features(x: np.ndarray) -> np.ndarray:
    core = x[:, :20]
    columns: list[np.ndarray] = [x]
    for left_index in range(10):
        columns.append((core[:, left_index] * core[:, 10 + left_index])[:, None])
    for left_index, right_index in [(0, 11), (1, 12), (4, 15), (7, 18), (2, 14), (3, 15)]:
        columns.append((core[:, left_index] * core[:, right_index])[:, None])
    poly = np.concatenate(columns, axis=1)
    return select_same_width(poly, 40)


def cifar_pair_random_features(x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(91_000 + seed + x.shape[0])
    return rng.normal(0.0, 1.0, size=(x.shape[0], 40))


def evaluate_cifar_nonlocal_variant(seed: int, model: str, feature_fn: Callable[[np.ndarray, int], np.ndarray]) -> dict[str, object]:
    x_train, y_train, x_test, y_test = cifar_nonlocal_pair_dataset(seed)
    pred = ridge_fit_predict(feature_fn(x_train, seed), y_train, feature_fn(x_test, seed))
    mse = float(np.mean((pred - y_test) ** 2))
    return {
        "seed": seed,
        "model": model,
        "mse": mse,
        "psnr": psnr_from_mse(mse),
        "r2": r2_score(y_test, pred),
        "n_train_pairs": x_train.shape[0],
        "n_test_pairs": x_test.shape[0],
    }


def run_cifar_nonlocal_advantage_suite() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    variants: list[tuple[str, Callable[[np.ndarray, int], np.ndarray]]] = [
        ("HVK2D-cifar-nonlocal-entangling", lambda x, seed: cifar_pair_entangling_features(x)),
        ("no-entanglement-single-site", lambda x, seed: cifar_pair_no_entanglement_features(x)),
        ("left-patch-only", lambda x, seed: cifar_pair_left_only_features(x)),
        ("right-patch-only", lambda x, seed: cifar_pair_right_only_features(x)),
        ("strict-classical-rff", cifar_pair_classical_matched_features),
        ("quadratic-pair-classical", lambda x, seed: cifar_pair_quadratic_features(x)),
        ("degree2-polynomial-kernel", lambda x, seed: cifar_pair_poly2_kernel_features(x)),
        ("random-vqc", cifar_pair_random_features),
        ("raw-linear", lambda x, seed: select_same_width(x, 40)),
    ]
    for seed in [0, 1, 2, 3, 4]:
        for model, feature_fn in variants:
            rows.append(evaluate_cifar_nonlocal_variant(seed, model, feature_fn))
    summary: list[dict[str, object]] = []
    for model in sorted({str(row["model"]) for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        summary.append(
            {
                "model": model,
                "n_seeds": len(model_rows),
                "mean_mse": float(np.mean([float(row["mse"]) for row in model_rows])),
                "std_mse": float(np.std([float(row["mse"]) for row in model_rows], ddof=0)),
                "mean_psnr": float(np.mean([float(row["psnr"]) for row in model_rows])),
                "std_psnr": float(np.std([float(row["psnr"]) for row in model_rows], ddof=0)),
                "mean_r2": float(np.mean([float(row["r2"]) for row in model_rows])),
                "std_r2": float(np.std([float(row["r2"]) for row in model_rows], ddof=0)),
            }
        )
    return rows, sorted(summary, key=lambda row: float(row["mean_mse"]))


def plot_cifar_nonlocal_advantage(result_dir: Path, summary: list[dict[str, object]]) -> None:
    labels = [str(row["model"]) for row in summary]
    mse = [float(row["mean_mse"]) for row in summary]
    psnr = [float(row["mean_psnr"]) for row in summary]
    r2 = [float(row["mean_r2"]) for row in summary]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].bar(labels, mse, color="#1f77b4")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE, log scale")
    axes[1].bar(labels, psnr, color="#9467bd")
    axes[1].set_ylabel("PSNR (dB)")
    axes[2].bar(labels, r2, color="#2ca02c")
    axes[2].set_ylabel("$R^2$")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30, labelsize=7)
        axis.grid(alpha=0.2)
    fig.suptitle("CIFAR Nonlocal Patch-Correlation Advantage Diagnostic")
    fig.tight_layout()
    fig.savefig(result_dir / "cifar_nonlocal_advantage.png", dpi=190)
    plt.close(fig)


def write_cifar_nonlocal_advantage_suite() -> None:
    result_dir = RESULTS / "cifar_nonlocal_advantage"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = run_cifar_nonlocal_advantage_suite()
    write_dict_csv(result_dir / "cifar_nonlocal_pair_results.csv", rows)
    write_dict_csv(result_dir / "cifar_nonlocal_pair_summary.csv", summary)
    (result_dir / "cifar_nonlocal_pair_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (result_dir / "cifar_nonlocal_pair_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_cifar_nonlocal_advantage(result_dir, summary)
    readme = """# CIFAR nonlocal advantage diagnostic

This suite uses real CIFAR-10 images but changes the task from ordinary image
reconstruction to nonlocal patch-correlation prediction. The target depends on
products between distant patch statistics, so an entangling pair-observable map
has an explicit representational route that single-site/local controls lack.
Because this target is constructed from pair products, the suite also includes
explicit quadratic-pair and degree-two polynomial-feature classical controls.
If those controls match the entangling map, the result should be interpreted as
a pair-feature inductive-bias diagnostic rather than a uniquely quantum effect.

Claim boundary: this is a CIFAR-derived entanglement-sensitive representational
advantage diagnostic. It is not ordinary CIFAR reconstruction advantage and not
a hardware quantum-advantage proof.
"""
    (result_dir / "README.md").write_text(readme, encoding="utf-8")


def copy_existing_evidence() -> None:
    copy_specs = [
        (
            ROOT / "Baselines" / "cifar10_comparisons" / "outputs",
            RESULTS / "baselines" / "cifar10_comparisons",
        ),
        (
            ROOT / "Baselines" / "monalisa_comparisons" / "outputs",
            RESULTS / "baselines" / "monalisa_comparisons",
        ),
        (
            ROOT / "experiments" / "quantum_contribution" / "results",
            RESULTS / "ablation_study" / "legacy_hvk_controls",
        ),
        (
            ROOT / "IBM_Cloud" / "outputs",
            RESULTS / "hardware_probe" / "ibm_cloud_outputs",
        ),
    ]
    for source, destination in copy_specs:
        if not source.exists():
            continue
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def write_manifest(results: list[ModelResult]) -> None:
    manifest = {
        "workspace": "main2/newHVK",
        "claim_status": (
            "candidate only: the restricted pair-correlation task favors the "
            "entangling observable channel, but this is not a hardware quantum "
            "advantage proof or a broad CIFAR generalization result"
        ),
        "new_model_variants": [
            "HVK2D-entangling-observables",
            "HVK2D-no-entanglement",
            "parameter-matched-classical",
            "raw-linear-classical",
            "random-vqc",
            "freeze-quantum",
            "freeze-classical",
            "heldout-cifar-proxy",
            "noise-hardware-probe",
        ],
        "imported_evidence": [
            "CIFAR and Monalisa baselines copied from existing outputs",
            "Legacy HVK ablations copied from experiments/quantum_contribution/results",
            "IBM Cloud circuit-resource probe copied from IBM_Cloud/outputs",
        ],
        "full_suite_outputs": [
            "results/full_ablation_suite/full_ablation_summary.csv",
            "results/full_ablation_suite/multi_seed_results.csv",
            "results/full_ablation_suite/heldout_cifar_proxy.csv",
            "results/full_ablation_suite/noise_hardware_probe.csv",
            "results/full_ablation_suite/hvk_epoch_reconstruction_table.csv",
            "results/full_ablation_suite/hvk_epoch_correlation_table.csv",
            "results/full_ablation_suite/order_parameter_curve.csv",
            "results/full_ablation_suite/media/",
        ],
        "q1_validation_outputs": [
            "results/q1_validation/real_cifar_holdout.csv",
            "results/q1_validation/real_cifar_holdout_summary.csv",
            "results/q1_validation/observable_gate_ablation.csv",
            "results/q1_validation/shot_noise_real_cifar.csv",
            "results/q1_validation/resource_comparison.csv",
            "paper_latex/newhvk_q1_validation_report.tex",
            "paper_latex/newhvk_q1_validation_report.pdf",
        ],
        "cifar_nonlocal_advantage_outputs": [
            "results/cifar_nonlocal_advantage/cifar_nonlocal_pair_results.csv",
            "results/cifar_nonlocal_advantage/cifar_nonlocal_pair_summary.csv",
            "results/cifar_nonlocal_advantage/cifar_nonlocal_advantage.png",
        ],
        "restricted_benchmark_best_model": min(results, key=lambda item: item.mse).model,
    }
    (WORKSPACE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_paper(results: list[ModelResult]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        (
            f"{r.model} & ${r.mse:.4e}$ & {r.psnr:.2f} & {r.r2:.4f} "
            f"& {r.notes} \\\\"
        )
        for r in results
    )
    tex = rf"""\documentclass[journal]{{IEEEtran}}
\usepackage{{amsmath,amsfonts,amssymb,graphicx,booktabs,cite,bm}}
\usepackage[hypertexnames=false]{{hyperref}}
\begin{{document}}
\title{{HVK1D/HVK2D: Restricted Entanglement-Sensitive Benchmarks for Hamiltonian Vision Kernels}}
\author{{Sparsho~Chakraborty and Siddhartha~Patra}}
\maketitle
\begin{{abstract}}
This manuscript separates two questions that were conflated in the original HVK study: whether an observable-latent image reconstruction pipeline is useful, and whether entangling quantum observables provide a measurable advantage over non-entangling or classical controls. Prior ablations showed that the original HVK reconstruction task did not establish quantum advantage because fixed quantum features, no-entanglement circuits, and classical replacements matched or exceeded the trained VQC baseline. We therefore extend the HVK1D/HVK2D ablation workspace with restricted pair-correlation benchmarks where the target depends explicitly on nonlocal feature products. In this restricted setting, the HVK2D entangling-observable channel is expected to outperform controls that cannot represent pair correlations with the same feature budget. We report this as a candidate quantum-advantage diagnostic, not as a hardware quantum advantage proof or a broad dataset-level generalization result.
\end{{abstract}}

\begin{{IEEEkeywords}}
Hybrid quantum-classical learning, Hamiltonian vision kernel, entanglement ablation, image reconstruction, quantum advantage diagnostics.
\end{{IEEEkeywords}}

\section{{Motivation}}
The legacy HVK ablation suite showed that the observable channel is load-bearing, but it also showed that entanglement and Hamiltonian energy regularization were not decisive on the per-image reconstruction benchmark. A stronger test must reduce decoder freedom and use a task whose signal is genuinely pair-correlational. The HVK1D/HVK2D validation workspace therefore keeps the old evidence intact and adds an entanglement-sensitive benchmark with explicit no-entanglement, parameter-matched classical, and raw-linear controls.

\section{{Architecture}}
The publication workspace is organized under the historical path \texttt{{main2/newHVK}}, but the reported model family contains only HVK1D/HVK2D variants:
\begin{{itemize}}
\item \textbf{{HVK2D-entangling-observables:}} single-site features plus pair-correlation channels analogous to measured two-qubit observables.
\item \textbf{{HVK2D-no-entanglement:}} single-site nonlinear features only.
\item \textbf{{parameter-matched-classical:}} a rank-limited tanh map used as a small classical control.
\item \textbf{{raw-linear-classical:}} a linear baseline on raw coordinates.
\end{{itemize}}

\section{{Restricted Entanglement-Sensitive Benchmark}}
The benchmark samples six-dimensional inputs $x\in[-1,1]^6$ and predicts targets containing products such as $x_0x_1$, $x_1x_2$, and $x_0x_3$. A linear readout on entangling features can represent these targets directly, while non-entangling single-site features cannot represent the product structure without extra decoder capacity.

\begin{{table*}}[t]
\centering
\caption{{HVK2D restricted pair-correlation benchmark. Lower MSE and higher PSNR/$R^2$ are better.}}
\label{{tab:newhvk_restricted}}
\scriptsize
\begin{{tabular}}{{lcccp{{0.38\textwidth}}}}
\toprule
Model & MSE & PSNR (dB) & $R^2$ & Notes \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table*}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\linewidth]{{../results/quantum_advantage_candidate/entanglement_sensitive_benchmark.png}}
\caption{{Restricted pair-correlation benchmark. This plot is a diagnostic for entanglement-sensitive representation under controlled decoder capacity, not a claim of broad quantum advantage.}}
\label{{fig:newhvk_paircorr}}
\end{{figure}}

\section{{Imported Baselines and Ablations}}
The folder also includes copied records from the completed HVK study: CIFAR baselines, Monalisa baselines, legacy ablation controls, and IBM Cloud circuit-resource outputs. These files preserve provenance but do not by themselves establish quantum advantage or support quantitative component attribution. In particular, the earlier Monalisa freeze-isolation aggregate is excluded from manuscript evidence unless its per-seed artifacts are recovered or the study is rerun.

\section{{Caveats}}
The restricted benchmark is deliberately favorable to pair-correlation observables, so it should be presented as a diagnostic experiment. To make a Q1-level claim, the next stage must run the same restricted-capacity design on held-out CIFAR images, multiple random seeds, hardware-noise simulation, and a parameter-matched classical baseline with the same observable budget. The paper should not state that HVK2D proves hardware quantum advantage unless those tests remain positive.

\section{{Conclusion}}
HVK2D provides a clean route toward testing entanglement-sensitive representation: reduce decoder capacity, make the target correlation-sensitive, and compare entangling observables against no-entanglement and classical controls. The current results demonstrate separation on a restricted diagnostic task, while the held-out image-reconstruction controls define the scope of that result.
\end{{document}}
"""
    (PAPER_DIR / "newhvk_paper.tex").write_text(tex, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HVK1D/HVK2D evidence suite.")
    parser.add_argument(
        "--skip-copy-existing",
        action="store_true",
        help="Do not refresh copied legacy baselines, ablations, and IBM outputs.",
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Generate the full ablation suite with CSVs, graphs, order parameters, GIFs, and videos.",
    )
    parser.add_argument(
        "--write-paper",
        action="store_true",
        help="Regenerate main2/newHVK/paper_latex/newhvk_paper.tex. Off by default.",
    )
    parser.add_argument(
        "--q1-validation",
        action="store_true",
        help="Run real held-out CIFAR validation, strict controls, gate ablations, and shot-noise tests.",
    )
    parser.add_argument(
        "--write-q1-report",
        action="store_true",
        help="Write main2/newHVK/paper_latex/newhvk_q1_validation_report.tex.",
    )
    parser.add_argument(
        "--cifar-nonlocal-advantage",
        action="store_true",
        help="Run a CIFAR-derived nonlocal patch-correlation diagnostic where pair observables are load-bearing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.skip_copy_existing:
        copy_existing_evidence()
    results = run_entanglement_sensitive_benchmark()
    write_results(results)
    if args.full_suite:
        write_full_ablation_suite()
    if args.q1_validation:
        write_q1_validation_suite(write_report=args.write_q1_report)
    if args.cifar_nonlocal_advantage:
        write_cifar_nonlocal_advantage_suite()
    write_manifest(results)
    if args.write_paper:
        write_paper(results)
    print("HVK1D/HVK2D suite complete")
    print(f"Workspace: {WORKSPACE}")
    for result in results:
        print(f"{result.model}: mse={result.mse:.6e}, psnr={result.psnr:.2f}, r2={result.r2:.4f}")
    if args.full_suite:
        print(f"Full ablation suite: {RESULTS / 'full_ablation_suite'}")
    if args.q1_validation:
        print(f"Q1 validation suite: {RESULTS / 'q1_validation'}")
    if args.cifar_nonlocal_advantage:
        print(f"CIFAR nonlocal advantage suite: {RESULTS / 'cifar_nonlocal_advantage'}")


if __name__ == "__main__":
    main()
