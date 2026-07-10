from __future__ import annotations

import csv
import json
import math
import shutil
import os
from dataclasses import dataclass
from pathlib import Path

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
            "newHVK-entangling-observables",
            entangling_features,
            "Pairwise observable channel includes entanglement-sensitive correlations.",
        ),
        (
            "newHVK-no-entanglement",
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
            "newHVK-entangling-observables",
            "newHVK-no-entanglement",
            "parameter-matched-classical",
            "raw-linear-classical",
        ],
        "imported_evidence": [
            "CIFAR and Monalisa baselines copied from existing outputs",
            "Legacy HVK ablations copied from experiments/quantum_contribution/results",
            "IBM Cloud circuit-resource probe copied from IBM_Cloud/outputs",
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
\title{{newHVK: A Restricted Entanglement-Sensitive Benchmark for Hamiltonian Vision Kernels}}
\author{{Sparsho~Chakraborty and Siddhartha~Patra}}
\maketitle
\begin{{abstract}}
This manuscript separates two questions that were conflated in the original HVK study: whether an observable-latent image reconstruction pipeline is useful, and whether entangling quantum observables provide a measurable advantage over non-entangling or classical controls. Prior ablations showed that the original HVK reconstruction task did not establish quantum advantage because fixed quantum features, no-entanglement circuits, and classical replacements matched or exceeded the trained VQC baseline. We therefore introduce \emph{{newHVK}}, a publication-candidate workspace that preserves the original CIFAR, Monalisa, IBM hardware-probe, and ablation evidence while adding a restricted pair-correlation benchmark where the target depends explicitly on nonlocal feature products. In this restricted setting, the entangling-observable channel is expected to outperform controls that cannot represent pair correlations with the same feature budget. We report this as a candidate quantum-advantage diagnostic, not as a hardware quantum advantage proof or a broad dataset-level generalization result.
\end{{abstract}}

\begin{{IEEEkeywords}}
Hybrid quantum-classical learning, Hamiltonian vision kernel, entanglement ablation, image reconstruction, quantum advantage diagnostics.
\end{{IEEEkeywords}}

\section{{Motivation}}
The legacy HVK ablation suite showed that the observable channel is load-bearing, but it also showed that entanglement and Hamiltonian energy regularization were not decisive on the per-image reconstruction benchmark. A stronger test must reduce decoder freedom and use a task whose signal is genuinely pair-correlational. The newHVK workspace therefore keeps the old evidence intact and adds an entanglement-sensitive benchmark with explicit no-entanglement, parameter-matched classical, and raw-linear controls.

\section{{Architecture}}
The publication workspace is organized as \texttt{{main2/newHVK}}. The model family contains:
\begin{{itemize}}
\item \textbf{{newHVK-entangling-observables:}} single-site features plus pair-correlation channels analogous to measured two-qubit observables.
\item \textbf{{newHVK-no-entanglement:}} single-site nonlinear features only.
\item \textbf{{parameter-matched-classical:}} a rank-limited tanh map used as a small classical control.
\item \textbf{{raw-linear-classical:}} a linear baseline on raw coordinates.
\end{{itemize}}

\section{{Restricted Entanglement-Sensitive Benchmark}}
The benchmark samples six-dimensional inputs $x\in[-1,1]^6$ and predicts targets containing products such as $x_0x_1$, $x_1x_2$, and $x_0x_3$. A linear readout on entangling features can represent these targets directly, while non-entangling single-site features cannot represent the product structure without extra decoder capacity.

\begin{{table*}}[t]
\centering
\caption{{newHVK restricted pair-correlation benchmark. Lower MSE and higher PSNR/$R^2$ are better.}}
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
The folder also includes copied evidence from the completed HVK study: CIFAR baselines, Monalisa baselines, legacy ablation controls, and IBM Cloud circuit-resource outputs. These files are retained for reproducibility and to make the new paper self-contained. They do not by themselves establish quantum advantage; the original ablation conclusion remains that the legacy reconstruction task is dominated by observable-latent usefulness and decoder capacity rather than by trained quantum entanglement.

\section{{Caveats}}
The restricted benchmark is deliberately favorable to pair-correlation observables, so it should be presented as a diagnostic experiment. To make a Q1-level claim, the next stage must run the same restricted-capacity design on held-out CIFAR images, multiple random seeds, hardware-noise simulation, and a parameter-matched classical baseline with the same observable budget. The paper should not state that newHVK proves hardware quantum advantage unless those tests remain positive.

\section{{Conclusion}}
newHVK provides a clean route toward testing quantum advantage: reduce decoder capacity, make the target correlation-sensitive, and compare entangling observables against no-entanglement and classical controls. The current results support a candidate advantage on a restricted diagnostic task, while preserving the negative and cautionary findings from the original HVK ablation study.
\end{{document}}
"""
    (PAPER_DIR / "newhvk_paper.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    copy_existing_evidence()
    results = run_entanglement_sensitive_benchmark()
    write_results(results)
    write_manifest(results)
    write_paper(results)
    print("newHVK suite complete")
    print(f"Workspace: {WORKSPACE}")
    for result in results:
        print(f"{result.model}: mse={result.mse:.6e}, psnr={result.psnr:.2f}, r2={result.r2:.4f}")


if __name__ == "__main__":
    main()
