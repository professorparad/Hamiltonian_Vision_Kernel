from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Statevector
from run_ibm_hvk_probe import (
    OUTPUT_DIR,
    build_hvk_circuit,
    chain_edges,
    counts_from_sampler_result,
    grid_edges,
    order_from_counts,
    run_on_ibm,
)


def order_from_probabilities(probabilities: dict[str, float], edges: list[tuple[int, int]], n_qubits: int) -> dict:
    z_values = []
    abs_orders = []
    corr_values = []
    for bitstring, probability in probabilities.items():
        bits = bitstring.replace(" ", "")[::-1]
        z = np.array([1.0 if bit == "0" else -1.0 for bit in bits[:n_qubits]])
        order = float(z.mean())
        z_values.append(order * probability)
        abs_orders.append(abs(order) * probability)
        corr_values.append(float(np.mean([z[u] * z[v] for u, v in edges])) * probability)
    mean_abs = float(sum(abs_orders))
    mean_corr = float(sum(corr_values))
    return {
        "mean_order_parameter": float(sum(z_values)),
        "mean_abs_order_parameter": mean_abs,
        "mean_zz_correlation": mean_corr,
        "hardware_proxy_loss": float((1.0 - mean_abs) + 0.5 * (1.0 - mean_corr)),
    }


def exact_statevector_metrics(circuit, edges: list[tuple[int, int]], n_qubits: int) -> dict:
    unitary_circuit = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(unitary_circuit)
    return order_from_probabilities(state.probabilities_dict(), edges, n_qubits)


def metric_delta(sim: dict, hardware: dict) -> dict:
    keys = [
        "mean_order_parameter",
        "mean_abs_order_parameter",
        "mean_zz_correlation",
        "hardware_proxy_loss",
    ]
    return {f"delta_{key}": float(hardware[key] - sim[key]) for key in keys}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_validation_plot(rows: list[dict], output_path: Path) -> None:
    names = [f"{row['variant']}-{row['patch_index']}" for row in rows]
    metrics = [
        ("mean_abs_order_parameter", "Abs order"),
        ("mean_zz_correlation", "ZZ correlation"),
        ("hardware_proxy_loss", "Proxy loss"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, (key, title) in zip(axes, metrics):
        sim_values = [row[f"sim_{key}"] for row in rows]
        hw_values = [row.get(f"hardware_{key}", np.nan) for row in rows]
        x = np.arange(len(names))
        width = 0.36
        ax.bar(x - width / 2, sim_values, width, label="statevector")
        if any(np.isfinite(hw_values)):
            ax.bar(x + width / 2, hw_values, width, label="hardware")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare exact simulator HVK latent observables with IBM hardware observables "
            "for the same image-patch circuits."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz",
    )
    parser.add_argument("--variant", choices=["hvk1d", "hvk2d", "both"], default="both")
    parser.add_argument("--backend")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--max-patches", type=int, default=1)
    parser.add_argument("--token", default=os.environ.get("IBM_QUANTUM_TOKEN"))
    parser.add_argument("--dry-run", action="store_true", help="Only compute exact simulator latents.")
    parser.add_argument("--allow-large-heron-job", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_qubits < 2:
        raise ValueError("--n-qubits must be at least 2.")
    if args.max_patches < 1:
        raise ValueError("--max-patches must be at least 1.")

    data = np.load(args.dataset, allow_pickle=False)
    vectors = data["patch_vectors"][: args.max_patches]
    variants = ["hvk1d", "hvk2d"] if args.variant == "both" else [args.variant]

    circuits = []
    labels = []
    sim_rows = []
    for variant in variants:
        edges = chain_edges(args.n_qubits) if variant == "hvk1d" else grid_edges(args.n_qubits)
        for patch_index, vector in enumerate(vectors):
            circuit = build_hvk_circuit(vector, variant, args.n_qubits)
            sim_metrics = exact_statevector_metrics(circuit, edges, args.n_qubits)
            circuits.append(circuit)
            labels.append({"variant": variant, "patch_index": patch_index})
            sim_rows.append(
                {
                    "variant": variant,
                    "patch_index": patch_index,
                    **{f"sim_{key}": value for key, value in sim_metrics.items()},
                }
            )

    if len(circuits) > 20 and not args.allow_large_heron_job:
        raise ValueError(
            f"This would submit {len(circuits)} circuits. "
            "Use fewer patches/variants or pass --allow-large-heron-job if the quota is acceptable."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = sim_rows
    backend = None
    job_id = None
    if not args.dry_run:
        if not args.token:
            raise ValueError("No IBM token found. Set IBM_QUANTUM_TOKEN before submitting to hardware.")
        backend, job_id, result = run_on_ibm(circuits, args.backend, args.shots, args.token, args.n_qubits)
        merged_rows = []
        for index, label in enumerate(labels):
            variant = label["variant"]
            edges = chain_edges(args.n_qubits) if variant == "hvk1d" else grid_edges(args.n_qubits)
            counts = counts_from_sampler_result(result, index)
            hardware_metrics = order_from_counts(counts, edges, args.n_qubits)
            sim_metrics = {
                key.removeprefix("sim_"): value
                for key, value in sim_rows[index].items()
                if key.startswith("sim_")
            }
            merged_rows.append(
                {
                    **sim_rows[index],
                    "backend": backend,
                    "job_id": job_id,
                    **{f"hardware_{key}": value for key, value in hardware_metrics.items()},
                    **metric_delta(sim_metrics, hardware_metrics),
                }
            )
        rows = merged_rows

    json_path = args.output_dir / "hardware_latent_validation.json"
    csv_path = args.output_dir / "hardware_latent_validation.csv"
    plot_path = args.output_dir / "hardware_latent_validation.png"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    save_validation_plot(rows, plot_path)

    if args.dry_run:
        print(f"Computed exact simulator latents for {len(rows)} circuits.")
    else:
        print(f"IBM latent validation complete: backend={backend}, job_id={job_id}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
