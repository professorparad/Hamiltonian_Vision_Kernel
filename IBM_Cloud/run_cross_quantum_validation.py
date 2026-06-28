from __future__ import annotations

# ruff: noqa: E402,I001

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

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


METRIC_KEYS = [
    "mean_order_parameter",
    "mean_abs_order_parameter",
    "mean_zz_correlation",
    "hardware_proxy_loss",
]


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


def extract_qiskit_counts(result: Any, index: int = 0) -> dict[str, int]:
    if hasattr(result, "get_counts"):
        return result.get_counts(index)
    if isinstance(result, dict):
        return result
    raise TypeError("Could not extract counts from provider result.")


def run_on_ionq(circuits, backend_name: str, shots: int):
    token = os.environ.get("IONQ_API_TOKEN")
    if not token:
        raise ValueError("Set IONQ_API_TOKEN before using --provider ionq.")
    try:
        from qiskit_ionq import IonQProvider
    except ImportError as exc:
        raise ImportError("Install IonQ support with: pip install qiskit-ionq") from exc

    provider = IonQProvider(token=token)
    backend = provider.get_backend(backend_name)
    job = backend.run(circuits, shots=shots)
    return backend_name, job.job_id(), job.result()


def run_on_braket(circuits, backend_name: str, shots: int):
    try:
        from qiskit_braket_provider import AWSBraketProvider
    except ImportError as exc:
        raise ImportError("Install Braket Qiskit support with: pip install qiskit-braket-provider") from exc

    provider = AWSBraketProvider()
    backend = provider.get_backend(backend_name)
    job = backend.run(circuits, shots=shots)
    return backend_name, job.job_id(), job.result()


def run_on_azure(circuits, backend_name: str, shots: int):
    resource_id = os.environ.get("AZURE_QUANTUM_RESOURCE_ID")
    location = os.environ.get("AZURE_QUANTUM_LOCATION")
    if not resource_id:
        raise ValueError("Set AZURE_QUANTUM_RESOURCE_ID before using --provider azure.")
    try:
        from azure.quantum.qiskit import AzureQuantumProvider
    except ImportError as exc:
        raise ImportError("Install Azure Quantum Qiskit support with: pip install 'azure-quantum[qiskit]'") from exc

    kwargs = {"resource_id": resource_id}
    if location:
        kwargs["location"] = location
    provider = AzureQuantumProvider(**kwargs)
    backend = provider.get_backend(backend_name)
    job = backend.run(circuits, shots=shots)
    return backend_name, job.id(), job.result()


def run_provider(provider: str, circuits, backend_name: str | None, shots: int, n_qubits: int):
    if provider == "ibm":
        return run_on_ibm(circuits, backend_name, shots, os.environ.get("IBM_QUANTUM_TOKEN"), n_qubits)
    if not backend_name:
        raise ValueError(f"--backend is required for provider {provider}.")
    if provider == "ionq":
        return run_on_ionq(circuits, backend_name, shots)
    if provider == "braket":
        return run_on_braket(circuits, backend_name, shots)
    if provider == "azure":
        return run_on_azure(circuits, backend_name, shots)
    raise ValueError(f"Unsupported provider: {provider}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows: list[dict], output_path: Path) -> None:
    providers = sorted({row["provider"] for row in rows})
    metrics = [
        ("mean_abs_order_parameter", "Abs order"),
        ("mean_zz_correlation", "ZZ correlation"),
        ("hardware_proxy_loss", "Proxy loss"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, (key, title) in zip(axes, metrics):
        labels = []
        values = []
        for provider in providers:
            provider_rows = [row for row in rows if row["provider"] == provider]
            labels.append(provider)
            values.append(float(np.mean([row[key] for row in provider_rows if key in row])))
        ax.bar(labels, values)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-provider HVK latent validation for IBM, IonQ, Braket, Azure, and local statevector."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz",
    )
    parser.add_argument("--provider", choices=["statevector", "ibm", "ionq", "braket", "azure"], default="statevector")
    parser.add_argument("--variant", choices=["hvk1d", "hvk2d", "both"], default="both")
    parser.add_argument("--backend")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--max-patches", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--allow-large-job", action="store_true")
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
    statevector_rows = []
    for variant in variants:
        edges = chain_edges(args.n_qubits) if variant == "hvk1d" else grid_edges(args.n_qubits)
        for patch_index, vector in enumerate(vectors):
            circuit = build_hvk_circuit(vector, variant, args.n_qubits)
            state_metrics = exact_statevector_metrics(circuit, edges, args.n_qubits)
            circuits.append(circuit)
            labels.append({"variant": variant, "patch_index": patch_index, "edges": edges})
            statevector_rows.append(
                {
                    "provider": "statevector",
                    "backend": "qiskit_statevector",
                    "variant": variant,
                    "patch_index": patch_index,
                    **state_metrics,
                }
            )

    if len(circuits) > 20 and args.provider != "statevector" and not args.allow_large_job:
        raise ValueError(
            f"This would submit {len(circuits)} circuits. "
            "Use fewer patches/variants or pass --allow-large-job if your quota allows it."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.provider == "statevector":
        rows = statevector_rows
        job_id = None
        backend = "qiskit_statevector"
    else:
        backend, job_id, result = run_provider(args.provider, circuits, args.backend, args.shots, args.n_qubits)
        rows = []
        for index, label in enumerate(labels):
            if args.provider == "ibm":
                counts = counts_from_sampler_result(result, index)
            else:
                counts = extract_qiskit_counts(result, index)
            provider_metrics = order_from_counts(counts, label["edges"], args.n_qubits)
            state_metrics = {key: statevector_rows[index][key] for key in METRIC_KEYS}
            rows.append(
                {
                    "provider": args.provider,
                    "backend": backend,
                    "job_id": job_id,
                    "variant": label["variant"],
                    "patch_index": label["patch_index"],
                    **provider_metrics,
                    **{f"statevector_{key}": value for key, value in state_metrics.items()},
                    **{f"delta_{key}": provider_metrics[key] - state_metrics[key] for key in METRIC_KEYS},
                }
            )

    stem = f"cross_quantum_validation_{args.provider}"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    plot_path = args.output_dir / f"{stem}.png"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(csv_path, rows)
    save_plot(rows, plot_path)

    print(f"Provider: {args.provider}")
    print(f"Backend: {backend}")
    if job_id:
        print(f"Job ID: {job_id}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
