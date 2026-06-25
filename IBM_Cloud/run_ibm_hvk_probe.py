from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
FREE_PLAN_MAX_CIRCUITS = 2
FREE_PLAN_MAX_SHOTS = 100


def chain_edges(n_qubits: int) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(n_qubits - 1)]


def grid_edges(n_qubits: int) -> list[tuple[int, int]]:
    cols = int(np.ceil(np.sqrt(n_qubits)))
    edges = []
    for index in range(n_qubits):
        row, col = divmod(index, cols)
        right = index + 1
        down = index + cols
        if col + 1 < cols and right < n_qubits:
            edges.append((index, right))
        if down < n_qubits:
            edges.append((index, down))
    return edges


def angles_from_patch(vector: np.ndarray, n_qubits: int) -> np.ndarray:
    chunks = np.array_split(vector, n_qubits)
    means = np.array([chunk.mean() for chunk in chunks], dtype=np.float32)
    stds = np.array([chunk.std() for chunk in chunks], dtype=np.float32)
    return np.stack([means, stds], axis=1) * np.pi


def build_hvk_circuit(vector: np.ndarray, variant: str, n_qubits: int) -> QuantumCircuit:
    angles = angles_from_patch(vector, n_qubits)
    circuit = QuantumCircuit(n_qubits, n_qubits)
    for qubit in range(n_qubits):
        circuit.ry(float(angles[qubit, 0]), qubit)
        circuit.rz(float(angles[qubit, 1]), qubit)

    edges = chain_edges(n_qubits) if variant == "hvk1d" else grid_edges(n_qubits)
    for source, target in edges:
        circuit.cx(source, target)
        circuit.ry(0.15, target)
        circuit.cx(source, target)

    circuit.measure(range(n_qubits), range(n_qubits))
    return circuit


def order_from_counts(counts: dict[str, int], edges: list[tuple[int, int]], n_qubits: int) -> dict:
    shots = sum(counts.values())
    if shots == 0:
        raise ValueError("No measurement shots returned.")
    z_values = []
    abs_orders = []
    corr_values = []
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")[::-1]
        z = np.array([1.0 if bit == "0" else -1.0 for bit in bits[:n_qubits]])
        order = float(z.mean())
        z_values.append(order * count)
        abs_orders.append(abs(order) * count)
        corr_values.append(float(np.mean([z[u] * z[v] for u, v in edges])) * count)
    mean_abs = float(sum(abs_orders) / shots)
    mean_corr = float(sum(corr_values) / shots)
    return {
        "shots": shots,
        "mean_order_parameter": float(sum(z_values) / shots),
        "mean_abs_order_parameter": mean_abs,
        "mean_zz_correlation": mean_corr,
        "hardware_proxy_loss": float((1.0 - mean_abs) + 0.5 * (1.0 - mean_corr)),
    }


def counts_from_sampler_result(result, index: int) -> dict[str, int]:
    pub_result = result[index]
    data = getattr(pub_result, "data", pub_result)
    if hasattr(data, "meas"):
        return data.meas.get_counts()
    if hasattr(data, "c"):
        return data.c.get_counts()
    if hasattr(pub_result, "get_counts"):
        return pub_result.get_counts()
    raise TypeError("Could not extract counts from Sampler result. Check qiskit-ibm-runtime version.")


def get_service(token: str | None):
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise ImportError(
            "qiskit-ibm-runtime is required. Install with: pip install -r IBM_Cloud/requirements-ibm.txt"
        ) from exc
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    return QiskitRuntimeService()


def select_backend(service, backend_name: str | None, n_qubits: int):
    if backend_name:
        return service.backend(backend_name)
    backends = service.backends(operational=True, simulator=False, min_num_qubits=n_qubits)
    if not backends:
        raise RuntimeError(f"No operational IBM backend with at least {n_qubits} qubits is available for this account.")
    return sorted(backends, key=lambda backend: getattr(backend.status(), "pending_jobs", 999999))[0]


def run_on_ibm(circuits: list[QuantumCircuit], backend_name: str | None, shots: int, token: str | None, n_qubits: int):
    try:
        from qiskit_ibm_runtime import SamplerV2 as Sampler
    except ImportError:
        from qiskit_ibm_runtime import Sampler
    service = get_service(token)
    backend = select_backend(service, backend_name, n_qubits)
    transpiled = transpile(circuits, backend=backend, optimization_level=1)
    sampler = Sampler(mode=backend)
    job = sampler.run(transpiled, shots=shots)
    return backend.name, job.job_id(), job.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tiny HVK1D/HVK2D probe circuits on IBM Quantum.")
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "datasets" / "monalisa_patches.npz")
    parser.add_argument("--variant", choices=["hvk1d", "hvk2d", "both"], default="both")
    parser.add_argument("--backend")
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--shots", type=int, default=FREE_PLAN_MAX_SHOTS)
    parser.add_argument("--max-patches", type=int, default=1)
    parser.add_argument("--allow-large-free-plan-job", action="store_true")
    parser.add_argument("--token", default=os.environ.get("IBM_QUANTUM_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_qubits < 2:
        raise ValueError("--n-qubits must be at least 2.")
    data = np.load(args.dataset, allow_pickle=False)
    vectors = data["patch_vectors"][: args.max_patches]
    variants = ["hvk1d", "hvk2d"] if args.variant == "both" else [args.variant]
    circuits = []
    labels = []
    for variant in variants:
        for patch_index, vector in enumerate(vectors):
            circuits.append(build_hvk_circuit(vector, variant, args.n_qubits))
            labels.append({"variant": variant, "patch_index": patch_index})
    if not args.allow_large_free_plan_job:
        if len(circuits) > FREE_PLAN_MAX_CIRCUITS or args.shots > FREE_PLAN_MAX_SHOTS:
            raise ValueError(
                "This job is larger than the conservative IBM free-plan guard "
                f"({FREE_PLAN_MAX_CIRCUITS} circuits, {FREE_PLAN_MAX_SHOTS} shots). "
                "Reduce --max-patches/--shots, run one variant at a time, or pass "
                "--allow-large-free-plan-job if you know your runtime quota can handle it."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    circuit_qasm_path = args.output_dir / "circuits_summary.json"
    circuit_qasm_path.write_text(
        json.dumps(
            [
                {
                    **label,
                    "num_qubits": circuit.num_qubits,
                    "depth": circuit.depth(),
                    "num_ops": dict(circuit.count_ops()),
                }
                for label, circuit in zip(labels, circuits)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    save_circuit_visualization(circuits, labels, args.output_dir / "circuit_summary.png")
    if args.dry_run:
        print(f"Built {len(circuits)} circuits. Summary: {circuit_qasm_path}")
        return
    if not args.token:
        raise ValueError(
            "No IBM token found. Set IBM_QUANTUM_TOKEN in your shell or save credentials with "
            "QiskitRuntimeService.save_account(). Do not hard-code tokens in this repo."
        )

    backend, job_id, result = run_on_ibm(circuits, args.backend, args.shots, args.token, args.n_qubits)
    rows = []
    for index, label in enumerate(labels):
        counts = counts_from_sampler_result(result, index)
        edges = chain_edges(args.n_qubits) if label["variant"] == "hvk1d" else grid_edges(args.n_qubits)
        rows.append({**label, "backend": backend, "job_id": job_id, **order_from_counts(counts, edges, args.n_qubits)})
    output_path = args.output_dir / "ibm_hvk_probe_results.json"
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_result_visualization(rows, args.output_dir / "ibm_hvk_probe_metrics.png")
    print(f"IBM job complete: backend={backend}, job_id={job_id}")
    print(f"Results: {output_path}")


def save_circuit_visualization(circuits: list[QuantumCircuit], labels: list[dict], output_path: Path) -> None:
    names = [f"{label['variant']}-{label['patch_index']}" for label in labels]
    depths = [circuit.depth() for circuit in circuits]
    two_qubit_ops = [circuit.count_ops().get("cx", 0) for circuit in circuits]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, depths, color="tab:blue")
    axes[0].set_title("Circuit depth")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(names, two_qubit_ops, color="tab:orange")
    axes[1].set_title("CX count")
    axes[1].tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_result_visualization(rows: list[dict], output_path: Path) -> None:
    names = [f"{row['variant']}-{row['patch_index']}" for row in rows]
    metrics = [
        ("mean_abs_order_parameter", "Abs order"),
        ("mean_zz_correlation", "ZZ correlation"),
        ("hardware_proxy_loss", "Proxy loss"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(names, [row[key] for row in rows], marker="o")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
