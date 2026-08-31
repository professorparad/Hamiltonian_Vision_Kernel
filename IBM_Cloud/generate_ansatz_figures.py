"""Regenerate figures/hvk1d_ansatz.pdf and figures/hvk2d_ansatz.pdf.

Rebuilds, gate-for-gate in Qiskit, the exact ansatz described in the paper's
own Figure caption (fig:hvk_ansatz): AngleEmbedding (RX) encoding, RY
positional rotations, then two entangling layers of RZ-RY-RZ single-qubit
rotations with a CNOT ring for HVK1D or fixed grid-edge CNOTs for HVK2D.

Single-qubit structure and the HVK1D ring topology are verified against the
real training circuit, `Main_new/src/quantum/circuit.py::VQC`
(`qml.AngleEmbedding` default rotation='X', `qml.RY` positional, then
`qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))`) by directly
inspecting that template's decomposition for n_qubits=6, n_layers=2: layer 0
is a range-1 ring (0-1,1-2,2-3,3-4,4-5,5-0), layer 1 is a range-2 ring
(0-2,1-3,2-4,3-5,4-0,5-1). The HVK2D grid topology reuses `grid_edges()`
from `run_ibm_hvk_probe.py`, the only other place in this repo that defines
an HVK2D qubit-connectivity graph, applied identically in both layers since
no per-layer variation is defined anywhere for the grid case.
"""

from __future__ import annotations

import sys
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ibm_hvk_probe import grid_edges  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "overleaf_docs" / "assets" / "figures"
N_QUBITS = 6
N_LAYERS = 2


def ring_edges(n_qubits: int, layer: int) -> list[tuple[int, int]]:
    r = (layer % (n_qubits - 1)) + 1
    return [(i, (i + r) % n_qubits) for i in range(n_qubits)]


def build_ansatz(variant: str) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS, name=variant)

    for q in range(N_QUBITS):
        qc.rx(Parameter(f"x_{q}"), q)
    qc.barrier(label="encode")

    for q in range(N_QUBITS):
        qc.ry(Parameter(f"pos_{q}"), q)
    qc.barrier(label="positional")

    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qc.rz(Parameter(f"phi{layer}_{q}"), q)
            qc.ry(Parameter(f"th{layer}_{q}"), q)
            qc.rz(Parameter(f"om{layer}_{q}"), q)

        edges = ring_edges(N_QUBITS, layer) if variant == "hvk1d" else grid_edges(N_QUBITS)
        for source, target in edges:
            qc.cx(source, target)

        qc.barrier(label=f"layer {layer + 1}")

    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in ("hvk1d", "hvk2d"):
        circuit = build_ansatz(variant)
        fig = circuit.draw(output="mpl", fold=-1)
        out_path = OUTPUT_DIR / f"{variant}_ansatz.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Wrote {out_path} (depth={circuit.depth()}, cx={circuit.count_ops().get('cx', 0)})")


if __name__ == "__main__":
    main()
