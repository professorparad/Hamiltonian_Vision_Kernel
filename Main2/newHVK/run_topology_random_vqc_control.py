"""Random-VQC floor for the real-circuit topology comparison (item G2).

`run_topology_comparison.py` reports held-out PSNR of 11.73 dB (HVK1D) and 11.57 dB
(HVK2D) at a reduced 90-step budget. Those absolutes sit close to the random-latent range
quoted elsewhere in the study, so the supplement cannot say whether the run learned
anything at all -- only that the *difference* between the two topologies is small. This
script measures the floor under exactly the same protocol so that question has an answer.

The control is the study's own `random-vqc`: the observable vector handed to the decoder
is resampled noise (`torch.randn`, the same thing `vqc_mode="random"` does in
`Main_new/src/quantum/quantum_model.py`), the energy is identically zero, and the decoder
is trained normally. Nothing else changes -- same two training images, same overlapping
8x8 patches at stride 4, same 90 steps, same learning rate per topology, same three
held-out images, same seeds. The decoder therefore gets the same budget and capacity but
no usable latent, which is the floor a real run has to clear.

    python Main2/newHVK/run_topology_random_vqc_control.py
    python Main2/newHVK/run_topology_random_vqc_control.py --smoke-test

Output: `Main2/newHVK/results/topology_comparison/random_vqc_control.json`, in the same
shape as `real_circuit_confirmation.json` so the two can be read side by side.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT / "Main2" / "newHVK", REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the comparison's own protocol verbatim -- constants, data loading and the
# held-out evaluation path all come from the script this control is matched to.
from run_topology_comparison import (  # noqa: E402
    HELDOUT_PATHS,
    PATCH_SIZE,
    POSITIONAL_DIM,
    SEEDS,
    STEPS,
    TRAIN_PATHS,
    eval_held_out,
    load_image_data,
    resolve_device,
)

OUT_DIR = REPO_ROOT / "Main2" / "newHVK" / "results" / "topology_comparison"
RESULT_FILE = OUT_DIR / "random_vqc_control.json"

# Observable widths of the two real models, matched exactly: HVK1D measures 6 local Z,
# 6 local X and 5 each of ZZ/XX/YY (27); HVK2D measures 6 local Z, 6 local X and the
# 7 grid-edge ZZ terms (19).
OBSERVABLE_DIM = {"HVK1D": 27, "HVK2D": 19}
LEARNING_RATE = {"HVK1D": 0.003, "HVK2D": 0.004}


class RandomLatentModel(torch.nn.Module):
    """Stand-in for the VQC that emits resampled noise instead of measured observables.

    Matches `vqc_mode="random"`: a fresh draw per forward pass, so the decoder cannot
    learn to read the latent and converges to the best input-independent prediction.
    """

    def __init__(self, observable_dim: int):
        super().__init__()
        self.observable_dim = observable_dim
        # A parameter the optimizer can hold on to; it never reaches the output, exactly
        # as the real random-vqc control's circuit weights never reach the decoder.
        self.unused = torch.nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor, positions: torch.Tensor):
        n = features.shape[0]
        observables = torch.randn(n, self.observable_dim, device=features.device)
        energies = observables.new_zeros(n)
        return observables, energies


def run_control(topology: str, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device("auto")

    train_data = [load_image_data(p, device) for p in TRAIN_PATHS]
    train_features = torch.cat([d["features"] for d in train_data], dim=0)
    train_positions = torch.cat([d["positions"] for d in train_data], dim=0)
    train_targets = torch.cat([d["targets"] for d in train_data], dim=0)

    if topology == "HVK1D":
        from src.decoder.patch_decoder import PatchDecoder

        decoder = PatchDecoder(
            observable_dim=OBSERVABLE_DIM[topology],
            positional_dim=POSITIONAL_DIM,
            patch_size=PATCH_SIZE,
        ).to(device)
    elif topology == "HVK2D":
        from Main2.src.model import PatchDecoder as PatchDecoder2D

        decoder = PatchDecoder2D(positional_dim=POSITIONAL_DIM, patch_size=PATCH_SIZE).to(device)
    else:
        raise ValueError(topology)

    model = RandomLatentModel(OBSERVABLE_DIM[topology]).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE[topology]
    )
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in decoder.parameters())

    t0 = time.perf_counter()
    for step in range(STEPS):
        model.train()
        decoder.train()
        optimizer.zero_grad()
        observables, energies = model(train_features, train_positions)
        output = decoder(observables, train_positions)
        loss = torch.mean((output - train_targets) ** 2) + 0.01 * torch.mean(energies)
        loss.backward()
        optimizer.step()
        if step % 30 == 0 or step == STEPS - 1:
            print(f"  [{topology}-random seed={seed}] step {step:>3d}: loss={loss.item():.6f}", flush=True)
    elapsed = time.perf_counter() - t0

    return {
        "topology": f"{topology}-random-vqc",
        "seed": seed,
        "n_params": n_params,
        "wall_time_s": elapsed,
        "steps": STEPS,
        "train_images": [p.name for p in TRAIN_PATHS],
        "held_out_metrics": {p.name: eval_held_out(model, decoder, device, p) for p in HELDOUT_PATHS},
    }


def summarize(runs: list[dict]) -> dict:
    out = {}
    for topology in ("HVK1D-random-vqc", "HVK2D-random-vqc"):
        psnrs = [
            m["psnr"]
            for r in runs
            if r["topology"] == topology
            for m in r["held_out_metrics"].values()
        ]
        if psnrs:
            out[topology] = {
                "n_image_seed_pairs": len(psnrs),
                "mean_psnr_db": statistics.mean(psnrs),
                "std_psnr_db": statistics.pstdev(psnrs),
                "min_psnr_db": min(psnrs),
                "max_psnr_db": max(psnrs),
            }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="2 steps, 1 seed")
    args = parser.parse_args()

    global STEPS, SEEDS  # noqa: PLW0603 - mirrors run_topology_comparison.main
    steps, seeds = STEPS, SEEDS
    if args.smoke_test:
        steps, seeds = 2, [0]

    import run_topology_comparison as base

    base.STEPS = steps
    globals()["STEPS"] = steps

    print("Train images:", [p.name for p in TRAIN_PATHS])
    print("Held-out images:", [p.name for p in HELDOUT_PATHS])
    print(f"Protocol: {steps} steps, seeds {seeds}, random-VQC latent (resampled noise)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    for topology in ("HVK1D", "HVK2D"):
        for seed in seeds:
            print(f"\n=== {topology} random-VQC control, seed={seed} ===")
            runs.append(run_control(topology, seed))
            RESULT_FILE.write_text(
                json.dumps({"runs": runs, "summary": summarize(runs)}, indent=2, default=str),
                encoding="utf-8",
            )

    print("\nSummary (held-out PSNR over 3 images x seeds):")
    for topology, stats in summarize(runs).items():
        print(
            f"  {topology:20} mean {stats['mean_psnr_db']:.2f} +/- {stats['std_psnr_db']:.2f} dB "
            f"(min {stats['min_psnr_db']:.2f}, max {stats['max_psnr_db']:.2f}, "
            f"n={stats['n_image_seed_pairs']})"
        )
    print("\nSaved to", RESULT_FILE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
