"""Rebuilds the old (never-committed) experiments/quantum_contribution/verify_shuffle_permutations.py.

That script lived under experiments/quantum_contribution/, which the root .gitignore
blanket-excluded, and it was never committed anywhere -- there is no git history to
recover it from (see TODO/todo.md B2). The underlying finding is already documented
and trusted (see Main2/newHVK/results/ablation_study/legacy_hvk_controls/eval_controls/
shuffle-observables/INTERPRETATION.md: mean PSNR drop of 0.301 +/- 0.054 dB over 5
non-identity permutations; the old 19.70dB / -12.5dB write-up in that folder is STALE
and must not be cited). This script is a from-scratch rebuild that reproduces that
finding end-to-end against the *current* Main_new pipeline, with explicit verification
steps the old finding's writeup describes but does not leave runnable code for:

1. Trains one fresh HVK1D baseline checkpoint via
   Main_new/src/training/training.py::train() (ablation_mode="baseline") and saves
   model.pt / decoder.pt to disk.
2. Reloads that checkpoint from disk (a genuine checkpoint round-trip, not in-memory
   reuse) and rebuilds the dataset with the exact same feature/positional pipeline.
3. For >=5 independently-sampled permutations, verifies via a forward-pre-hook that
   the decoder's actual received observable tensor is bit-for-bit `observables[perm]`
   -- not a stale/discarded copy of the unpermuted observables -- and logs each
   permutation's fixed-point count, asserting it is non-identity.
4. Reports mean +/- std PSNR (and SSIM) drop vs the unshuffled baseline across those
   permutations and compares it against the documented 0.301 +/- 0.054 dB finding and
   the stale -12.5 dB number (this run must NOT reproduce anything resembling -12.5 dB;
   if it does, this script reports that honestly rather than forcing agreement).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MAIN_NEW_DIR = ROOT / "Main_new"
if str(MAIN_NEW_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_NEW_DIR))

from src.decoder.patch_decoder import PatchDecoder  # noqa: E402
from src.quantum.quantum_model import QuantumModel  # noqa: E402
from src.reconstruction.patch_stitching import stictch_patches  # noqa: E402
from src.reconstruction.seam_bleading import blend_seams  # noqa: E402
from src.training.training import (  # noqa: E402
    DEFAULT_IMAGE_PATH,
    build_dataset,
    compute_image_metrics,
    resolve_device,
    train,
)

RESULT_DIR = ROOT / "Main2" / "newHVK" / "results" / "verify_shuffle_permutations"

DOCUMENTED_MEAN_DROP_DB = 0.301
DOCUMENTED_STD_DROP_DB = 0.054
STALE_DROP_DB = 12.5  # the discredited number this run must NOT reproduce

IMAGE_SIZE = 256
PATCH_SIZE = 64
POSITIONAL_DIM = 8
MPS_BOND_DIM = 4
QUBIT_COUNT = 6
OBSERVABLE_SET = "full"


def checkpoint_dir_for(use_energy_feature: bool | None) -> Path:
    tag = "default" if use_energy_feature is None else ("with_energy_feature" if use_energy_feature else "no_energy_feature")
    return RESULT_DIR / f"checkpoint_{tag}"


def result_path_for(use_energy_feature: bool | None) -> Path:
    tag = "default" if use_energy_feature is None else ("with_energy_feature" if use_energy_feature else "no_energy_feature")
    return RESULT_DIR / f"verify_shuffle_permutations_result_{tag}.json"


def train_checkpoint(steps: int, seed: int, device: str, use_energy_feature: bool | None, checkpoint_dir: Path) -> dict:
    """Train a fresh baseline checkpoint; save model.pt / decoder.pt + our own config.

    use_energy_feature=None means "use train()'s own current default" (currently True,
    an uncommitted parallel WIP change to Main_new/src/decoder/patch_decoder.py that
    feeds the Hamiltonian energy directly into the decoder as an extra input feature --
    the legacy documented 0.301+/-0.054dB shuffle finding predates that change).
    use_energy_feature=False reproduces the legacy (pre-WIP) decoder input shape, for a
    like-for-like comparison against the documented finding.
    """
    print(
        f"Training fresh baseline checkpoint: steps={steps} seed={seed} device={device} "
        f"use_energy_feature={use_energy_feature}",
        flush=True,
    )
    train_kwargs = dict(
        steps=steps,
        ablation_mode="baseline",
        seed=seed,
        save_outputs=True,
        output_dir=checkpoint_dir,
        track_order_parameters=False,
        save_epoch_media=False,
        show_plots=False,
        device=device,
    )
    if use_energy_feature is not None:
        train_kwargs["use_energy_feature"] = use_energy_feature
    _, _, outputs = train(**train_kwargs)
    config = {
        "steps": steps,
        "seed": seed,
        "device": device,
        "image_path": str(DEFAULT_IMAGE_PATH),
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "positional_dim": POSITIONAL_DIM,
        "mps_bond_dim": MPS_BOND_DIM,
        "qubit_count": QUBIT_COUNT,
        "observable_set": OBSERVABLE_SET,
        "use_energy_feature": bool(outputs.get("use_energy_feature", True)),
        "feature_mode": outputs.get("feature_mode", "mps"),
        "baseline_psnr_from_training_run_db": outputs["reconstruction_metrics"]["psnr"],
        "baseline_ssim_from_training_run": outputs["reconstruction_metrics"]["ssim"],
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config["checkpoint_dir"] = str(checkpoint_dir)
    (checkpoint_dir / "verify_shuffle_checkpoint_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return config


def load_checkpoint(config: dict, device: torch.device):
    """Rebuild model + decoder from their class definitions and load saved state dicts
    from disk -- a genuine checkpoint round-trip, exercising the same load path a
    downstream consumer of this checkpoint would use."""
    checkpoint_dir = Path(config["checkpoint_dir"])
    image, patches, raw_positions, features, positions, targets = build_dataset(
        image_path=config["image_path"],
        image_size=config["image_size"],
        patch_size=config["patch_size"],
        positional_dim=config["positional_dim"],
        device=device,
        feature_mode=config["feature_mode"],
        mps_bond_dim=config["mps_bond_dim"],
    )

    model = QuantumModel(
        feature_dim=features.shape[1],
        positional_dim=positions.shape[1],
        qubit_count=config["qubit_count"],
        observable_set=config["observable_set"],
    ).to(device)
    decoder = PatchDecoder(
        observable_dim=model.observable_dim,
        positional_dim=positions.shape[1],
        patch_size=config["patch_size"],
        use_energy_feature=config["use_energy_feature"],
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_dir / "model.pt", map_location=device))
    decoder.load_state_dict(torch.load(checkpoint_dir / "decoder.pt", map_location=device))
    model.eval()
    decoder.eval()

    return model, decoder, image, patches, features, positions


def reconstruct(pred_patches: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    return blend_seams(
        stictch_patches(pred_patches, image_size=image_size, patch_size=patch_size),
        patch_size=patch_size,
    )


def run_verification(config: dict, n_permutations: int, base_perm_seed: int, device: torch.device) -> dict:
    model, decoder, image, patches, features, positions = load_checkpoint(config, device)
    use_energy_feature = config["use_energy_feature"]

    with torch.no_grad():
        observables, energies = model(features, positions)

    n_patches = observables.shape[0]
    identity = torch.arange(n_patches)

    # --- Baseline (unshuffled) reconstruction ---
    captured = {}

    def capture_hook(_module, args, _kwargs):
        captured["received_observables"] = args[0].detach().clone()

    handle = decoder.register_forward_pre_hook(capture_hook, with_kwargs=True)
    with torch.no_grad():
        baseline_energy_arg = energies if use_energy_feature else None
        baseline_pred = decoder(observables, positions, baseline_energy_arg).cpu().numpy()
    handle.remove()
    assert torch.equal(captured["received_observables"], observables), (
        "Decoder did not receive the exact baseline observables tensor."
    )

    baseline_img = reconstruct(baseline_pred, config["image_size"], config["patch_size"])
    baseline_metrics = compute_image_metrics(baseline_img, image)
    print(
        f"Baseline (unshuffled): PSNR={baseline_metrics['psnr']:.4f}dB "
        f"SSIM={baseline_metrics['ssim']:.4f} MSE={baseline_metrics['mse']:.8f}",
        flush=True,
    )

    # --- Permutation sweep ---
    permutation_results = []
    for i in range(n_permutations):
        perm_seed = base_perm_seed + i
        generator = torch.Generator().manual_seed(perm_seed)
        perm = torch.randperm(n_patches, generator=generator)
        fixed_points = int(torch.sum(perm == identity).item())
        retries = 0
        while fixed_points == n_patches and retries < 10:  # guard against a freak identity draw
            perm_seed += 10_000
            generator = torch.Generator().manual_seed(perm_seed)
            perm = torch.randperm(n_patches, generator=generator)
            fixed_points = int(torch.sum(perm == identity).item())
            retries += 1
        is_identity = fixed_points == n_patches
        assert not is_identity, f"Permutation {i} is the identity after {retries} retries; aborting."

        shuffled_observables = observables[perm]
        shuffled_energies = energies[perm] if use_energy_feature else None

        captured.clear()
        handle = decoder.register_forward_pre_hook(capture_hook, with_kwargs=True)
        with torch.no_grad():
            shuffled_pred = decoder(shuffled_observables, positions, shuffled_energies).cpu().numpy()
        handle.remove()

        # Verify the decoder's actual received input IS observables[perm] -- the exact
        # permuted object -- and not a stale copy of the unpermuted observables.
        received = captured["received_observables"]
        decoder_received_permuted_tensor = bool(torch.equal(received, shuffled_observables))
        decoder_received_unpermuted_tensor = bool(torch.equal(received, observables))
        assert decoder_received_permuted_tensor, (
            f"Permutation {i}: decoder did not receive the exact permuted observable tensor."
        )
        assert not decoder_received_unpermuted_tensor, (
            f"Permutation {i}: decoder received the UNPERMUTED observables -- shuffle is a no-op bug."
        )

        shuffled_img = reconstruct(shuffled_pred, config["image_size"], config["patch_size"])
        shuffled_metrics = compute_image_metrics(shuffled_img, image)
        psnr_drop = baseline_metrics["psnr"] - shuffled_metrics["psnr"]
        ssim_drop = baseline_metrics["ssim"] - shuffled_metrics["ssim"]

        print(
            f"  perm {i} (seed={perm_seed}): fixed_points={fixed_points}/{n_patches} "
            f"psnr={shuffled_metrics['psnr']:.4f}dB drop={psnr_drop:.4f}dB "
            f"decoder_received_permuted_tensor={decoder_received_permuted_tensor}",
            flush=True,
        )

        permutation_results.append(
            {
                "permutation_index": i,
                "perm_seed": perm_seed,
                "permutation": perm.tolist(),
                "fixed_points": fixed_points,
                "n_patches": n_patches,
                "is_identity": is_identity,
                "shuffled_psnr_db": shuffled_metrics["psnr"],
                "shuffled_ssim": shuffled_metrics["ssim"],
                "shuffled_mse": shuffled_metrics["mse"],
                "psnr_drop_db": psnr_drop,
                "ssim_drop": ssim_drop,
                "decoder_received_permuted_tensor": decoder_received_permuted_tensor,
                "decoder_received_unpermuted_tensor": decoder_received_unpermuted_tensor,
            }
        )

    drops = np.array([r["psnr_drop_db"] for r in permutation_results], dtype=float)
    mean_drop = float(drops.mean())
    std_drop = float(drops.std(ddof=0))

    resembles_stale_number = bool(mean_drop > 5.0)  # anything approaching -12.5 dB
    same_order_of_magnitude_as_documented = bool(mean_drop < 2.0)

    return {
        "checkpoint_config": config,
        "baseline": baseline_metrics,
        "n_permutations": n_permutations,
        "permutations": permutation_results,
        "mean_psnr_drop_db": mean_drop,
        "std_psnr_drop_db": std_drop,
        "documented_mean_drop_db": DOCUMENTED_MEAN_DROP_DB,
        "documented_std_drop_db": DOCUMENTED_STD_DROP_DB,
        "stale_drop_db_must_not_reproduce": STALE_DROP_DB,
        "resembles_stale_minus_12_5db_number": resembles_stale_number,
        "same_order_of_magnitude_as_documented_finding": same_order_of_magnitude_as_documented,
        "all_permutations_non_identity": all(not r["is_identity"] for r in permutation_results),
        "all_decoder_calls_received_exact_permuted_tensor": all(
            r["decoder_received_permuted_tensor"] for r in permutation_results
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=240, help="Training steps for the fresh checkpoint.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed for the fresh checkpoint.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--perm-base-seed", type=int, default=1000)
    parser.add_argument(
        "--reuse-checkpoint",
        action="store_true",
        help="Skip training if a matching checkpoint config already exists on disk.",
    )
    parser.add_argument(
        "--use-energy-feature",
        dest="use_energy_feature",
        action="store_true",
        default=None,
        help="Force the decoder to receive the Hamiltonian energy as an input feature.",
    )
    parser.add_argument(
        "--no-energy-feature",
        dest="use_energy_feature",
        action="store_false",
        default=None,
        help=(
            "Force the legacy (pre-WIP) decoder input shape, without the energy feature, "
            "for a like-for-like comparison against the documented 0.301+/-0.054dB finding "
            "(omit this flag to use train()'s own current default, currently True)."
        ),
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = checkpoint_dir_for(args.use_energy_feature)
    result_path = result_path_for(args.use_energy_feature)
    config_path = checkpoint_dir / "verify_shuffle_checkpoint_config.json"
    if args.reuse_checkpoint and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        print(f"Reusing existing checkpoint at {checkpoint_dir}", flush=True)
    else:
        config = train_checkpoint(args.steps, args.seed, args.device, args.use_energy_feature, checkpoint_dir)

    device = resolve_device(args.device)
    result = run_verification(config, args.n_permutations, args.perm_base_seed, device)

    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== FINAL RESULT ===")
    print(f"use_energy_feature = {config['use_energy_feature']}")
    print(f"Baseline PSNR: {result['baseline']['psnr']:.4f} dB, SSIM: {result['baseline']['ssim']:.4f}")
    print(
        f"Mean PSNR drop over {result['n_permutations']} non-identity permutations: "
        f"{result['mean_psnr_drop_db']:.4f} +/- {result['std_psnr_drop_db']:.4f} dB"
    )
    print(
        f"Documented (already-trusted) finding: {DOCUMENTED_MEAN_DROP_DB} +/- {DOCUMENTED_STD_DROP_DB} dB. "
        f"Stale number that must NOT be reproduced: {STALE_DROP_DB} dB."
    )
    print(f"All permutations non-identity: {result['all_permutations_non_identity']}")
    print(
        "All decoder calls verified to receive the exact permuted tensor "
        f"(not a discarded copy): {result['all_decoder_calls_received_exact_permuted_tensor']}"
    )
    if result["resembles_stale_minus_12_5db_number"]:
        print("WARNING: this run's mean drop resembles the stale -12.5dB number. Reporting honestly, not forcing agreement.")
    elif result["same_order_of_magnitude_as_documented_finding"]:
        print("CONSISTENT with the documented 0.301 +/- 0.054 dB finding (same order of magnitude, weak degradation).")
    else:
        print(
            "NOTE: this run's mean drop is neither close to the documented 0.301dB finding nor to the stale "
            "-12.5dB number -- reporting the real measured value above as-is."
        )
    print(f"\nWrote {result_path}")


if __name__ == "__main__":
    main()
