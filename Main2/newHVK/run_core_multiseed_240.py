"""Matched-budget core ablation comparison: 9 variants x N seeds, all at 240 steps.

Rebuilds the old (never-committed) experiments/quantum_contribution/run_core_multiseed_240.py.
That script lived under experiments/quantum_contribution/, which the root .gitignore
blanket-excluded, and it was never committed anywhere -- there is no git history to
recover it from (see TODO/todo.md B1). This is a from-scratch rebuild against the
*current* pipeline rather than a restoration of lost code.

The current codebase's training entry point,
Main_new/src/training/training.py::train(), already implements all 9 variants the old
spec asked for via its `ablation_mode` kwarg/CLI flag (see `valid_ablation_modes` in
that file) -- no new ablation code was written, this script only drives the existing
function 9 x N times and aggregates the results:

    old spec name          -> current ablation_mode
    --------------------------------------------------
    baseline                -> baseline
    freeze-quantum           -> freeze-quantum
    freeze-classical          -> freeze-classical
    no-entanglement           -> no-entanglement
    no-MPS                    -> no-mps
    no-energy                 -> no-energy-loss
    classical-replacement     -> classical-replacement
    classical-matched         -> classical-matched
    random-VQC                -> random-vqc

Matched budget: every run trains for exactly 240 steps (same LR, same schedule, same
default image=Main_new/data/monalisa.jpg) regardless of variant. Reports mean +/- std
PSNR and SSIM per variant over multiple seeds, and marks any baseline-vs-classical/
random-control PSNR gap smaller than one pooled std as "not significant", per the old
task's own instruction (do not claim a within-noise gap).

Timing (measured on this machine, CPU, see --time-one-run): the 6 VQC-backed variants
(baseline, freeze-quantum, freeze-classical, no-entanglement, no-mps, no-energy-loss)
cost ~2.2s/step ~= 9 minutes for 240 steps each -- dominated by PennyLane's default.qubit
per-patch QNode call overhead, not by tensor-op FLOPs. GPU is *not* used (--device cpu):
default.qubit does not run on GPU and per-patch host<->device transfer makes CUDA slower
for this workload, consistent with the note in q1_revision/07_phase_transition_scope.
The 3 non-VQC variants (classical-replacement, classical-matched, random-vqc) cost only
a few seconds each (no QNode calls at all).

Use --skip-existing to resume an interrupted sweep from runs.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MAIN_NEW_DIR = ROOT / "Main_new"
if str(MAIN_NEW_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_NEW_DIR))

from src.training.training import train  # noqa: E402

RESULT_DIR = ROOT / "Main2" / "newHVK" / "results" / "core_multiseed_240"

STEPS = 240
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# (old_spec_name, ablation_mode)
VARIANTS: list[tuple[str, str]] = [
    ("baseline", "baseline"),
    ("freeze-quantum", "freeze-quantum"),
    ("freeze-classical", "freeze-classical"),
    ("no-entanglement", "no-entanglement"),
    ("no-MPS", "no-mps"),
    ("no-energy", "no-energy-loss"),
    ("classical-replacement", "classical-replacement"),
    ("classical-matched", "classical-matched"),
    ("random-VQC", "random-vqc"),
]

# Pairs to run the "gap < 1 pooled std => not significant" check on. baseline is the
# quantum (VQC-trained) reference; the others each remove or replace the quantum
# component in a different way.
QUANTUM_VS_CLASSICAL_PAIRS: list[tuple[str, str]] = [
    ("baseline", "classical-replacement"),
    ("baseline", "classical-matched"),
    ("baseline", "random-VQC"),
    ("baseline", "freeze-quantum"),
]

FIELDNAMES = [
    "old_variant_name",
    "ablation_mode",
    "seed",
    "steps",
    "psnr_db",
    "ssim",
    "mse",
    "elapsed_seconds",
    "error",
]


def run_one(old_name: str, ablation_mode: str, seed: int, device: str) -> dict:
    t0 = time.time()
    error = None
    psnr = ssim = mse = float("nan")
    try:
        _, _, outputs = train(
            steps=STEPS,
            ablation_mode=ablation_mode,
            seed=seed,
            save_outputs=False,
            track_order_parameters=False,
            show_plots=False,
            device=device,
        )
        metrics = outputs["reconstruction_metrics"]
        psnr, ssim, mse = metrics["psnr"], metrics["ssim"], metrics["mse"]
    except Exception as exc:  # noqa: BLE001 - keep the sweep alive, report honestly
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - t0
    return {
        "old_variant_name": old_name,
        "ablation_mode": ablation_mode,
        "seed": seed,
        "steps": STEPS,
        "psnr_db": psnr,
        "ssim": ssim,
        "mse": mse,
        "elapsed_seconds": elapsed,
        "error": error,
    }


def load_existing_rows(runs_csv: Path) -> list[dict]:
    rows = []
    if not runs_csv.exists():
        return rows
    with runs_csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            row["seed"] = int(row["seed"])
            row["steps"] = int(row["steps"])
            for key in ("psnr_db", "ssim", "mse", "elapsed_seconds"):
                row[key] = float(row[key]) if row[key] not in ("", None) else float("nan")
            row["error"] = row["error"] or None
            rows.append(row)
    return rows


def append_row(runs_csv: Path, row: dict) -> None:
    write_header = not runs_csv.exists()
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    with runs_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def summarize(rows: list[dict]) -> list[dict]:
    summary = []
    for old_name, ablation_mode in VARIANTS:
        variant_rows = [r for r in rows if r["ablation_mode"] == ablation_mode and r["error"] is None]
        if not variant_rows:
            summary.append(
                {
                    "old_variant_name": old_name,
                    "ablation_mode": ablation_mode,
                    "n_seeds": 0,
                    "mean_psnr_db": None,
                    "std_psnr_db": None,
                    "mean_ssim": None,
                    "std_ssim": None,
                    "note": "all runs failed or missing",
                }
            )
            continue
        psnrs = np.array([r["psnr_db"] for r in variant_rows], dtype=float)
        ssims = np.array([r["ssim"] for r in variant_rows], dtype=float)
        summary.append(
            {
                "old_variant_name": old_name,
                "ablation_mode": ablation_mode,
                "n_seeds": len(variant_rows),
                "mean_psnr_db": float(psnrs.mean()),
                "std_psnr_db": float(psnrs.std(ddof=0)),
                "mean_ssim": float(ssims.mean()),
                "std_ssim": float(ssims.std(ddof=0)),
                "note": "",
            }
        )
    return summary


def compare_quantum_vs_classical(summary: list[dict]) -> list[dict]:
    by_name = {row["old_variant_name"]: row for row in summary}
    comparisons = []
    for a, b in QUANTUM_VS_CLASSICAL_PAIRS:
        ra, rb = by_name.get(a), by_name.get(b)
        if not ra or not rb or ra["mean_psnr_db"] is None or rb["mean_psnr_db"] is None:
            continue
        gap = abs(ra["mean_psnr_db"] - rb["mean_psnr_db"])
        pooled_std = float(np.sqrt(((ra["std_psnr_db"] ** 2) + (rb["std_psnr_db"] ** 2)) / 2.0))
        if pooled_std > 0:
            significant = bool(gap >= pooled_std)
            verdict = (
                "significant (gap >= 1 pooled std)"
                if significant
                else "not significant (gap < 1 pooled std)"
            )
        else:
            significant = None
            verdict = "undefined (zero variance in both variants)"
        comparisons.append(
            {
                "variant_a": a,
                "variant_b": b,
                "mean_psnr_a_db": ra["mean_psnr_db"],
                "mean_psnr_b_db": rb["mean_psnr_db"],
                "psnr_gap_db": gap,
                "pooled_std_db": pooled_std,
                "significant": significant,
                "verdict": verdict,
            }
        )
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--time-one-run",
        action="store_true",
        help="Run exactly one (baseline, first seed) run, print elapsed time, and exit.",
    )
    parser.add_argument(
        "--scope-note",
        default="",
        help="If the seed count was reduced from the 5-seed spec, record why here; written into summary.json.",
    )
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    runs_csv = RESULT_DIR / "runs.csv"

    if args.time_one_run:
        row = run_one("baseline", "baseline", args.seeds[0], args.device)
        print(json.dumps(row, indent=2))
        return

    if args.skip_existing:
        all_rows = load_existing_rows(runs_csv)
        done = {(r["ablation_mode"], r["seed"]) for r in all_rows if r["error"] is None}
    else:
        all_rows = []
        done = set()
        if runs_csv.exists():
            runs_csv.unlink()

    total = len(VARIANTS) * len(args.seeds)
    counter = 0
    for old_name, ablation_mode in VARIANTS:
        for seed in args.seeds:
            counter += 1
            if (ablation_mode, seed) in done:
                print(f"[{counter}/{total}] SKIP (already done) {old_name} seed={seed}", flush=True)
                continue
            print(
                f"[{counter}/{total}] RUN {old_name} (ablation_mode={ablation_mode}) seed={seed} steps={STEPS}",
                flush=True,
            )
            row = run_one(old_name, ablation_mode, seed, args.device)
            append_row(runs_csv, row)
            all_rows.append(row)
            if row["error"] is None:
                print(
                    f"    -> OK psnr={row['psnr_db']:.3f}dB ssim={row['ssim']:.4f} "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            else:
                print(f"    -> ERROR: {row['error']} (elapsed={row['elapsed_seconds']:.1f}s)", flush=True)

    summary = summarize(all_rows)
    comparisons = compare_quantum_vs_classical(summary)

    with (RESULT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(summary[0].keys()) if summary else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    summary_json = {
        "steps": STEPS,
        "seeds_requested": args.seeds,
        "n_seeds_requested": len(args.seeds),
        "spec_seeds": 5,
        "scope_reduced": len(args.seeds) < 5,
        "scope_note": args.scope_note,
        "variants": summary,
        "quantum_vs_classical_comparisons": comparisons,
        "significance_rule": (
            "gap in mean PSNR (dB) between variant A and variant B is compared against the "
            "pooled std = sqrt((std_A^2 + std_B^2) / 2) computed across each variant's seeds; "
            "gap < pooled_std => marked 'not significant' (per the old task spec's own "
            "instruction not to claim within-noise gaps)."
        ),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print("\n=== Summary (mean +/- std over available seeds) ===")
    for row in summary:
        if row["mean_psnr_db"] is None:
            print(f"{row['old_variant_name']:24s} n=0 ALL RUNS FAILED")
            continue
        print(
            f"{row['old_variant_name']:24s} n={row['n_seeds']} "
            f"psnr={row['mean_psnr_db']:.3f}+/-{row['std_psnr_db']:.3f}dB "
            f"ssim={row['mean_ssim']:.4f}+/-{row['std_ssim']:.4f}"
        )
    print("\n=== Quantum vs classical/random-control comparisons ===")
    for c in comparisons:
        print(
            f"{c['variant_a']} vs {c['variant_b']}: gap={c['psnr_gap_db']:.3f}dB "
            f"pooled_std={c['pooled_std_db']:.3f}dB -> {c['verdict']}"
        )
    print(f"\nWrote {runs_csv}")
    print(f"Wrote {RESULT_DIR / 'summary.csv'}")
    print(f"Wrote {RESULT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
