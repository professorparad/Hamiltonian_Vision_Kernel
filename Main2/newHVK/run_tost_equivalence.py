"""TOST equivalence test for held-out CIFAR-10 reconstruction (N2, q1 task list).

This does not train or run anything new -- it is a statistical re-analysis of the
existing per-image, per-seed held-out results already recorded in
`results/q1_validation/real_cifar_holdout.csv` (5 split seeds x 4 held-out images x
9 feature maps). The paper's held-out table (Table `tab:hvk_real_cifar_stats` in the
supplement) already reports the *difference* test (Wilcoxon, bootstrap CI) for
"does HVK2D differ from each control". That framing can only ever fail to reject a
null of "no difference" -- it cannot license the word "competitive", because failing
to find a difference is not the same as showing the difference is small. A Two
One-Sided Tests (TOST) equivalence test instead asks the question the paper actually
wants to make: is the HVK2D-minus-control PSNR gap, whichever sign it has, safely
inside a pre-declared +/-1 dB margin of practical equivalence?

For each control C and each of the 5 split seeds, the paired unit is the same one
used elsewhere in this study: the seed-level mean of (HVK2D PSNR - control PSNR)
over that seed's 4 held-out images (`inference_unit: seed mean over four held-out
images` in `paired_statistical_tests.json`). TOST is then two one-sided paired
t-tests on those 5 seed-level differences against the declared margin
delta = 1.0 dB:
  H0_lower: true mean difference <= -delta   vs  H1_lower: > -delta
  H0_upper: true mean difference >=  delta   vs  H1_upper: <  delta
Equivalence (both nulls rejected at alpha=0.05) requires both one-sided p-values
below 0.05, equivalently the two-sided 90% CI of the mean difference lying entirely
inside [-delta, +delta].
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_CSV = ROOT / "Main2" / "newHVK" / "results" / "q1_validation" / "real_cifar_holdout.csv"
OUT_DIR = ROOT / "Main2" / "newHVK" / "results" / "q1_validation"
OUT_JSON = OUT_DIR / "tost_equivalence.json"
OUT_CSV = OUT_DIR / "tost_equivalence.csv"

TREATMENT = "HVK2D-real-cifar"
DELTA_DB = 1.0
ALPHA = 0.05

# Display names matching the supplement's Table tab:hvk_real_cifar_stats.
CONTROL_LABELS = {
    "raw-linear-classical": "Raw-linear classical",
    "local-observables-only": "Local observables only",
    "no-entanglement": "No entanglement",
    "zz-only": "ZZ-only observables",
    "quadratic-classical": "Quadratic classical",
    "shuffled-pair-observables": "Shuffled pair observables",
    "strict-classical-rff": "Strict classical random features",
    "random-vqc": "Random VQC",
}


def load_seed_image_psnr(csv_path: Path) -> dict[str, dict[int, dict[str, float]]]:
    """model -> seed -> {image: psnr}."""
    data: dict[str, dict[int, dict[str, float]]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = row["model"]
            seed = int(row["seed"])
            image = row["image"]
            psnr = float(row["psnr"])
            data.setdefault(model, {}).setdefault(seed, {})[image] = psnr
    return data


def seed_level_diffs(data: dict, treatment: str, control: str) -> np.ndarray:
    """Per-seed mean(treatment PSNR - control PSNR) over the seed's held-out images."""
    t_by_seed = data[treatment]
    c_by_seed = data[control]
    seeds = sorted(set(t_by_seed) & set(c_by_seed))
    diffs = []
    for seed in seeds:
        t_images = t_by_seed[seed]
        c_images = c_by_seed[seed]
        shared_images = sorted(set(t_images) & set(c_images))
        assert shared_images, f"No shared held-out images for seed {seed}, control {control}"
        per_image_diff = [t_images[img] - c_images[img] for img in shared_images]
        diffs.append(float(np.mean(per_image_diff)))
    return np.array(diffs, dtype=float)


def tost_paired(diffs: np.ndarray, delta: float, alpha: float) -> dict:
    n = len(diffs)
    mean = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    se = sd / np.sqrt(n)
    df = n - 1

    # Lower one-sided test: H0 mean <= -delta vs H1 mean > -delta
    t_lower = (mean - (-delta)) / se
    p_lower = float(1 - stats.t.cdf(t_lower, df))
    # Upper one-sided test: H0 mean >= delta vs H1 mean < delta
    t_upper = (mean - delta) / se
    p_upper = float(stats.t.cdf(t_upper, df))

    p_tost = max(p_lower, p_upper)  # TOST p-value is the max of the two one-sided p-values
    equivalent = bool(p_lower < alpha and p_upper < alpha)

    # Two-sided 90% CI (alpha=0.05 each side -> 90% CI), the standard TOST-equivalent CI.
    t_crit_90 = stats.t.ppf(1 - alpha, df)
    ci90_low = mean - t_crit_90 * se
    ci90_high = mean + t_crit_90 * se

    return {
        "n_seeds": n,
        "mean_diff_db": mean,
        "sd_diff_db": sd,
        "se_diff_db": se,
        "df": df,
        "delta_margin_db": delta,
        "alpha": alpha,
        "t_lower": float(t_lower),
        "p_lower": p_lower,
        "t_upper": float(t_upper),
        "p_upper": p_upper,
        "tost_p_value": p_tost,
        "ci90_low_db": float(ci90_low),
        "ci90_high_db": float(ci90_high),
        "equivalent_at_1db": equivalent,
        "ci90_within_margin": bool(ci90_low > -delta and ci90_high < delta),
    }


def main() -> None:
    data = load_seed_image_psnr(HOLDOUT_CSV)
    assert TREATMENT in data, f"{TREATMENT} not found in {HOLDOUT_CSV}"

    results = []
    for control, label in CONTROL_LABELS.items():
        if control not in data:
            continue
        diffs = seed_level_diffs(data, TREATMENT, control)
        stats_row = tost_paired(diffs, DELTA_DB, ALPHA)
        stats_row["control"] = control
        stats_row["control_label"] = label
        stats_row["seed_diffs_db"] = diffs.tolist()
        results.append(stats_row)
        verdict = "EQUIVALENT" if stats_row["equivalent_at_1db"] else "not equivalent"
        print(
            f"{label:35s} mean={stats_row['mean_diff_db']:+.3f} dB  "
            f"90% CI=[{stats_row['ci90_low_db']:+.3f}, {stats_row['ci90_high_db']:+.3f}]  "
            f"p_lower={stats_row['p_lower']:.4f} p_upper={stats_row['p_upper']:.4f}  "
            f"TOST p={stats_row['tost_p_value']:.4f}  -> {verdict} (+/-{DELTA_DB} dB)"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "control_label", "control", "n_seeds", "mean_diff_db", "sd_diff_db",
                "se_diff_db", "df", "delta_margin_db", "alpha", "t_lower", "p_lower",
                "t_upper", "p_upper", "tost_p_value", "ci90_low_db", "ci90_high_db",
                "equivalent_at_1db", "ci90_within_margin",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    n_equiv = sum(r["equivalent_at_1db"] for r in results)
    print(f"\n{n_equiv}/{len(results)} controls statistically equivalent to HVK2D within +/-{DELTA_DB} dB (alpha={ALPHA}).")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
