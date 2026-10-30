import csv
import json
import unittest
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


class SubmissionClaimAuditTests(unittest.TestCase):
    def setUp(self):
        self.claims = load_json("project_artifacts/submission_claim_audit.json")["claims"]

    def test_d4_headline_matches_artifact(self):
        claim = self.claims["d4_equivariance"]
        records = {row["model"]: row for row in load_json(claim["source"])}
        retained = records["D4-pooled-HVK2D"]
        self.assertEqual(retained["mean_equivariance_error"], claim["artifact_value"])
        self.assertEqual(retained["n"], claim["n_evaluations"])

    def test_restricted_pair_headline_matches_artifact(self):
        claim = self.claims["restricted_pair_diagnostic"]
        records = {row["model"]: row for row in load_json(claim["source"])}
        self.assertEqual(
            records["HVK2D-entangling-observables"]["mean_r2"],
            claim["artifact_mean_r2"],
        )
        self.assertEqual(
            records["no-entanglement"]["mean_r2"],
            claim["strongest_non_entangling_control_r2"],
        )

    def test_hardware_range_matches_all_five_images(self):
        claim = self.claims["hardware_reconstruction"]
        monalisa = load_json(claim["monalisa_source"])
        cifar = load_json(claim["cifar_source"])
        values = [monalisa["psnr_hardware_db"]] + [row["psnr_hardware_db"] for row in cifar]
        self.assertEqual(len(values), claim["n_images"])
        self.assertEqual(min(values), claim["minimum_hardware_psnr_db"])
        self.assertEqual(max(values), claim["maximum_hardware_psnr_db"])

    def test_sameset_multi_dataset_matches_artifact(self):
        claim = self.claims["sameset_multi_dataset"]
        records = {row["dataset"]: row for row in load_json(claim["source"])}
        for dataset, expected in claim["rows"].items():
            actual = records[dataset]
            self.assertEqual(actual["n_images"], expected["n_images"])
            self.assertEqual(actual["mean_psnr"], expected["mean_psnr_db"])
            self.assertEqual(actual["mean_ssim"], expected["mean_ssim"])

    def test_zero_shot_generalization_matches_artifact(self):
        claim = self.claims["zero_shot_generalization"]
        record = load_json(claim["source"])
        self.assertEqual(record["second_image_zero_shot"], claim["artifact_value"]["second_image_zero_shot"])
        self.assertEqual(
            record["second_image_multi_image_training"],
            claim["artifact_value"]["second_image_multi_image_training"],
        )

    def test_tost_equivalence_matches_artifact(self):
        claim = self.claims["tost_equivalence"]
        records = load_json(claim["source"])
        self.assertEqual(len(records), claim["n_controls_total"])
        equivalent = [row for row in records if row["equivalent_at_1db"]]
        self.assertEqual(len(equivalent), claim["n_controls_equivalent"])
        headline = next(row for row in records if row["control"] == claim["headline_control"])
        # Guard against reusing a TOST table from a different held-out run.
        # The paired unit is one seed mean over the same four test images.
        holdout = ROOT / "Main2/newHVK/results/q1_validation/real_cifar_holdout.csv"
        by_model_seed = defaultdict(dict)
        with holdout.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                by_model_seed[row["model"], int(row["seed"])][row["image"]] = float(row["psnr"])
        for record in records:
            seed_differences = []
            for seed in range(claim["n_seeds"]):
                treatment = by_model_seed["HVK2D-real-cifar", seed]
                control = by_model_seed[record["control"], seed]
                self.assertEqual(set(treatment), set(control))
                self.assertEqual(len(treatment), 4)
                seed_differences.append(mean(treatment[image] - control[image] for image in treatment))
            self.assertAlmostEqual(record["mean_diff_db"], mean(seed_differences), places=12)
            for actual, expected in zip(record["seed_diffs_db"], seed_differences):
                self.assertAlmostEqual(actual, expected, places=12)
        self.assertEqual(headline["mean_diff_db"], claim["headline_artifact_value_db"])
        self.assertEqual(headline["tost_p_value"], claim["headline_tost_p"])
        self.assertEqual(headline["delta_margin_db"], claim["margin_db"])
        self.assertEqual(headline["holm_family_size"], claim["holm_family_size"])
        by_control = {row["control"]: row for row in records}
        audit_holm = claim["holm_adjusted_p_values"]
        for control in (
            "shuffled-pair-observables", "zz-only", "quadratic-classical",
            "no-entanglement", "strict-classical-rff", "random-vqc",
        ):
            self.assertEqual(by_control[control]["holm_adjusted_p_value"], audit_holm[control])
        self.assertEqual(
            by_control["local-observables-only"]["holm_adjusted_p_value"],
            audit_holm["raw-local-linear"],
        )
        self.assertEqual(
            by_control["raw-linear-classical"]["holm_adjusted_p_value"],
            audit_holm["raw-local-linear"],
        )
        corrected = [
            row for row in records
            if row["control"] != "raw-linear-classical" and row["equivalent_after_holm"]
        ]
        self.assertEqual(len(corrected), claim["n_distinct_controls_equivalent_after_holm"])
        # Every control HVK2D is NOT equivalent to must be one it is ahead of,
        # never one it loses to -- otherwise "competitive" would be unsupported.
        for row in records:
            if not row["equivalent_at_1db"]:
                self.assertGreater(row["mean_diff_db"], 0.0)

    def test_hamiltonian_controls_extension_matches_artifact(self):
        claim = self.claims["hamiltonian_controls_extension"]
        records = {row["ablation_mode"]: row for row in load_json(claim["source"])}
        self.assertEqual(records["no-obs-noise"]["psnr_db"], claim["rows"]["no_observable_noise"]["fresh_rerun_db"])
        self.assertEqual(records["zz-only"]["psnr_db"], claim["rows"]["zz_only_observables"]["fresh_rerun_db"])


if __name__ == "__main__":
    unittest.main()
