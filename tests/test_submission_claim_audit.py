import json
import unittest
from pathlib import Path


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

    def test_rounded_headlines_remain_in_manuscript(self):
        manuscript = (ROOT / "latex_outputs/paper_latex/paper_hvk.tex").read_text(encoding="utf-8")
        self.assertIn(r"9.57\times10^{-17}", manuscript)
        self.assertIn(r"R^2=0.974", manuscript)
        self.assertIn(r"25.90$--$31.52", manuscript)


if __name__ == "__main__":
    unittest.main()
