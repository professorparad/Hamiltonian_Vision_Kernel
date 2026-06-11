import unittest

import numpy as np

from utils.pathing import add_main_to_path

add_main_to_path()

from src.tensornetworks.mps_features import extract_mps_features
from src.tensornetworks.mps_reconstruction import mps_reconstruct


class MpsHelperTests(unittest.TestCase):
    def test_extract_mps_features_returns_expected_length_for_small_patch(self):
        patch = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)

        features = extract_mps_features(patch, n_sites=6, bond_dim=2)

        self.assertEqual(features.shape, (22,))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(features)))

    def test_mps_reconstruct_returns_patch_shape_for_small_patch(self):
        patch = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)

        reconstructed = mps_reconstruct(patch, n_sites=6, bond_dim=2, patch_size=8)

        self.assertEqual(reconstructed.shape, (8, 8))
        self.assertTrue(np.all(np.isfinite(reconstructed)))


if __name__ == "__main__":
    unittest.main()
