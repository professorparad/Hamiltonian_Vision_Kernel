import unittest

import numpy as np
import torch
from utils.pathing import add_main_to_path

add_main_to_path()

from src.preprocessing.patching import extract_patches
from src.preprocessing.positional_encoding import sinusoidal_positional_encoding


class PreprocessingTests(unittest.TestCase):
    def test_extract_patches_returns_patches_and_normalized_positions(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)

        patches, positions = extract_patches(image, patch_size=2)

        self.assertEqual(patches.shape, (4, 2, 2))
        self.assertEqual(positions.shape, (4, 2))
        np.testing.assert_array_equal(patches[0], np.array([[0, 1], [4, 5]]))
        np.testing.assert_allclose(
            positions,
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.5],
                    [0.5, 0.0],
                    [0.5, 0.5],
                ],
                dtype=np.float32,
            ),
        )

    def test_extract_patches_rejects_non_divisible_image_shape(self):
        image = np.zeros((5, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "divisible"):
            extract_patches(image, patch_size=2)

    def test_extract_patches_supports_overlap_with_stride(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)

        patches, positions = extract_patches(image, patch_size=2, stride=1)

        self.assertEqual(patches.shape, (9, 2, 2))
        self.assertEqual(positions.shape, (9, 2))
        np.testing.assert_array_equal(patches[1], np.array([[1, 2], [5, 6]]))
        np.testing.assert_allclose(positions[-1], np.array([0.5, 0.5], dtype=np.float32))

    def test_sinusoidal_positional_encoding_shape_and_origin_values(self):
        positions = np.array([[0.0, 0.0], [0.5, 0.25]], dtype=np.float32)

        encoded = sinusoidal_positional_encoding(positions, d_model=8)

        self.assertIsInstance(encoded, torch.Tensor)
        self.assertEqual(encoded.shape, (2, 8))
        torch.testing.assert_close(
            encoded[0],
            torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        )


if __name__ == "__main__":
    unittest.main()
