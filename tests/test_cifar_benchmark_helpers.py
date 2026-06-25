import unittest

import numpy as np

from Baselines.cifar10_comparisons.common import stitch_overlapping_patches


class CifarBenchmarkHelperTests(unittest.TestCase):
    def test_stitch_overlapping_patches_averages_shared_pixels(self):
        patches = np.array(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
            ],
            dtype=np.float32,
        )
        positions = np.array([[0.0, 0.0], [0.0, 0.25]], dtype=np.float32)

        image = stitch_overlapping_patches(
            patches,
            positions,
            image_size=4,
            patch_size=2,
        )

        expected = np.array(
            [
                [1.0, 2.0, 3.0, 0.0],
                [1.0, 2.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(image, expected)


if __name__ == "__main__":
    unittest.main()
