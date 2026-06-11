import unittest
from unittest import mock

import numpy as np
import torch
from utils.pathing import add_main_to_path

add_main_to_path()

from src.training import training


class TrainingSmokeTests(unittest.TestCase):
    def test_build_dataset_uses_training_pipeline_dependencies(self):
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        original_extract_patches = training.extract_patches

        def extract_small_patches(image_arg, patch_size):
            self.assertEqual(patch_size, 2)
            return original_extract_patches(image_arg, patch_size=2)

        patches = [
            mock.patch.object(
                training, "load_image_grayscale", lambda path, size: image
            ),
            mock.patch.object(
                training,
                "extract_mps_features",
                lambda patch: np.array([patch.mean(), patch.std()], dtype=np.float32),
            ),
            mock.patch.object(training, "extract_patches", extract_small_patches),
        ]

        with patches[0], patches[1], patches[2]:
            (
                _,
                patch_array,
                raw_positions,
                features,
                positions,
                targets,
            ) = training.build_dataset(
                image_path="dummy-image.jpg",
                image_size=4,
                patch_size=2,
                device=torch.device("cpu"),
            )

        self.assertEqual(patch_array.shape, (4, 2, 2))
        self.assertEqual(raw_positions.shape, (4, 2))
        self.assertEqual(features.shape, (4, 2))
        self.assertEqual(positions.shape, (4, 8))
        self.assertEqual(targets.shape, (4, 1, 2, 2))
        self.assertEqual(features.device.type, "cpu")
        self.assertEqual(positions.device.type, "cpu")
        self.assertEqual(targets.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
