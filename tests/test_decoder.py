import unittest

import torch
from utils.pathing import add_main_to_path

add_main_to_path()

from src.decoder.patch_decoder import PatchDecoder


class PatchDecoderTests(unittest.TestCase):
    def test_patch_decoder_accepts_observable_and_position_dimensions(self):
        decoder = PatchDecoder(observable_dim=4, positional_dim=2, patch_size=3)

        output = decoder(torch.zeros(2, 4), torch.zeros(2, 2))

        self.assertEqual(output.shape, (2, 1, 3, 3))
        self.assertTrue(torch.all((0.0 <= output) & (output <= 1.0)))

    def test_patch_decoder_requires_input_dimensions(self):
        with self.assertRaisesRegex(ValueError, "input_dim"):
            PatchDecoder()


if __name__ == "__main__":
    unittest.main()
