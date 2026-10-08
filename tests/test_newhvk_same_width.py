import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "main2" / "newHVK" / "run_newhvk_suite.py"
SPEC = importlib.util.spec_from_file_location("run_newhvk_suite", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUITE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUITE
SPEC.loader.exec_module(SUITE)


def test_same_width_feature_maps_reserve_all_positional_channels():
    rng = np.random.default_rng(123)
    base = rng.normal(size=(5, 26))
    expected_position = base[:, 18:26]

    feature_maps = [
        SUITE.real_newhvk_features,
        SUITE.real_no_entanglement_features,
        SUITE.real_zz_only_features,
        SUITE.real_local_observables_only,
        SUITE.real_quadratic_classical_features,
        SUITE.real_raw_linear_features,
    ]

    for feature_map in feature_maps:
        features = feature_map(base)
        assert features.shape == (5, 32)
        np.testing.assert_array_equal(features[:, 24:32], expected_position)


def test_shuffled_pair_map_preserves_position_and_changes_only_pair_columns():
    rng = np.random.default_rng(456)
    base = rng.normal(size=(12, 26))
    original = SUITE.real_newhvk_features(base)
    shuffled = SUITE.real_shuffled_pair_features(base, seed=0)

    np.testing.assert_array_equal(shuffled[:, :18], original[:, :18])
    np.testing.assert_array_equal(shuffled[:, 24:32], base[:, 18:26])
    assert not np.array_equal(shuffled[:, 18:24], original[:, 18:24])
