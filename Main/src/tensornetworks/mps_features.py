import numpy as np
import quimb.tensor as qtn

Z_OP = np.array([[1, 0], [0, -1]], dtype=complex)
X_OP = np.array([[0, 1], [1, 0]], dtype=complex)


def local_expectation_values(mps, operator, site):
    mps_copy = mps.copy()
    mps_copy.gate_(operator, site)
    return float((mps.H @ mps_copy).real)


def two_site_expectation(mps, operator_1, operator_2, site_1, site_2):
    mps_copy = mps.copy()
    mps_copy.gate_(operator_1, site_1)
    mps_copy.gate_(operator_2, site_2)
    return float((mps.H @ mps_copy).real)


def extract_mps_features(patch: np.ndarray, n_sites: int = 12, bond_dim: int = 4):
    vector = patch.flatten()
    if vector.size != 2**n_sites:
        raise ValueError("patch size must contain exactly 2 ** n_sites values")

    vector = vector / (np.linalg.norm(vector) + 1e-8)
    psi = vector.reshape([2] * n_sites)
    mps = qtn.MatrixProductState.from_dense(psi)
    mps.compress(max_bond=bond_dim)
    mps.normalize()
    features = []

    for site in range(n_sites):
        features.append(local_expectation_values(mps, Z_OP, site))
        features.append(local_expectation_values(mps, X_OP, site))

    for site in range(n_sites - 1):
        features.append(two_site_expectation(mps, Z_OP, Z_OP, site, site + 1))

    for bond in range(1, n_sites):
        schmidt_values = mps.schmidt_values(bond)
        probabilities = schmidt_values**2
        probabilities = probabilities / (probabilities.sum() + 1e-8)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8))
        features.append(float(entropy))
    return np.array(features, dtype=np.float32)


def extract_patch_statistics_features(
    patch: np.ndarray,
    feature_dim: int = 46,
) -> np.ndarray:
    flat = patch.astype(np.float32).reshape(-1)
    percentiles = np.percentile(flat, [5, 10, 25, 50, 75, 90, 95])
    hist_bins = max(feature_dim - 11, 1)
    hist, _ = np.histogram(flat, bins=hist_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    stats = np.array(
        [
            flat.mean(),
            flat.std(),
            flat.min(),
            flat.max(),
            *percentiles,
        ],
        dtype=np.float32,
    )
    features = np.concatenate([stats, hist]).astype(np.float32)
    if features.size < feature_dim:
        features = np.pad(features, (0, feature_dim - features.size))
    return features[:feature_dim]
