import numpy as np
import quimb.tensor as qtn


def mps_reconstruct(
    patch: np.ndarray,
    n_sites: int = 12,
    bond_dim: int = 4,
    patch_size: int = 64,
):
    vector = patch.flatten()
    if vector.size != 2**n_sites:
        raise ValueError("patch size must contain exactly 2 ** n_sites values")

    norm = np.linalg.norm(vector) + 1e-8
    vector = vector / norm
    psi = vector.reshape([2] * n_sites)
    mps = qtn.MatrixProductState.from_dense(psi)
    mps.compress(max_bond=bond_dim)
    mps.normalize()
    reconstructed = mps.to_dense().reshape(-1).real * norm
    return reconstructed.reshape(patch_size, patch_size)
