from typing import List

import numpy as np
import torch
from cormorant.so3_lib import SO3Vec


def generate_fibonacci_grid(n: int) -> np.ndarray:
    # Based on: http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    golden_ratio = (1 + 5**0.5) / 2
    offset = 0.5

    index = np.arange(0, n)
    theta = np.arccos(1 - 2 * (index + offset) / n)
    phi = 2 * np.pi * index / golden_ratio

    theta_phi = np.stack([theta, phi], axis=-1)

    return spherical_to_cartesian(theta_phi)


def spherical_to_cartesian(theta_phi: np.ndarray) -> np.ndarray:
    theta, phi = theta_phi[..., 0], theta_phi[..., 1]
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def cartesian_to_spherical(pos: np.ndarray) -> np.ndarray:
    theta_phi = np.empty(shape=pos.shape[:-1] + (2, ))

    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)
    theta_phi[..., 0] = np.arccos(z / r)  # theta
    theta_phi[..., 1] = np.arctan2(y, x)  # phi

    return theta_phi


def complex_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_r, a_i = a.unbind(-1)
    b_r, b_i = b.unbind(-1)
    return torch.stack([a_r * b_r - a_i * b_i, a_i * b_r + a_r * b_i], dim=-1)


def sum_product_alms_ylms(a_lms: SO3Vec, y_lms: SO3Vec) -> torch.Tensor:
    # Dimensions of SO3Vec's for each ell: (batches, taus, ms, 2)
    assert a_lms.ells == y_lms.ells

    summands = []
    for ell in a_lms.ells:
        product = complex_product(a_lms[ell], y_lms[ell])
        summand = torch.einsum('...tmx->...x', product)  # sum over tau and m
        summands.append(summand)

    # sum over ell's
    return torch.sum(torch.stack(summands, dim=0), dim=0)  # (..., batches, 2)


def get_normalization_constant(a_lms: SO3Vec) -> torch.Tensor:
    # Dimensions of SO3Vec's for each ell: (batches, taus, ms, 2)
    summands = []
    for ell in a_lms.ells:
        a_lm = torch.einsum('...btmx->...bmx', a_lms[ell])  # sum over tau's
        squared = torch.square(a_lm)
        item = torch.einsum('...bmx->...b', squared)  # sum over m's and real and imaginary components
        summands.append(item)

    return torch.sum(torch.stack(summands, dim=0), dim=0)  # sum over ell's


def normalize_alms(a_lms: SO3Vec) -> SO3Vec:
    # Normalize a_lms such that:
    # \sum_\ell \sum_m | a_lm |^2 = 1
    k = get_normalization_constant(a_lms)  # [batches]
    clamped_k = k.clamp(min=1e-10)
    sqrt_k = torch.sqrt(clamped_k).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batches, 1, 1, 1]
    return SO3Vec([part / sqrt_k for part in a_lms])


def estimate_alms(y_lms_conj: SO3Vec) -> SO3Vec:
    # Dimensions of SO3Vec's for each ell: (batches, taus, ms, 2)

    # Compute mean over samples
    means = []
    for ell in y_lms_conj.ells:
        # select all batch dimensions
        dim = list(range(len(y_lms_conj[ell].shape) - 3))
        means.append(torch.mean(y_lms_conj[ell], dim=dim, keepdim=True))
    return SO3Vec(means)


def concat_so3vecs(so3vecs: List[SO3Vec]) -> SO3Vec:
    # Concat SO3Vecs along batch dimension
    # Dimensions of SO3Vec's for each ell: (batches, taus, ms, 2)

    # Ensure that all SO3 vectors are of the same kind
    assert all(so3vec.ells == so3vecs[0].ells for so3vec in so3vecs)

    return SO3Vec(list(map(lambda tensors: torch.cat(tensors, dim=0), zip(*so3vecs))))


def unsqueeze_so3vec(vec: SO3Vec, dim: int) -> SO3Vec:
    return SO3Vec([t.unsqueeze(dim) for t in vec])


def select_atomic_covariats(vec: SO3Vec, focus: torch.Tensor) -> SO3Vec:
    # vec (per ell): [batches, atoms, taus, ms, 2]
    # focus: [batches, atoms]
    vectors = []
    for ell in vec.ells:
        vectors.append(torch.einsum('ba,batmx->btmx', focus, vec[ell]))  # type: ignore

    return SO3Vec(vectors)  # (batches, taus, ms, 2)


def select_taus(vec: SO3Vec, indices: torch.Tensor) -> SO3Vec:
    vectors = []
    # vec: (..., taus, ms, 2)
    for ell in vec.ells:
        gather_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, (2 * ell + 1), 2)
        vectors.append(torch.gather(vec[ell], dim=1, index=gather_indices))

    return SO3Vec(vectors)  # (..., sliced_taus, ms, 2)


def select_atomic_invariats(invariats: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
    # invariats: [batches, atoms, feats]
    # focus: [batches, atoms]
    # return: [batches, feats]
    return torch.einsum('ba,baf->bf', focus, invariats)  # type: ignore


def select_element(vec: SO3Vec, element_oh: torch.Tensor) -> SO3Vec:
    # vec (per ell): [batches, taus, ms, 2]
    # element_oh: [batches, taus]
    tensors = []
    for ell in vec.ells:
        t = torch.einsum('bt,btmx->bmx', element_oh, vec[ell])  # type: ignore # [batches, ms, 2]
        t = t.unsqueeze(dim=-3)  # [batches, 1, ms, 2]
        tensors.append(t)

    return SO3Vec(tensors)  # [batches, 1, ms, 2]


class AtomicScalars(torch.nn.Module):
    """
    Based on Cormorant's GetScalarsAtom class.
    Construct a set of scalar feature vectors for each atom by using the
    covariant atom :class:`SO3Vec` representations.
    """
    def __init__(self, maxl: int, full_scalars=True, device=None, dtype=torch.float):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.maxl = maxl

        signs = [torch.pow(-1, torch.arange(-m, m + 1)) for m in range(self.maxl + 1)]
        signs = [torch.stack([s, -s], dim=-1) for s in signs]
        self.signs = [s.to(device=self.device, dtype=self.dtype) for s in signs]  # (ms, 2)

        self.full_scalars = full_scalars

    def get_output_dim(self, channels: int) -> int:
        if self.full_scalars:
            return (self.maxl + 2) * channels * 2
        else:
            return channels * 2

    def forward(self, vec: SO3Vec) -> torch.Tensor:
        # Selection of invariant part
        scalars = [vec[0]]  # (..., taus, 1, 2)

        if self.full_scalars:
            # Scalar product with itself
            scalars_prod = [(sign * part * part.flip(-2)).sum(dim=(-1, -2), keepdim=True)
                            for part, sign in zip(vec, self.signs)]  # (..., taus, 1, 1)

            # SO3 invariant norm
            scalars_norm = [(part * part).sum(dim=(-1, -2), keepdim=True) for part in vec]  # (..., taus, 1, 1)

            # Put invariant components together
            # (..., taus, 1, 2)
            scalars += [torch.cat([s_prod, s_norm], dim=-1) for s_prod, s_norm in zip(scalars_prod, scalars_norm)]

        # Concat parts together along tau dimension
        scalars_cat = torch.cat(scalars, dim=-3)  # (..., x * taus, 1, 2)

        return scalars_cat.flatten(start_dim=-3)  # (..., output_dim)
