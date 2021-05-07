import logging
from abc import ABC

import numpy as np
import quadpy
import torch
from cormorant.cg_lib import SphericalHarmonics
from cormorant.so3_lib import SO3Vec
from torch.distributions import Uniform
from torch.distributions.distribution import Distribution

from .so3_tools import sum_product_alms_ylms, generate_fibonacci_grid, normalize_alms


class SphericalDistribution(Distribution, ABC):
    arg_constraints = {}  # type: ignore
    has_rsample = False

    def __init__(self, batch_shape=torch.Size(), validate_args=None, device=None, dtype=torch.float) -> None:
        super().__init__(batch_shape, event_shape=torch.Size((3, )), validate_args=validate_args)
        self.device = device
        self.dtype = dtype

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(SphericalDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.device = self.device
        new.dtype = self.dtype
        super(SphericalDistribution, new).__init__(batch_shape, event_shape=self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @staticmethod
    def _spherical_to_cartesian(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        x = torch.sin(theta) * torch.cos(phi)
        y = torch.sin(theta) * torch.sin(phi)
        z = torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    def argmax(self) -> torch.Tensor:
        raise NotImplementedError


class SphericalUniform(SphericalDistribution):
    def __init__(self, batch_shape=torch.Size(), validate_args=None, device=None, dtype=torch.float) -> None:
        super().__init__(batch_shape, validate_args=validate_args, device=device, dtype=dtype)
        self.uniform_dist = Uniform(0.0, 1.0)

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        # Based on: http://corysimon.github.io/articles/uniformdistn-on-sphere/
        # Get shape
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        shape = sample_shape + self._batch_shape

        # Sample from transformed uniform
        theta = torch.acos(1 - 2 * self.uniform_dist.sample(shape).to(self.device))
        phi = 2 * np.pi * self.uniform_dist.sample(shape).to(self.device)

        # Convert to Cartesian coordinates
        return self._spherical_to_cartesian(theta=theta, phi=phi)

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        if self._validate_args:
            self._validate_sample(value)

        return torch.ones(size=value.shape[:-1], device=self.device) / (4 * np.pi)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return torch.log(self.prob(value).clamp(min=1e-10))

    def get_max_prob(self) -> torch.Tensor:
        return torch.ones(size=self.batch_shape, device=self.device) / (4 * np.pi)

    def argmax(self) -> torch.Tensor:
        return self.sample()


class SO3Distribution(SphericalDistribution):
    def __init__(self,
                 a_lms: SO3Vec,
                 sphs: SphericalHarmonics,
                 empty: torch.Tensor = None,
                 validate_args=None,
                 device=None,
                 dtype=torch.float) -> None:
        # SO3Vec: -ell, ..., ell: (batch size, tau's, m's, 2)
        assert all(a_lm.shape[:-3] == a_lms[0].shape[:-3] for a_lm in a_lms)
        super().__init__(batch_shape=a_lms[0].shape[:-3], validate_args=validate_args, device=device, dtype=dtype)

        assert sphs.sh_norm == 'qm'
        self.sphs = sphs

        assert empty is None or empty.shape == self.batch_shape
        self.empty = empty

        self.coefficients = normalize_alms(a_lms)  # (batches, taus, ms, 2)

        self.spherical_uniform = SphericalUniform(batch_shape=self.batch_shape,
                                                  device=device,
                                                  dtype=dtype,
                                                  validate_args=validate_args)
        self.uniform_dist = Uniform(low=0.0, high=1.0, validate_args=validate_args)

    def get_max_prob(self) -> torch.Tensor:
        # grid_points: (samples, 1, 3)
        grid_points = torch.tensor(generate_fibonacci_grid(n=1024), dtype=self.dtype, device=self.device).unsqueeze(-2)

        probs = self.prob(grid_points)  # (samples, batches)

        # Maximum over grid points
        maximum, _ = torch.max(probs, dim=0)

        return maximum  # (batches, )

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        assert len(self.batch_shape) == 1
        num_batches = self.batch_shape[0]

        accepted_t = torch.empty(size=(0, num_batches), dtype=torch.bool, device=self.device)
        candidates_t = torch.empty(size=(0, num_batches) + self.event_shape, dtype=self.dtype, device=self.device)

        max_prob = self.get_max_prob()
        max_prob_proposal = self.spherical_uniform.get_max_prob()

        m_value = max_prob / max_prob_proposal  # (batches, )
        logging.debug(f'Mean M value: {torch.mean(m_value).item():.3f}')
        count = min(max(1, int(2 * torch.max(m_value).item())), 1024)

        # number of samples per batch item
        num_samples = int(np.product(sample_shape))

        while torch.any(accepted_t.sum(dim=0) < num_samples):
            candidates = self.spherical_uniform.sample(torch.Size((count, )))  # (count, batches, event)
            threshold = self.prob(candidates) / (m_value * self.spherical_uniform.prob(candidates))  # (count, batches)
            u = self.uniform_dist.sample(torch.Size((count, ))).unsqueeze(1).to(self.device)  # (count, 1)
            accepted = u < threshold  # (count, batches)

            accepted_t = torch.cat([accepted_t, accepted], dim=0)
            candidates_t = torch.cat([candidates_t, candidates], dim=0)

        # Collect accepted samples
        samples = []
        for i in range(num_batches):
            cs = candidates_t[:, i]  # (count, event)
            acs = accepted_t[:, i]  # (count, )
            samples.append(cs[acs][:num_samples])

        samples_t = torch.stack(samples, dim=0)  # (batches, samples, event)
        return samples_t.transpose(0, 1).reshape(sample_shape + self.batch_shape + self.event_shape).contiguous()

    def argmax(self, count=256) -> torch.Tensor:
        samples = self.sample(sample_shape=torch.Size((count, )))  # (samples, batches, 3)
        probs = self.prob(samples)  # (samples, batches)
        indices = torch.argmax(probs, dim=0)  # (batches, )
        gather_indices = indices.unsqueeze(0).unsqueeze(-1).expand((-1, -1) + self.event_shape)  # (1, batches, 3)
        result = torch.gather(samples, dim=0, index=gather_indices)  # (1, batches, 3)
        return result.squeeze(0)  # squeeze out samples dimension

    def prob(self, value: torch.Tensor) -> torch.Tensor:
        # value: (..., batches, 3)
        y_lms = self.sphs.forward(value)  # (..., batches, taus, ms, 2)

        # Compute sum of products over ells, taus, and ms
        s = sum_product_alms_ylms(a_lms=self.coefficients, y_lms=y_lms)  # (...., batches, 2)

        # Compute sum of squares
        p = torch.sum(torch.square(s), dim=-1, keepdim=False)  # (..., batches)

        # Apply mask where probability is not defined
        if self.empty is not None:
            empty = self.empty.reshape((1, ) * (len(p.shape) - 1) + self.batch_shape)
            constant = self.spherical_uniform.prob(value)
            p = torch.where(empty, constant, p)

        return p

    def log_prob(self, value: torch.Tensor):
        return torch.log(self.prob(value).clamp(min=1e-10))


class ExpSO3Distribution(SphericalDistribution):
    def __init__(self,
                 a_lms: SO3Vec,
                 sphs: SphericalHarmonics,
                 beta: float,
                 validate_args=None,
                 device=None,
                 dtype=torch.float) -> None:
        # SO3Vec: -ell, ..., ell: (batch size, tau's, m's, 2)
        assert all(a_lm.shape[:-3] == a_lms[0].shape[:-3] for a_lm in a_lms)
        super().__init__(batch_shape=a_lms[0].shape[:-3], validate_args=validate_args, device=device, dtype=dtype)

        assert sphs.sh_norm == 'qm'
        self.sphs = sphs

        self.coefficients = normalize_alms(a_lms)  # (batches, taus, ms, 2)
        self.sphs = sphs
        self.beta = beta

        self.spherical_uniform = SphericalUniform(batch_shape=self.batch_shape,
                                                  device=device,
                                                  dtype=dtype,
                                                  validate_args=validate_args)
        self.uniform_dist = Uniform(low=0.0, high=1.0, validate_args=validate_args)
        self.log_z = self.compute_log_z()

    def compute_log_z(self) -> torch.Tensor:
        grid = quadpy.u3._lebedev.lebedev_071()
        # grid_points: (samples, 1, 3)
        grid_points = torch.tensor(grid.points.transpose(), dtype=self.dtype, device=self.device).unsqueeze(-2)
        weights = torch.tensor(grid.weights, dtype=self.dtype, device=self.device).unsqueeze(-1)  # (samples, 1)
        log_probs_unnormalized = self.log_prob_unnormalized(grid_points)  # (samples, batches)
        result = np.log(4 * np.pi) + torch.logsumexp(log_probs_unnormalized + torch.log(weights), dim=0)
        return result

    def get_max_log_prob(self) -> torch.Tensor:
        # grid_points: (samples, 1, 3)
        grid_points = torch.tensor(generate_fibonacci_grid(n=4096), dtype=self.dtype, device=self.device).unsqueeze(1)
        log_probs = self.log_prob(grid_points)  # (samples, batches)

        # Maximum over grid points
        maximum, _ = torch.max(log_probs, dim=0)

        return maximum  # (batches, )

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        assert len(self.batch_shape) == 1
        num_batches = self.batch_shape[0]

        accepted_t = torch.empty(size=(0, num_batches), dtype=torch.bool, device=self.device)
        candidates_t = torch.empty(size=(0, num_batches) + self.event_shape, dtype=self.dtype, device=self.device)

        max_log_prob = self.get_max_log_prob()
        max_log_prob_proposal = torch.log(self.spherical_uniform.get_max_prob())

        log_m_value = max_log_prob - max_log_prob_proposal  # (batches, )
        m_value = torch.exp(log_m_value.clamp(-8, 8))

        logging.debug(f'Mean M value: {torch.mean(m_value):.3f}')
        count = min(max(1, int(2 * torch.max(m_value).item())), 1024)

        # number of samples per batch item
        num_samples = int(np.product(sample_shape))

        while torch.any(accepted_t.sum(dim=0) < num_samples):
            candidates = self.spherical_uniform.sample(torch.Size((count, )))  # (count, batches, event)
            log_threshold = self.log_prob(candidates) - log_m_value - self.spherical_uniform.log_prob(candidates)
            u = self.uniform_dist.sample(torch.Size((count, ))).unsqueeze(1).to(self.device)  # (count, 1)
            accepted = u < torch.exp(log_threshold)  # (count, batches)

            accepted_t = torch.cat([accepted_t, accepted], dim=0)
            candidates_t = torch.cat([candidates_t, candidates], dim=0)

        # Collect accepted samples
        samples = []
        for i in range(num_batches):
            cs = candidates_t[:, i]  # (count, event)
            acs = accepted_t[:, i]  # (count, )
            samples.append(cs[acs][:num_samples])

        samples_t = torch.stack(samples, dim=0)  # (batches, samples, event)
        return samples_t.transpose(0, 1).reshape(sample_shape + self.batch_shape + self.event_shape).contiguous()

    def argmax(self, count=128) -> torch.Tensor:
        samples = self.sample(sample_shape=torch.Size((count, )))  # (samples, batches, 3)
        log_probs_unnormalized = self.log_prob_unnormalized(samples)  # (samples, batches)
        indices = torch.argmax(log_probs_unnormalized, dim=0)  # (batches, )
        gather_indices = indices.unsqueeze(0).unsqueeze(-1).expand((-1, -1) + self.event_shape)  # (1, batches, 3)
        result = torch.gather(samples, dim=0, index=gather_indices)  # (1, batches, 3)
        return result.squeeze(0)  # squeeze out samples dimension

    def log_prob_unnormalized(self, value: torch.Tensor) -> torch.Tensor:
        # value: (..., batches, 3)
        y_lms = self.sphs.forward(value)  # (..., batches, taus, ms, 2)

        # Compute sum of products over ells, taus, and ms
        s = sum_product_alms_ylms(a_lms=self.coefficients, y_lms=y_lms)  # (...., batches, 2)

        # Compute sum of squares
        log_p_unnormalized = -self.beta * torch.sum(torch.square(s), dim=-1, keepdim=False)  # (..., batches)

        return log_p_unnormalized

    def log_prob(self, value: torch.Tensor):
        return self.log_prob_unnormalized(value) - self.log_z
