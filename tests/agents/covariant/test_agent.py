import os
from unittest import TestCase

import ase.data
import ase.io
import numpy as np
import pkg_resources
import torch
from cormorant.so3_lib import rotations

from molgym.agents.covariant.agent import CovariantAC
from molgym.agents.covariant.so3_tools import generate_fibonacci_grid, AtomicScalars
from molgym.spaces import ActionSpace, ObservationSpace
from molgym.tools import util

RESOURCES_FOLDER = 'resources'


class CovariantAgentTest(TestCase):
    RESOURCES = pkg_resources.resource_filename(__package__, RESOURCES_FOLDER)

    def setUp(self) -> None:
        util.set_seeds(0)
        self.device = torch.device('cpu')
        self.action_space = ActionSpace(zs=[1])
        self.observation_space = ObservationSpace(canvas_size=5, zs=[0, 1, 6, 8])
        self.agent = CovariantAC(
            observation_space=self.observation_space,
            action_space=self.action_space,
            min_max_distance=(0.9, 1.8),
            network_width=64,
            bag_scale=1,
            device=self.device,
            beta=100,
            maxl=4,
            num_cg_levels=3,
            num_channels_hidden=10,
            num_channels_per_element=4,
            num_gaussians=3,
        )
        self.formula = ((1, 1), )

    def verify_alms(self, atoms):
        observation = self.observation_space.build(atoms, formula=self.formula)
        util.set_seeds(0)
        action = self.agent.step([observation])
        so3_dist = action['dists'][-1]

        # Rotate
        wigner_d, rot_mat, angles = rotations.gen_rot(self.agent.max_sh, dtype=self.agent.dtype)
        atoms.positions = np.einsum('ij,...j->...i', rot_mat, atoms.positions)

        observation = self.observation_space.build(atoms, formula=self.formula)
        util.set_seeds(0)
        action = self.agent.step([observation])
        so3_dist_rot = action['dists'][-1]

        rotated_b_lms = so3_dist.coefficients.apply_wigner(wigner_d)
        for part1, part2 in zip(so3_dist_rot.coefficients, rotated_b_lms):
            max_delta = torch.max(torch.abs(part1 - part2))
            self.assertTrue(max_delta < 1e-5)

    def test_rotations(self):
        for file in ['h2o.xyz', 'ch3.xyz', 'ch4.xyz']:
            self.verify_alms(atoms=ase.io.read(filename=os.path.join(self.RESOURCES, file), format='xyz', index=0))

    def verify_probs(self, atoms):
        grid_points = torch.tensor(generate_fibonacci_grid(n=100_000), dtype=torch.float, device=self.device)
        grid_points = grid_points.unsqueeze(-2)

        observation = self.observation_space.build(atoms, formula=self.formula)
        util.set_seeds(0)
        action = self.agent.step([observation])
        so3_dist = action['dists'][-1]

        # Rotate atoms
        wigner_d, rot_mat, angles = rotations.gen_rot(self.agent.max_sh, dtype=self.agent.dtype)
        atoms_rotated = atoms.copy()
        atoms_rotated.positions = np.einsum('ij,...j->...i', rot_mat, atoms.positions)

        observation = self.observation_space.build(atoms_rotated, formula=self.formula)
        util.set_seeds(0)
        action = self.agent.step([observation])
        so3_dist_rot = action['dists'][-1]

        log_probs = so3_dist.log_prob(grid_points)  # (samples, batches)
        log_probs_rot = so3_dist_rot.log_prob(grid_points)  # (samples, batches)

        # Maximum over grid points
        maximum, max_indices = torch.max(log_probs, dim=0)
        minimum, min_indices = torch.min(log_probs, dim=0)

        maximum_rot, max_indices_rot = torch.max(log_probs_rot, dim=0)
        minimum_rot, min_indices_rot = torch.min(log_probs_rot, dim=0)

        self.assertTrue(torch.allclose(maximum, maximum_rot, atol=5e-3))
        self.assertTrue(torch.allclose(minimum, minimum_rot, atol=5e-3))

    def test_distribution(self):
        for file in ['h2o.xyz', 'ch3.xyz', 'ch4.xyz']:
            self.verify_probs(atoms=ase.io.read(filename=os.path.join(self.RESOURCES, file), format='xyz', index=0))

    def verify_invariance(self, atoms):
        atomic_scalars = AtomicScalars(maxl=self.agent.max_sh)

        observation = self.observation_space.build(atoms, formula=self.formula)
        util.set_seeds(0)
        action = self.agent.step([observation])
        so3_dist = action['dists'][-1]
        scalars = atomic_scalars(so3_dist.coefficients)

        # Rotate atoms
        wigner_d, rot_mat, angles = rotations.gen_rot(self.agent.max_sh, dtype=self.agent.dtype)
        atoms_rotated = atoms.copy()
        atoms_rotated.positions = np.einsum('ij,...j->...i', rot_mat, atoms.positions)

        observation = self.observation_space.build(atoms_rotated, formula=self.formula)
        util.set_seeds(0)
        action = self.agent.step([observation])
        so3_dist_rot = action['dists'][-1]
        scalars_rot = atomic_scalars(so3_dist_rot.coefficients)

        self.assertTrue(torch.allclose(scalars, scalars_rot, atol=1e-05))

    def test_invariance(self):
        for file in ['h2o.xyz', 'ch3.xyz', 'ch4.xyz']:
            self.verify_invariance(
                atoms=ase.io.read(filename=os.path.join(self.RESOURCES, file), format='xyz', index=0))
