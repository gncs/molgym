from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.distributions
from cormorant.cg_lib import SphericalHarmonics, CGDict
from cormorant.so3_lib import SO3Tau, SO3Vec

from molgym.agents.base import AbstractActorCritic
from molgym.agents.covariant import tools, so3_tools
from molgym.agents.covariant.gmm import GaussianMixtureModel
from molgym.agents.covariant.modules import Cormorant, CormorantMixer
from molgym.agents.covariant.so3_tools import AtomicScalars
from molgym.agents.covariant.spherical_dists import SphericalDistribution, ExpSO3Distribution, SO3Distribution
from molgym.modules import MLP, masked_softmax, to_one_hot
from molgym.spaces import ObservationType, ActionType, ObservationSpace, ActionSpace
from molgym.tools.util import to_numpy


class CovariantAC(AbstractActorCritic):
    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        min_max_distance: Tuple[float, float],
        network_width: int,
        maxl: int,
        num_cg_levels: int,
        num_channels_hidden: int,
        num_channels_per_element: int,
        num_gaussians: int,
        bag_scale: int,
        beta: Optional[float] = None,
        device=None,
    ):
        super().__init__(observation_space, action_space)
        self.device = device
        self.dtype = torch.float

        self.zs = self.observation_space.zs
        self.zs_tensor = torch.tensor(self.zs, dtype=self.dtype, device=self.device)

        self.min_distance, self.max_distance = min_max_distance
        assert self.min_distance < self.max_distance
        self.beta = beta

        self.max_sh = maxl
        self.num_cg_levels = num_cg_levels
        self.num_channels_hidden = num_channels_hidden
        self.num_channels_per_element = num_channels_per_element
        self.num_gaussians = num_gaussians

        self.num_channels_out = len(self.zs) * self.num_channels_per_element
        self.channel_offsets = torch.arange(start=0,
                                            end=self.num_channels_per_element,
                                            dtype=torch.long,
                                            device=self.device).unsqueeze(0)

        self.cg_dict = CGDict(maxl=self.max_sh, device=self.device, dtype=self.dtype)
        self.cg_model = Cormorant(
            maxl=self.max_sh,  # Cutoff in CG operations (default: [3])
            max_sh=self.max_sh,  # Number of spherical harmonic powers to use (default: [3])
            num_cg_levels=self.num_cg_levels,  # Number of CG levels (default: 4)
            num_channels=[self.num_channels_hidden] * self.num_cg_levels + [self.num_channels_out],
            num_species=len(self.zs),
            cutoff_type=['soft'],  # Types of cutoffs to include
            hard_cut_rad=min(self.max_distance, 2.1),  # Radius of hard cutoff (in AA)
            soft_cut_rad=min(self.max_distance, 2.1),  # Radius of soft cutoff (in AA)
            soft_cut_width=0.2,  # Width of SOFT cutoff in Angstroms (default: 0.2)
            weight_init='rand',  # Weight initialization function to use (default: rand)
            level_gain=[10.0],  # Gain at each level (default: [10.])
            charge_power=2,  # Maximum power to take in one-hot (default: 2)
            basis_set=[3, 3],  # Use gaussian mask instead of sigmoid mask.
            charge_scale=max(self.zs),
            bag_scale=bag_scale,
            device=self.device,
            dtype=self.dtype,
            cg_dict=self.cg_dict,
        )

        self.cg_mix = CormorantMixer(
            tau_in=SO3Tau([self.num_channels_per_element] * (self.max_sh + 1)),
            tau_other=SO3Tau([self.num_channels_per_element]),
            maxl=self.max_sh,
            num_channels=self.num_channels_per_element,
            level_gain=10.0,
            weight_init='rand',
            device=self.device,
            dtype=self.dtype,
            cg_dict=self.cg_dict,
        )

        self.sph_harms = SphericalHarmonics(maxl=self.max_sh,
                                            conj=False,
                                            sh_norm='qm',
                                            device=self.device,
                                            dtype=self.dtype,
                                            cg_dict=self.cg_dict)

        self.atomic_scalars = AtomicScalars(maxl=self.max_sh, full_scalars=True, device=self.device, dtype=self.dtype)

        self.num_latent = self.atomic_scalars.get_output_dim(self.num_channels_out)
        self.num_latent_element = self.atomic_scalars.get_output_dim(self.num_channels_per_element)

        # Focus
        self.phi_focus = MLP(
            input_dim=self.num_latent,
            output_dims=(network_width, 1),
        )

        # Element
        self.phi_element = MLP(
            input_dim=self.num_latent,
            output_dims=(network_width, len(self.zs)),
        )

        # Distance: Gaussian Mixture Model
        self.phi_d = MLP(
            input_dim=self.num_latent_element,
            output_dims=(network_width, 2 * self.num_gaussians),
        )
        self.pad_zeros = torch.nn.ConstantPad1d(padding=(0, 1), value=0.0)  # Pad with one 0.0 to the right

        self.distance_half_width = torch.tensor((self.max_distance - self.min_distance) / 2,
                                                dtype=self.dtype,
                                                device=self.device)
        self.distance_center = torch.tensor((self.min_distance + self.max_distance) / 2,
                                            dtype=self.dtype,
                                            device=self.device)

        self.distance_log_stds = torch.nn.Parameter(torch.log(
            torch.tensor([0.1] * self.num_gaussians, dtype=self.dtype, device=self.device)),
                                                    requires_grad=True)  # (gaussians, )

        # Value function
        self.phi_trans = MLP(
            input_dim=self.num_latent,
            output_dims=(network_width, network_width),
        )
        self.phi_v = MLP(
            input_dim=network_width,
            output_dims=(network_width, 1),
        )

        self.to(self.device)

    def to_action_space(self, action: torch.Tensor, observation: ObservationType) -> ActionType:
        assert action.shape == (6, )
        action = to_numpy(action)

        focus = int(round(action[0].item()))
        element_index = int(round(action[1].item()))
        d = action[2]
        so3 = action[-3:]

        atoms, bag = self.observation_space.parse(observation)

        if len(atoms):
            position = atoms[focus].position + d * so3
        else:
            position = (0.0, 0.0, 0.0)

        return element_index, position

    def parse_observations(self, observations: List[ObservationType]) -> Dict[str, torch.Tensor]:
        parsed_observations = [self.observation_space.parse(observation) for observation in observations]
        atoms_list = [tup[0] for tup in parsed_observations]
        bags = [observation[1] for observation in observations]

        # Canvas
        data = tools.process_atoms_list(atoms_list,
                                        max_num_atoms=self.observation_space.canvas_space.size,
                                        dtype=self.dtype,
                                        device=self.device)

        data['one_hot'] = data['charges'].unsqueeze(-1) == self.zs_tensor.unsqueeze(0).unsqueeze(0)
        data['atom_mask'] = data['charges'] > 0
        data['edge_mask'] = data['atom_mask'].unsqueeze(1) * data['atom_mask'].unsqueeze(2)

        # At least one atom needs to be selectable
        default = torch.zeros_like(data['atom_mask'])
        default[..., 0] = 1

        # If the canvas is empty, focus 0th index
        data['focus_mask'] = torch.logical_or(data['atom_mask'], default)

        # Is canvas empty?
        data['empty'] = torch.tensor([len(atoms) == 0 for atoms in atoms_list], dtype=torch.bool, device=self.device)

        # Bag
        data['bags'] = torch.tensor([list(bag) for bag in bags], dtype=self.dtype, device=self.device)  # (batches, zs)
        data['element_mask'] = data['bags'] > 0  # (batches, zs)

        # Value mask
        data['value_mask'] = data['atom_mask']

        return data

    def get_so3_distribution(self, a_lms: SO3Vec, empty: torch.Tensor) -> SphericalDistribution:
        if self.beta is not None:
            return ExpSO3Distribution(a_lms=a_lms,
                                      sphs=self.sph_harms,
                                      beta=self.beta,
                                      dtype=self.dtype,
                                      device=self.device)
        else:
            return SO3Distribution(a_lms=a_lms, sphs=self.sph_harms, empty=empty, dtype=self.dtype, device=self.device)

    def step(self, observations: List[ObservationType], actions: Optional[np.ndarray] = None) -> dict:
        data = self.parse_observations(observations)

        # Cast action to tensor
        if actions is not None:
            actions = torch.as_tensor(actions, dtype=torch.float, device=self.device)

        # SO3Vec (batches, atoms, taus, ms, 2)
        covariats = self.cg_model(data)

        # Compute invariants
        invariats = self.atomic_scalars(covariats)  # (batches, atoms, inv_feats)

        # Focus
        focus_logits = self.phi_focus(invariats)  # (batches, atoms, 1)
        focus_logits = focus_logits.squeeze(-1)  # (batches, atoms)
        focus_probs = masked_softmax(focus_logits, mask=data['focus_mask'])  # (batches, atoms)
        focus_dist = torch.distributions.Categorical(probs=focus_probs)

        # focus: (batches, 1)
        if actions is not None:
            focus = torch.round(actions[:, :1]).long()
        elif self.training:
            focus = focus_dist.sample().unsqueeze(-1)
        else:
            focus = torch.argmax(focus_probs, dim=-1).unsqueeze(-1)

        focus_oh = to_one_hot(focus, num_classes=self.observation_space.canvas_space.size,
                              device=self.device)  # (batches, atoms)

        focused_cov = so3_tools.select_atomic_covariats(covariats, focus_oh)  # (batches, taus, ms, 2)
        focused_inv = so3_tools.select_atomic_invariats(invariats, focus_oh)  # (batches, feats)

        # Element
        element_logits = self.phi_element(focused_inv)  # (batches, zs)
        element_probs = masked_softmax(element_logits, mask=data['element_mask'])  # (batches, zs)
        element_dist = torch.distributions.Categorical(probs=element_probs)

        # element: (batches, 1)
        if actions is not None:
            element = torch.round(actions[:, 1:2]).long()
        elif self.training:
            element = element_dist.sample().unsqueeze(-1)
        else:
            element = torch.argmax(element_probs, dim=-1).unsqueeze(-1)

        # Crop element
        offsets = self.channel_offsets.expand(len(observations), -1)  # (batches, channels_per_element)
        indices = offsets + element * self.num_channels_per_element
        element_cov = so3_tools.select_taus(focused_cov, indices=indices)
        element_inv = self.atomic_scalars(element_cov)  # (batches, inv_feats)

        # Distance: Gaussian mixture model
        # gmm_log_probs, d_mean_trans: (batches, gaussians)
        gmm_log_probs, d_mean_trans = self.phi_d(element_inv).split(self.num_gaussians, dim=-1)
        distance_mean = torch.tanh(d_mean_trans) * self.distance_half_width + self.distance_center
        distance_dist = GaussianMixtureModel(log_probs=gmm_log_probs,
                                             means=distance_mean,
                                             stds=torch.exp(self.distance_log_stds).clamp(1e-6))

        # distance: (batches, 1)
        if actions is not None:
            distance = actions[:, 2:3]
        elif self.training:
            # Ensure that the sampled distance is > 0
            distance = distance_dist.sample().clamp(0.001).unsqueeze(-1)
        else:
            distance = distance_dist.argmax().unsqueeze(-1)

        # Condition on distance
        transformed_d = distance.unsqueeze(1).unsqueeze(1).expand(-1, self.num_channels_per_element, 1, -1)
        transformed_d = self.pad_zeros(transformed_d)
        distance_so3 = SO3Vec([transformed_d])
        cond_cov = self.cg_mix(element_cov, distance_so3)

        so3_dist = self.get_so3_distribution(a_lms=cond_cov, empty=data['empty'])

        # so3: (batches, 3)
        if actions is not None:
            orientation = actions[..., 3:6]
        elif self.training:
            orientation = so3_dist.sample()
        else:
            orientation = so3_dist.argmax()

        # Log prob
        log_prob_list = [
            focus_dist.log_prob(focus.squeeze(-1)),
            element_dist.log_prob(element.squeeze(-1)),
            distance_dist.log_prob(distance.squeeze(-1)),
            so3_dist.log_prob(orientation),
        ]
        log_prob = torch.stack(log_prob_list, dim=-1).sum(dim=-1)  # (batches, )

        # Entropy
        entropy_list = [
            focus_dist.entropy(),
            element_dist.entropy(),
        ]
        entropy = torch.stack(entropy_list, dim=-1).sum(dim=-1)  # (batches, )

        # Value function
        # atom_mask: (batches, atoms)
        # invariants: (batches, atoms, feats)
        trans_invariats = self.phi_trans(invariats)
        value_feats = torch.einsum(  # type: ignore
            'ba,baf->bf', data['value_mask'].to(self.dtype), trans_invariats)  # (batches, inv_feats)
        value = self.phi_v(value_feats).squeeze(-1)  # (batches, )

        # Action
        response: Dict[str, Any] = {}
        if actions is None:
            actions = torch.cat([focus.float(), element.float(), distance, orientation], dim=-1)

            # Build correspond action in action space
            response['actions'] = [self.to_action_space(a, o) for a, o in zip(actions, observations)]

        response.update({
            'a': actions,  # (batches, subactions)
            'logp': log_prob,  # (batches, )
            'ent': entropy,  # (batches, )
            'v': value,  # (batches, )
            'dists': [focus_dist, element_dist, distance_dist, so3_dist],
        })

        return response
