from typing import Tuple

import torch
from cormorant.cg_lib import CGModule, SphericalHarmonicsRel, CGProduct
from cormorant.models.cormorant_cg import CormorantCG
from cormorant.models.cormorant_qm9 import expand_var_list
from cormorant.nn import NoLayer, RadialFilters, CatMixReps, InputLinear
from cormorant.so3_lib import SO3Vec


class Cormorant(CGModule):
    def __init__(
        self,
        maxl,
        max_sh,
        num_cg_levels,
        num_channels,
        num_species,
        cutoff_type,
        hard_cut_rad,
        soft_cut_rad,
        soft_cut_width,
        weight_init,
        level_gain,
        charge_power,
        basis_set,
        charge_scale,
        bag_scale,
        device=None,
        dtype=None,
        cg_dict=None,
    ) -> None:
        # Parameters
        level_gain = expand_var_list(level_gain, num_cg_levels)
        hard_cut_rad = expand_var_list(hard_cut_rad, num_cg_levels)
        soft_cut_rad = expand_var_list(soft_cut_rad, num_cg_levels)
        soft_cut_width = expand_var_list(soft_cut_width, num_cg_levels)
        maxl = expand_var_list(maxl, num_cg_levels)
        max_sh = expand_var_list(max_sh, num_cg_levels)
        num_channels = expand_var_list(num_channels, num_cg_levels + 1)

        super().__init__(maxl=max(maxl + max_sh), device=device, dtype=dtype, cg_dict=cg_dict)

        self.num_cg_levels = num_cg_levels
        self.num_channels = num_channels
        self.charge_power = charge_power
        self.charge_scale = charge_scale
        self.bag_scale = bag_scale
        self.num_species = num_species

        # Set up spherical harmonics
        self.sph_harms = SphericalHarmonicsRel(maxl=max(max_sh),
                                               conj=True,
                                               device=self.device,
                                               dtype=self.dtype,
                                               cg_dict=self.cg_dict)

        # Set up position functions, now independent of spherical harmonics
        self.rad_funcs = RadialFilters(
            max_sh=max_sh,
            basis_set=basis_set,
            num_channels_out=num_channels,
            num_levels=num_cg_levels,
            device=self.device,
            dtype=self.dtype,
        )
        tau_pos = self.rad_funcs.tau

        num_scalars_in = self.num_species * (self.charge_power + 1) + self.num_species
        num_scalars_out = num_channels[0]

        self.input_func_atom = InputLinear(num_scalars_in, num_scalars_out, device=self.device, dtype=self.dtype)
        self.input_func_edge = NoLayer()

        tau_in_atom = self.input_func_atom.tau
        tau_in_edge = self.input_func_edge.tau

        self.cormorant_cg = CormorantCG(maxl=maxl,
                                        max_sh=max_sh,
                                        tau_in_atom=tau_in_atom,
                                        tau_in_edge=tau_in_edge,
                                        tau_pos=tau_pos,
                                        num_cg_levels=num_cg_levels,
                                        num_channels=num_channels,
                                        level_gain=level_gain,
                                        weight_init=weight_init,
                                        cutoff_type=cutoff_type,
                                        hard_cut_rad=hard_cut_rad,
                                        soft_cut_rad=soft_cut_rad,
                                        soft_cut_width=soft_cut_width,
                                        cat=True,
                                        gaussian_mask=False,
                                        device=self.device,
                                        dtype=self.dtype,
                                        cg_dict=self.cg_dict)

    def forward(self, data) -> SO3Vec:
        # Get and prepare the data
        atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions = self.prepare_input(data)

        # Calculate spherical harmonics and radial functions
        spherical_harmonics, norms = self.sph_harms(atom_positions, atom_positions)
        rad_func_levels = self.rad_funcs(norms, edge_mask * (norms > 0))

        # Prepare the input reps for both the atom and edge network
        atom_reps_in = self.input_func_atom(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)
        edge_net_in = self.input_func_edge(atom_scalars, atom_mask, edge_scalars, edge_mask, norms)

        # Clebsch-Gordan layers central to the network
        atoms_all, edges_all = self.cormorant_cg(atom_reps_in, atom_mask, edge_net_in, edge_mask, rad_func_levels,
                                                 norms, spherical_harmonics)

        # Return last atomic layer
        return atoms_all[-1]

    def prepare_input(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_positions = data['positions'].to(self.device, self.dtype)
        one_hot = data['one_hot'].to(self.device, self.dtype)
        charges = data['charges'].to(self.device, self.dtype)

        atom_mask = data['atom_mask'].to(self.device)
        edge_mask = data['edge_mask'].to(self.device)

        charge_tensor = (charges.unsqueeze(-1) / self.charge_scale).pow(
            torch.arange(self.charge_power + 1, device=self.device, dtype=self.dtype))
        charge_tensor = charge_tensor.view(charges.shape + (1, self.charge_power + 1))
        charge_tensor = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1, ))

        bag_tiled = (data['bags'] / self.bag_scale).unsqueeze(1)  # (batches, 1, feats)
        bag_tiled = bag_tiled.expand(charge_tensor.shape[:-1] + (-1, ))  # (batches, atoms, feats)
        atom_scalars = torch.cat([charge_tensor, bag_tiled], dim=-1)

        edge_scalars = torch.tensor([])

        return atom_scalars, atom_mask, edge_scalars, edge_mask, atom_positions


class CormorantMixer(CGModule):
    def __init__(self,
                 tau_in,
                 tau_other,
                 maxl,
                 num_channels,
                 level_gain,
                 weight_init,
                 device=None,
                 dtype=None,
                 cg_dict=None) -> None:
        super().__init__(maxl=maxl, device=device, dtype=dtype, cg_dict=cg_dict)

        self.tau_in = tau_in
        self.tau_other = tau_other

        # Operations linear in input reps
        self.cg_aggregate = CGProduct(self.tau_other,
                                      self.tau_in,
                                      maxl=self.maxl,
                                      device=self.device,
                                      dtype=self.dtype,
                                      cg_dict=self.cg_dict)
        tau_ag = list(self.cg_aggregate.tau)

        self.cg_power = CGProduct(tau_ag,
                                  tau_ag,
                                  maxl=self.maxl,
                                  device=self.device,
                                  dtype=self.dtype,
                                  cg_dict=self.cg_dict)
        tau_sq = list(self.cg_power.tau)

        self.cat_mix = CatMixReps([tau_ag, tau_sq, self.tau_in],
                                  num_channels,
                                  maxl=self.maxl,
                                  weight_init=weight_init,
                                  gain=level_gain,
                                  device=self.device,
                                  dtype=self.dtype)
        self.tau = self.cat_mix.tau

    def forward(self, atom_reps, other_reps):
        # Aggregate information based upon other reps
        reps_ag = self.cg_aggregate(other_reps, atom_reps)

        # CG non-linearity
        reps_sq = self.cg_power(reps_ag, reps_ag)

        # Concatenate and mix results
        reps_out = self.cat_mix([reps_ag, reps_sq, atom_reps])

        return reps_out
