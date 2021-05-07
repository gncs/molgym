from typing import List, Dict

import ase.data
import torch
from ase import Atoms


def atoms_to_feats(atoms: Atoms, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        'num_atoms': torch.tensor(len(atoms), dtype=torch.int, device=device),
        'charges': torch.tensor([ase.data.atomic_numbers[atom.symbol] for atom in atoms],
                                dtype=torch.int,
                                device=device),
        'positions': torch.tensor(atoms.positions, dtype=dtype, device=device),
    }


def pad_sequence(sequences: List[torch.Tensor], max_length: int, padding_value=0) -> torch.Tensor:
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (len(sequences), max_length) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)  # type: ignore
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor


def process_atoms_list(atoms_list: List[Atoms], max_num_atoms: int, dtype: torch.dtype,
                       device: torch.device) -> Dict[str, torch.Tensor]:
    # Gather features
    feats_list = [atoms_to_feats(atoms, dtype=dtype, device=device) for atoms in atoms_list]

    # Convert list-of-dicts to dict-of-lists
    props = feats_list[0].keys()
    prop_dict = {prop: [feats[prop] for feats in feats_list] for prop in props}

    # Pad and stack
    molecules = {
        key: pad_sequence(val, max_length=max_num_atoms) if val[0].dim() > 0 else torch.stack(val)
        for key, val in prop_dict.items()
    }

    return molecules
