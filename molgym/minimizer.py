from typing import Tuple

import numpy as np
import scipy.optimize
from ase import Atoms


def minimize(
    calculator,
    atoms: Atoms,
    charge: int,
    spin_multiplicity: int,
    max_iter=120,
    fixed_indices=None,
    verbose=False,
) -> Tuple[Atoms, bool]:
    atoms = atoms.copy()
    calculator.set_elements(list(atoms.symbols))
    calculator.set_settings({'molecular_charge': charge, 'spin_multiplicity': spin_multiplicity})

    mask = np.ones((len(atoms) * 3, ), dtype=np.float)
    if fixed_indices:
        for index in fixed_indices:
            mask[index * 3:(index + 1) * 3] = 0

    def function(coords: np.ndarray) -> Tuple[float, np.ndarray]:
        calculator.set_positions(coords.reshape(-1, 3))
        energy = calculator.calculate_energy()
        gradients = calculator.calculate_gradients()
        return energy, gradients.flatten() * mask

    initial_coords = atoms.positions.flatten()

    minimize_result = scipy.optimize.minimize(
        function,
        x0=initial_coords,
        jac=True,
        method='BFGS',
        options={
            'maxiter': max_iter,
            'disp': verbose,
            'norm': np.inf,  # equivalent to taking numpy.amax(numpy.abs(gradient))
            'gtol': 3e-4,  # TolMaxG=3e-4 (ORCA)
        },
    )

    atoms.positions = minimize_result.x.reshape(-1, 3)

    return atoms, minimize_result.success
