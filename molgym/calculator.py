# Try loading energy computation backends.
# For each one that is successful, set its entry in `calculators`.

import numpy as np

calculators = {"sparrow_v2": None, "sparrow_v3": None}


class SparrowCalc:
    """
    Calculation object for sparrow v3.
    """

    def __init__(self, method):
        self.calc = manager.get("calculator", method)
        self.calc.set_required_properties([su.Property.Energy, su.Property.Gradients])
        self.elements = None
        self.positions = None

    def set_elements(self, codes):
        elems = []
        for code in codes:
            if isinstance(code, str):
                code = getattr(su.ElementType, code)
            elems.append(code)

        self.elements = elems

    def set_positions(self, crd):
        self.positions = np.array(crd) * su.BOHR_PER_ANGSTROM

    def set_settings(self, attr):
        """
        This routine will be called with `attr`:

        { 'unrestricted_calculation' : int
          'spin_multiplicity' : int
        }

        Available attributes in self.calc.settings:

            molecular_charge 0
            spin_multiplicity 1
            spin_mode any
            temperature 298.15
            electronic_temperature 0.0
            symmetry_number 1
            self_consistence_criterion 1e-07
            density_rmsd_criterion 1e-05
            max_scf_iterations 100
            scf_mixer diis
            method_parameters
            nddo_dipole True
        """
        # for k,v in self.calc.settings.items():
        #    print(k, v)
        for k, v in attr.items():
            if k == "unrestricted_calculation":
                if v:
                    self.calc.settings["spin_mode"] = "unrestricted"
                else:
                    self.calc.settings["spin_mode"] = "restricted"
                continue
            try:
                self.calc.settings[k] = v
            except RuntimeError as e:
                print(f"Unable to set {k} = {v}: {e}")

    def _structure(self):
        structure = su.AtomCollection(len(self.elements))
        structure.elements = self.elements
        structure.positions = self.positions
        return structure

    def calculate_energy(self):
        self.calc.structure = self._structure()
        return self.calc.calculate().energy

    def calculate_gradients(self):
        self.calc.structure = self._structure()
        return self.calc.calculate().gradients


try:  # try sparrow v2
    from scine_sparrow import Calculation

    calculators["sparrow_v2"] = Calculation
except:  # try sparrow v3
    import scine_utilities as su
    import scine_sparrow

    manager = su.core.ModuleManager()
    calculators["sparrow_v3"] = SparrowCalc

# Use the first loaded backend.
for k, v in calculators.items():
    if v is not None:
        calculator = k
        Sparrow = v
        break
