try: # sparrow earlier than v3
    from scine_sparrow import Calculation
    Sparrow = Calculation
    sparrow_v3 = False
except:
    import scine_utilities as su
    import scine_sparrow
    manager = su.core.ModuleManager()
    sparrow_v3 = True

class SparrowCalc:
    """
    Calculation object for sparrow v3 mimicking older
    sparrow API.
    """
    def __init__(self, method):
        self.calc = manager.get('calculator', method)
        self.calc.set_required_properties([su.Property.Energy])
    def set_elements(self, codes):
        elems = []
        for code in codes:
            if isinstance(code, str):
                code = getattr(su.ElementType, code)
            elems.append(code)

        self.structure = su.AtomCollection(len(elems))
        self.structure.elements = elems
    def set_positions(self, crd):
        self.structure.positions = crd
    def set_settings(self, attr):
        """
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
        for k,v in self.calc.settings.items():
            print(k, v)
        for k, v in attr.items():
            if k == 'unrestricted_calculation':
                if v:
                    self.calc.settings["spin_mode"] = "unrestricted"
                else:
                    self.calc.settings["spin_mode"] = "restricted"
                continue
            try:
                self.calc.settings[k] = v
            except RuntimeError as e:
                print(f"Unable to set {k} = {v}: {e}")
    def calculate_energy(self):
        self.calc.structure = self.structure
        return self.calc.calculate().energy

if sparrow_v3:
    Sparrow = SparrowCalc
