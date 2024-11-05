import math
from typing import Optional, List, Tuple

from ttnopt.src.Observable import Observable

class Hamiltonian:
    """A class for Hamiltonian.
    This class is used to store Hamiltonian as a List of Observable.
    """

    def __init__(self,
                 system_size: int,
                 spin_size: List[str],
                 model: str,
                 interaction_indices: List[List[int]],
                 interaction_coefs: List[List[float]],
                 magnetic_field: Optional[List[float]] = None,
                 ion_anisotropy: Optional[List[float]] = None,
                 dzyaloshinskii_moriya: Optional[List[List[float]]] = None,
                ):
        """Initialize a Hamiltonian object.

        Args:
            system_size : The size of the system.
            spin_size : The size of the spin.
            model : The model of the Hamiltonian, "XXZ" or "XYZ".
            interaction_indices : The indices of the interaction.
            interaction_coefs : The coefficients of the interaction.
            magnetic_field : The magnetic field.
            ion_anisotropy : The ion anisotropy.
            dzyaloshinskii_moriya : The Dzyaloshinskii-Moriya interaction.
        """
        self.system_size = system_size
        self.spin_size = {i: spin_size[i] for i in range(self.system_size)}
        self.observables = []

        if model == "XXZ":
            for i, coef in zip(interaction_indices, interaction_coefs):
                if all([c == 0.0 for c in coef]):
                    continue
                operator_list = []
                coef_list = []
                if not math.isclose(coef[0], 0.0):
                    operator_list.append(["S+", "S-"])
                    coef_list.append(coef[0] / 2.0)
                    operator_list.append(["S-", "S+"])
                    coef_list.append(coef[0] / 2.0)
                if not math.isclose(coef[1], 0.0):
                    operator_list.append(["Sz", "Sz"])
                    coef_list.append(coef[1])
                ob = Observable(i, operator_list, coef_list)
                self.observables.append(ob)
        if model == "XYZ":
            for i, coef in zip(interaction_indices, interaction_coefs):
                if all([c == 0.0 for c in coef]):
                    continue
                operator_list = []
                coef_list = []
                if not math.isclose(coef[0], 0.0):
                    operator_list.append(["Sx", "Sx"])
                    coef_list.append(coef[0])
                if not math.isclose(coef[1], 0.0):
                    operator_list.append(["Sy", "Sy"])
                    coef_list.append(coef[1])
                if not math.isclose(coef[2], 0.0):
                    operator_list.append(["Sz", "Sz"])
                    coef_list.append(coef[2])
                ob = Observable(i, operator_list, coef_list)
                self.observables.append(ob)

        if magnetic_field is not None:
            for idx, c in enumerate(magnetic_field):
                if not math.isclose(c, 0.0):
                    ob = Observable([idx], ["Sx"], [-c])
                    self.observables.append(ob)

        if ion_anisotropy is not None:
            for idx, c in enumerate(ion_anisotropy):
                if not math.isclose(c, 0.0):
                    ob = Observable([idx], [["Sz^2"]], [-c])
                    self.observables.append(ob)

        if dzyaloshinskii_moriya is not None:
            for i, coefs in zip(interaction_indices, dzyaloshinskii_moriya):
                if all([c == 0.0 for c in coefs]):
                    continue
                if not math.isclose(coef[0], 0.0):
                    ob = Observable(i, [["Sx", "Sy"], ["Sy", "Sx"]], [coefs[0] - coefs[0]])
                    self.observables.append(ob)
                if not math.isclose(coef[1], 0.0):
                    ob = Observable(i, [["Sy", "Sz"], ["Sz", "Sy"]], [coefs[1] - coefs[1]])
                    self.observables.append(ob)
                if not math.isclose(coef[2], 0.0):
                    ob = Observable(i, [["Sz", "Sx"], ["Sx", "Sz"]], [coefs[2] - coefs[2]])
                    self.observables.append(ob)

