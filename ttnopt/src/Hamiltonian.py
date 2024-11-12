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
                 magnetic_field_indices: Optional[List[int]] = None,
                 magnetic_field: Optional[List[float]] = None,
                 magnetic_field_axis: Optional[str] = None,
                 ion_anisotropy_indices: Optional[List[int]] = None,
                 ion_anisotropy: Optional[List[float]] = None,
                 dzyaloshinskii_moriya_indices: Optional[List[List[int]]] = None,
                 dzyaloshinskii_moriya: Optional[List[List[float]]] = None,
                 dzyaloshinskii_moriya_axis: Optional[str] = None,
                ):
        """Initialize a Hamiltonian object.

        Args:
            system_size (int): The size of the system.
            spin_size (List[str]): The size of the spin.
            model (str): The model of the Hamiltonian.
            interaction_indices (List[Tuple[int]]): The indices of the interaction.
            interaction_coefs (List[float]): The coefficients of the interaction.
            magnetic_field (Optional[List[float]], optional): The magnetic field axis. Defaults to None.
            magnetic_field_axis: The magnetic field axis. Defaults to None.
            ion_anisotropy (Optional[List[float]], optional): The ion anisotropy. Defaults to None.
            dzyaloshinskii_moriya (Optional[List[float]], optional): The Dzyaloshinskii-Moriya interaction. Defaults to None.
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
                if coef[0] != 0.0:
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
            for idx, c in zip(magnetic_field_indices, magnetic_field):
                if c != 0.0:
                    if magnetic_field_axis == "X":
                        ob = Observable([idx], [["Sx"]], [-c])
                    elif magnetic_field_axis == "Y":
                        ob = Observable([idx], [["Sy"]], [-c])
                    elif magnetic_field_axis == "Z":
                        ob = Observable([idx], [["Sz"]], [-c])
                    self.observables.append(ob)

        if ion_anisotropy is not None:
            for idx, c in zip(ion_anisotropy_indices, ion_anisotropy):
                if not math.isclose(c, 0.0):
                    ob = Observable([idx], [["Sz^2"]], [-c])
                    self.observables.append(ob)

        if dzyaloshinskii_moriya is not None:
            for idx, c in zip(dzyaloshinskii_moriya_indices, dzyaloshinskii_moriya):
                if c != 0.0:
                    if dzyaloshinskii_moriya_axis == "X":
                        ob = Observable(i, [["Sx", "Sy"], ["Sy", "Sx"]], [c, -c])
                        self.observables.append(ob)
                    if dzyaloshinskii_moriya_axis == "Y":
                        ob = Observable(i, [["Sy", "Sz"], ["Sz", "Sy"]], [c, -c])
                        self.observables.append(ob)
                    if dzyaloshinskii_moriya_axis == "Z":
                        ob = Observable(i, [["Sz", "Sx"], ["Sx", "Sz"]], [c, -c])
                        self.observables.append(ob)

