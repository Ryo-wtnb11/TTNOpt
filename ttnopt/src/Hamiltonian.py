import math
import numpy as np

from typing import Optional, List
from ttnopt.src.Observable import Observable


class Hamiltonian:
    """A class for Hamiltonian.
    This class is used to store Hamiltonian as a List of Observable.
    """

    def __init__(
        self,
        system_size: int,
        spin_size: List[str],
        model: str,
        interaction_indices: List[List[int]],
        interaction_coefs: List[List[float]],
        magnetic_field_X_indices: Optional[List[int]] = None,
        magnetic_field_X: Optional[List[float]] = None,
        magnetic_field_Y_indices: Optional[List[int]] = None,
        magnetic_field_Y: Optional[List[float]] = None,
        magnetic_field_Z_indices: Optional[List[int]] = None,
        magnetic_field_Z: Optional[List[float]] = None,
        ion_anisotropy_indices: Optional[List[int]] = None,
        ion_anisotropy: Optional[List[float]] = None,
        dzyaloshinskii_moriya_X_indices: Optional[List[List[int]]] = None,
        dzyaloshinskii_moriya_X: Optional[List[float]] = None,
        dzyaloshinskii_moriya_Y_indices: Optional[List[List[int]]] = None,
        dzyaloshinskii_moriya_Y: Optional[List[float]] = None,
        dzyaloshinskii_moriya_Z_indices: Optional[List[List[int]]] = None,
        dzyaloshinskii_moriya_Z: Optional[List[float]] = None,
        sod_X_indices: Optional[List[List[int]]] = None,
        sod_X: Optional[List[float]] = None,
        sod_Y_indices: Optional[List[List[int]]] = None,
        sod_Y: Optional[List[float]] = None,
        sod_Z_indices: Optional[List[List[int]]] = None,
        sod_Z: Optional[List[float]] = None,
    ):
        """
        Args:
            system_size (int): The size of the system, indicating the number of sites or particles.
            spin_size (List[str]): A list of spin types or values, each corresponding to the spin state of each site in the system.
            model (str): The type of Hamiltonian model, which determines the specific interactions and parameters used (e.g., "Heisenberg", "Ising").
            interaction_indices (List[List[int]]): A nested list of integer pairs, where each sub-list specifies two indices corresponding
                to sites or particles between which an interaction occurs.
            interaction_coefs (List[List[float]]): A nested list where each sub-list contains the coefficients for the interactions
                specified in `interaction_indices`, defining the strength and nature of each interaction.
            magnetic_field_indices (Optional[List[int]], optional): A list of site indices at which the magnetic field is applied.
                If None, no magnetic field is applied to any specific site. Defaults to None.
            magnetic_field (Optional[List[float]], optional): A list of magnetic field strengths, where each value corresponds
                to a specific site defined in `magnetic_field_indices`. Defaults to None.
            magnetic_field_axis (Optional[str], optional): The axis along which the magnetic field is applied, specified as a string
                ("x", "y", or "z"). Defaults to None.
            ion_anisotropy_indices (Optional[List[int]], optional): A list of indices specifying the sites with single-ion anisotropy.
                Defaults to None.
            ion_anisotropy (Optional[List[float]], optional): A list of single-ion anisotropy values corresponding to each site specified
                in `ion_anisotropy_indices`. Defaults to None.
            dzyaloshinskii_moriya_indices (Optional[List[List[int]]], optional): A nested list of integer pairs, where each sub-list specifies
                two indices corresponding to sites with Dzyaloshinskii-Moriya (DM) interaction. Defaults to None.
            dzyaloshinskii_moriya (Optional[List[List[float]]], optional): A list of DM interaction strengths, where each value corresponds
                to the pair specified in `dzyaloshinskii_moriya_indices`. Defaults to None.
            sod_indices (Optional[List[List[int]]], optional): A nested list of integer pairs, where each sub-list specifies two indices
                two indices corresponding to sites with second-order symmetric off-diagonal anisotropy interaction. Defaults to None.
            sod (Optional[List[List[float]]], optional): A list of SOD interaction strengths, where each value corresponds
                to the pair specified in `sod_indices`. Defaults to None.
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
                if not coef[0] == 0.0:
                    operator_list.append(["S+", "S-"])
                    coef_list.append(coef[0] / 2.0)
                    operator_list.append(["S-", "S+"])
                    coef_list.append(coef[0] / 2.0)
                if not coef[0] == 0.0 and not coef[1] == 0.0:
                    operator_list.append(["Sz", "Sz"])
                    coef_list.append(coef[0] * coef[1])
                ob = Observable(i, operator_list, coef_list)
                self.observables.append(ob)
        if model == "XYZ":
            for i, coef in zip(interaction_indices, interaction_coefs):
                if all([c == 0.0 for c in coef]):
                    continue
                operator_list = []
                coef_list = []
                if not coef[0] == 0.0:
                    operator_list.append(["Sx", "Sx"])
                    coef_list.append(coef[0])
                if not coef[1] == 0.0:
                    operator_list.append(["Sy", "Sy"])
                    coef_list.append(coef[1])
                if not coef[2] == 0.0:
                    operator_list.append(["Sz", "Sz"])
                    coef_list.append(coef[2])
                ob = Observable(i, operator_list, coef_list)
                self.observables.append(ob)

        if magnetic_field_X is not None and magnetic_field_X_indices is not None:
            for idx, c in zip(magnetic_field_X_indices, magnetic_field_X):
                if not c == 0.0:
                    ob = Observable([idx], [["Sx"]], [-c])
                    self.observables.append(ob)

        if magnetic_field_Y is not None and magnetic_field_Y_indices is not None:
            for idx, c in zip(magnetic_field_Y_indices, magnetic_field_Y):
                if not c == 0.0:
                    ob = Observable([idx], [["Sy"]], [-c])
                    self.observables.append(ob)

        if magnetic_field_Z is not None and magnetic_field_Z_indices is not None:
            for idx, c in zip(magnetic_field_Z_indices, magnetic_field_Z):
                if not c == 0.0:
                    ob = Observable([idx], [["Sz"]], [-c])
                    self.observables.append(ob)

        if ion_anisotropy is not None and ion_anisotropy_indices is not None:
            for idx, c in zip(ion_anisotropy_indices, ion_anisotropy):
                if not c == 0.0:
                    ob = Observable([idx], [["Sz2"]], [c])
                    self.observables.append(ob)

        if (
            dzyaloshinskii_moriya_X is not None
            and dzyaloshinskii_moriya_X_indices is not None
        ):
            for i, c in zip(dzyaloshinskii_moriya_X_indices, dzyaloshinskii_moriya_X):
                if not c == 0.0:
                    ob = Observable(i, [["Sy", "Sz"], ["Sz", "Sy"]], [c, -c])
                    self.observables.append(ob)

        if (
            dzyaloshinskii_moriya_Y is not None
            and dzyaloshinskii_moriya_Y_indices is not None
        ):
            for i, c in zip(dzyaloshinskii_moriya_Y_indices, dzyaloshinskii_moriya_Y):
                if not c == 0.0:
                    ob = Observable(i, [["Sz", "Sx"], ["Sx", "Sz"]], [c, -c])
                    self.observables.append(ob)

        if (
            dzyaloshinskii_moriya_Z is not None
            and dzyaloshinskii_moriya_Z_indices is not None
        ):
            for i, c in zip(dzyaloshinskii_moriya_Z_indices, dzyaloshinskii_moriya_Z):
                if not c == 0.0:
                    cc = np.array([c / 2.0j])
                    ob = Observable(
                        i,
                        [["S-", "S+"], ["S+", "S-"]],
                        [cc.item(), -cc.item()],
                    )
                    self.observables.append(ob)

        if sod_X is not None and sod_X_indices is not None:
            for i, c in zip(sod_X_indices, sod_X):
                if not c == 0.0:
                    ob = Observable(
                        i,
                        [["Sy", "Sz"], ["Sz", "Sy"]],
                        [c, c],
                    )
                    self.observables.append(ob)

        if sod_Y is not None and sod_Y_indices is not None:
            for i, c in zip(sod_Y_indices, sod_Y):
                if not c == 0.0:
                    ob = Observable(
                        i,
                        [["Sz", "Sx"], ["Sx", "Sz"]],
                        [c, c],
                    )
                    self.observables.append(ob)

        if sod_Z is not None and sod_Z_indices is not None:
            for i, c in zip(sod_Z_indices, sod_Z):
                if not c == 0.0:
                    ob = Observable(
                        i,
                        [["Sx", "Sy"], ["Sy", "Sz"]],
                        [c, c],
                    )
                    self.observables.append(ob)
