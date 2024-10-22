
from ttnopt.src import Hamiltonian

from typing import Dict
import numpy as np
import pandas as pd
from dotmap import DotMap

def hamiltonian(config: Dict):
    """_summary_

    Args:
        config Dict: the configuration of the system
    """
    if config.spin_size.uniform == 1:
        # Ensure that config.spin_size.S is set and valid before using it
        if isinstance(config.spin_size.S, DotMap):
            raise ValueError("Please input S value in spin_size")

        spin_sizes = ["S=" + str(config.spin_size.S)] * config.N
    else:
        # Ensure that config.spin_size.S is set and valid before using it
        if isinstance(config.spin_size.Si_file, DotMap):
            raise ValueError("Please input Si_file in spin_size")

        spin_sizes = ["S=" + str(config.spin_size.S)] * config.system.N

    interaction_indices = pd.read_csv(config.model.ij_file, delimiter=",", header=None, dtype=int).values
    interaction_coefs = pd.read_csv(config.model.Jij_file, delimiter=",", header=None, dtype=float).values

    if interaction_indices.max() >= config.N:
        raise ValueError(f"Interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N}).")
    if interaction_indices.shape[0] != interaction_coefs.shape[0]:
        raise ValueError("Number of rows in Jij_file must match number of rows in ij_file")

    # magnetic_field
    if config.magnetic_field.active == 1:
        if config.magnetic_field.uniform == 1:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.magnetic_field.h, DotMap):
                raise ValueError("Please input h value in magnetic_field if uniform is 1")

            magnetic_field = [config.magnetic_field.h] * config.N

        else:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.magnetic_field.hi_file, DotMap):
                raise ValueError("Please input hi_file in magnetic_field if uniform is 0")

            magnetic_field = pd.read_csv(config.magnetic_field.hi_file, delimiter=",", header=None, dtype=float).values
    else:
        magnetic_field = None

    # ion_anisotropy
    if config.ion_anisotropy.active == 1:
        if config.ion_anisotropy.uniform == 1:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.ion_anisotropy.D, DotMap):
                raise ValueError("Please input D value in ion_anisotropy if uniform is 1")

            ion_anisotropy = [config.ion_anisotropy.D] * config.N

        else:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.ion_anisotropy.Di_file, DotMap):
                raise ValueError("Please input Di_file in ion_anisotropy if uniform is 0")

            ion_anisotropy = pd.read_csv(config.ion_anisotropy.Di_file, delimiter=",", header=None, dtype=float).values
    else:
        ion_anisotropy = None

    # dzyaloshinskii_moriya
    if config.dzyaloshinskii_moriya.active == 1:
        if isinstance(config.dzyaloshinskii_moriya.DM_file, DotMap):
            raise ValueError("Please input Di_file in dzyaloshinskii_moriya")
        dzyaloshinskii_moriya = pd.read_csv(config.dzyaloshinskii_moriya.DM_file, delimiter=",", header=None, dtype=float).values
        if interaction_indices.shape[0] != dzyaloshinskii_moriya.shape[0]:
            raise ValueError("Number of rows in DM_file must match number of rows in ij_file")
    else:
        dzyaloshinskii_moriya = None

    if config.model.type == "XXZ":
        if interaction_coefs.shape[1] == 2:
            hamiltonian = Hamiltonian(
                config.N,
                spin_sizes,
                config.model.type,
                interaction_indices,
                interaction_coefs,
                magnetic_field=magnetic_field,
                ion_anisotropy=ion_anisotropy,
                dzyaloshinskii_moriya=dzyaloshinskii_moriya,
            )
        else:
            raise ValueError("Please input two columns in Jij_file for XXZ model")
    elif config.model.type == "XYZ":
        if interaction_coefs.shape[1] == 3:
            hamiltonian = Hamiltonian(
                config.N,
                spin_sizes,
                config.model.type,
                interaction_indices,
                interaction_coefs,
                magnetic_field=magnetic_field,
                ion_anisotropy=ion_anisotropy,
                dzyaloshinskii_moriya=dzyaloshinskii_moriya,
            )
        else:
            raise ValueError("Please input two columns in Jij_file for XYZ model")
    else:
        raise ValueError("Please select XXZ or XYZ model within hamiltonian")

    return hamiltonian





