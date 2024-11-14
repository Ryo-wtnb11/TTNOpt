
from ttnopt.src import Hamiltonian

from typing import Dict
import numpy as np
import pandas as pd
from dotmap import DotMap

def hamiltonian(config: DotMap):
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

        spin_sizes = ["S=" + s for s in pd.read_csv(config.spin_size.Si_file, delimiter=",", header=None).values.flatten()]

    interaction_csv = pd.read_csv(config.model.Jij_file, delimiter=",", header=None)
    interaction_indices = interaction_csv.iloc[:, :2].values
    interaction_coefs = interaction_csv.iloc[:, 2:].values

    if interaction_indices.max() >= config.N:
        raise ValueError(f"Interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N}).")
    if interaction_indices.shape[0] != interaction_coefs.shape[0]:
        raise ValueError("Number of rows in Jij_file must match number of rows in ij_file")

    # magnetic_field_X
    if config.MF_X.active == 1:
        if config.MF_X.uniform == 1:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.MF_X.h, DotMap):
                raise ValueError("Please input h value in magnetic_field if uniform is 1")
            magnetic_field_X_indices = [i for i in range(config.N)]
            magnetic_field_X = [config.MF_X.h] * config.N
        else:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.MF_X.hi_file, DotMap):
                raise ValueError("Please input hi_file in magnetic_field if uniform is 0")
            magnetic_field_X_csv = pd.read_csv(config.MF_X.hi_file, delimiter=",", header=None)
            magnetic_field_X_indices = magnetic_field_X_csv.iloc[:, 0].values
            magnetic_field_X = magnetic_field_X_csv.iloc[:, 1].values
    else:
        magnetic_field_X_indices = None
        magnetic_field_X = None

    # magnetic_field_Y
    if config.MF_Y.active == 1:
        if config.MF_Y.uniform == 1:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.MF_Y.h, DotMap):
                raise ValueError("Please input h value in magnetic_field if uniform is 1")
            magnetic_field_Y_indices = [i for i in range(config.N)]
            magnetic_field_Y = [config.MF_Y.h] * config.N
        else:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.MF_Y.hi_file, DotMap):
                raise ValueError("Please input hi_file in magnetic_field if uniform is 0")
            magnetic_field_Y_csv = pd.read_csv(config.MF_Y.hi_file, delimiter=",", header=None)
            magnetic_field_Y_indices = magnetic_field_Y_csv.iloc[:, 0].values
            magnetic_field_Y = magnetic_field_Y_csv.iloc[:, 1].values
    else:
        magnetic_field_Y_indices = None
        magnetic_field_Y = None

    # magnetic_field_Z
    if config.MF_Z.active == 1:
        if config.MF_Z.uniform == 1:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.MF_Z.h, DotMap):
                raise ValueError("Please input h value in magnetic_field if uniform is 1")
            magnetic_field_Z_indices = [i for i in range(config.N)]
            magnetic_field_Z = [config.MF_Z.h] * config.N
        else:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.MF_Z.hi_file, DotMap):
                raise ValueError("Please input hi_file in magnetic_field if uniform is 0")
            magnetic_field_Z_csv = pd.read_csv(config.MF_Z.hi_file, delimiter=",", header=None)
            magnetic_field_Z_indices = magnetic_field_Z_csv.iloc[:, 0].values
            magnetic_field_Z = magnetic_field_Z_csv.iloc[:, 1].values
    else:
        magnetic_field_Z_indices = None
        magnetic_field_Z = None


    # ion_anisotropy
    if config.SIA.active == 1:
        if config.SIA.uniform == 1:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.SIA.D, DotMap):
                raise ValueError("Please input D value in ion_anisotropy if uniform is 1")
            ion_anisotropy_indices = [i for i in range(config.N)]
            ion_anisotropy = [config.SIA.D] * config.N

        else:
            # Ensure that config.spin_size.S is set and valid before using it
            if isinstance(config.SIA.Di_file, DotMap):
                raise ValueError("Please input Di_file in ion_anisotropy if uniform is 0")
            ion_anisotropy_csv = pd.read_csv(config.SIA.Di_file, delimiter=",", header=None)
            ion_anisotropy_indices = ion_anisotropy_csv.iloc[:, 0].values
            ion_anisotropy = ion_anisotropy_csv.iloc[:, 1].values
    else:
        ion_anisotropy = None
        ion_anisotropy_indices = None

    # dzyaloshinskii_moriya_X
    if config.DM_X.active == 1:
        if isinstance(config.DM_X.DM_file, DotMap):
            raise ValueError("Please input Di_file in dzyaloshinskii_moriya")
        dzyaloshinskii_moriya_X_csv = pd.read_csv(config.DM_X.DM_file, delimiter=",", header=None)
        dzyaloshinskii_moriya_X_indices = dzyaloshinskii_moriya_X_csv.iloc[:, :2].values
        dzyaloshinskii_moriya_X = dzyaloshinskii_moriya_X_csv.iloc[:, 2:].values

        if dzyaloshinskii_moriya_X_indices.max() >= config.N:
            raise ValueError(f"Interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N}).")
        if dzyaloshinskii_moriya_X_indices.shape[0] != dzyaloshinskii_moriya_X.shape[0]:
            raise ValueError("Number of rows in Jij_file must match number of rows in ij_file")
    else:
        dzyaloshinskii_moriya_X_indices = None
        dzyaloshinskii_moriya_X = None

    # dzyaloshinskii_moriya
    if config.DM_Y.active == 1:
        if isinstance(config.DM_Y.DM_file, DotMap):
            raise ValueError("Please input Di_file in dzyaloshinskii_moriya")
        dzyaloshinskii_moriya_Y_csv = pd.read_csv(config.DM_Y.DM_file, delimiter=",", header=None)
        dzyaloshinskii_moriya_Y_indices = dzyaloshinskii_moriya_Y_csv.iloc[:, :2].values
        dzyaloshinskii_moriya_Y = dzyaloshinskii_moriya_Y_csv.iloc[:, 2:].values

        if dzyaloshinskii_moriya_Y_indices.max() >= config.N:
            raise ValueError(f"Interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N}).")
        if dzyaloshinskii_moriya_Y_indices.shape[0] != dzyaloshinskii_moriya_Y.shape[0]:
            raise ValueError("Number of rows in Jij_file must match number of rows in ij_file")
    else:
        dzyaloshinskii_moriya_Y_indices = None
        dzyaloshinskii_moriya_Y = None

    # dzyaloshinskii_moriya
    if config.DM_Z.active == 1:
        if isinstance(config.DM_Z.DM_file, DotMap):
            raise ValueError("Please input Di_file in dzyaloshinskii_moriya")
        dzyaloshinskii_moriya_Z_csv = pd.read_csv(config.DM_Z.DM_file, delimiter=",", header=None)
        dzyaloshinskii_moriya_Z_indices = dzyaloshinskii_moriya_Z_csv.iloc[:, :2].values
        dzyaloshinskii_moriya_Z = dzyaloshinskii_moriya_Z_csv.iloc[:, 2:].values

        if dzyaloshinskii_moriya_Z_indices.max() >= config.N:
            raise ValueError(f"Interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N}).")
        if dzyaloshinskii_moriya_Z_indices.shape[0] != dzyaloshinskii_moriya_Z.shape[0]:
            raise ValueError("Number of rows in Jij_file must match number of rows in ij_file")
    else:
        dzyaloshinskii_moriya_Z_indices = None
        dzyaloshinskii_moriya_Z = None

    if config.model.type == "XXZ":
        if interaction_coefs.shape[1] == 2:
            hamiltonian = Hamiltonian(
                config.N,
                spin_sizes,
                config.model.type,
                interaction_indices,
                interaction_coefs,
                magnetic_field_X_indices=magnetic_field_X_indices,
                magnetic_field_X=magnetic_field_X,
                magnetic_field_Y_indices=magnetic_field_Y_indices,
                magnetic_field_Y=magnetic_field_Y,
                magnetic_field_Z_indices=magnetic_field_Z_indices,
                magnetic_field_Z=magnetic_field_Z,
                ion_anisotropy_indices=ion_anisotropy_indices,
                ion_anisotropy=ion_anisotropy,
                dzyaloshinskii_moriya_X_indices=dzyaloshinskii_moriya_X_indices,
                dzyaloshinskii_moriya_X=dzyaloshinskii_moriya_X,
                dzyaloshinskii_moriya_Y_indices=dzyaloshinskii_moriya_Y_indices,
                dzyaloshinskii_moriya_Y=dzyaloshinskii_moriya_Y,
                dzyaloshinskii_moriya_Z_indices=dzyaloshinskii_moriya_Z_indices,
                dzyaloshinskii_moriya_Z=dzyaloshinskii_moriya_Z,
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
                magnetic_field_X_indices=magnetic_field_X_indices,
                magnetic_field_X=magnetic_field_X,
                magnetic_field_Y_indices=magnetic_field_Y_indices,
                magnetic_field_Y=magnetic_field_Y,
                magnetic_field_Z_indices=magnetic_field_Z_indices,
                magnetic_field_Z=magnetic_field_Z,
                ion_anisotropy_indices=ion_anisotropy_indices,
                ion_anisotropy=ion_anisotropy,
                dzyaloshinskii_moriya_X_indices=dzyaloshinskii_moriya_X_indices,
                dzyaloshinskii_moriya_X=dzyaloshinskii_moriya_X,
                dzyaloshinskii_moriya_Y_indices=dzyaloshinskii_moriya_Y_indices,
                dzyaloshinskii_moriya_Y=dzyaloshinskii_moriya_Y,
                dzyaloshinskii_moriya_Z_indices=dzyaloshinskii_moriya_Z_indices,
                dzyaloshinskii_moriya_Z=dzyaloshinskii_moriya_Z,
            )
        else:
            raise ValueError("Please input two columns in Jij_file for XYZ model")
    else:
        raise ValueError("Please select XXZ or XYZ model within hamiltonian")

    return hamiltonian





