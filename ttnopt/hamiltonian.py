from ttnopt.src import Hamiltonian
import pandas as pd
from dotmap import DotMap


def hamiltonian(config: DotMap):
    """_summary_

    Args:
        config Dict: the configuration of the system
    """
    # Ensure that config.spin_size.S is set and valid before using it
    if isinstance(config.spin_size.S, DotMap) and isinstance(
        config.spin_size.file, DotMap
    ):
        print("=" * 50)
        print("⚠️  Error: Please input spin value in spin_size as S or Si_file")
        print("=" * 50)
        exit()

    if not isinstance(config.spin_size.S, DotMap):
        spin_sizes = ["S=" + str(config.spin_size.S)] * config.N
    if not isinstance(config.spin_size.file, DotMap):  # Si_file is a string
        if not isinstance(config.spin_size.S, DotMap):
            print("=" * 50)
            print("⚠️  Note: Both S and file are set.")
            print("     Using file setting as the default value for spin size!")
            print("=" * 50)

        spin_csv = pd.read_csv(config.spin_size.file, delimiter=",", header=None)
        spin_csv_sorted = spin_csv.sort_values(by=0)
        spin_size = spin_csv_sorted.iloc[:, 1].values
        spin_sizes = [f"S={s}" for s in spin_size]

    interaction_csv = pd.read_csv(config.model.file, delimiter=",", header=None)
    interaction_indices = interaction_csv.iloc[:, :2].values
    interaction_coefs = interaction_csv.iloc[:, 2:].values

    if interaction_indices.max() >= config.N:
        print("=" * 50)
        print(
            f"⚠️  Error: XXY, XYZ interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
        )
        print("=" * 50)
        exit()

    # magnetic_field_X
    magnetic_field_X_indices = None
    magnetic_field_X = None
    if not isinstance(config.MF_X.h, DotMap):
        magnetic_field_X_indices = [i for i in range(config.N)]
        magnetic_field_X = [config.MF_X.h] * config.N
    if not isinstance(config.MF_X.file, DotMap):
        if not isinstance(config.MF_X.h, DotMap):
            print("=" * 50)
            print("⚠️  Note: Both h and file are set on MF_X.")
            print("     Using file setting as the default value for magnetic field!")
            print("=" * 50)
        magnetic_field_X_csv = pd.read_csv(config.MF_X.file, delimiter=",", header=None)
        magnetic_field_X_indices = magnetic_field_X_csv.iloc[:, 0].values
        magnetic_field_X = magnetic_field_X_csv.iloc[:, 1].values

    # magnetic_field_Y
    magnetic_field_Y_indices = None
    magnetic_field_Y = None
    if not isinstance(config.MF_Y.h, DotMap):
        magnetic_field_Y_indices = [i for i in range(config.N)]
        magnetic_field_Y = [config.MF_Y.h] * config.N
    if not isinstance(config.MF_Y.file, DotMap):
        if not isinstance(config.MF_Y.h, DotMap):
            print("=" * 50)
            print("⚠️  Note: Both h and file are set on MF_Y.")
            print("     Using file setting as the default value for magnetic field!")
            print("=" * 50)
        magnetic_field_Y_csv = pd.read_csv(config.MF_Y.file, delimiter=",", header=None)
        magnetic_field_Y_indices = magnetic_field_Y_csv.iloc[:, 0].values
        magnetic_field_Y = magnetic_field_Y_csv.iloc[:, 1].values

    # magnetic_field_Z
    magnetic_field_Z_indices = None
    magnetic_field_Z = None
    if not isinstance(config.MF_Z.h, DotMap):
        magnetic_field_Z_indices = [i for i in range(config.N)]
        magnetic_field_Z = [config.MF_Z.h] * config.N
    if not isinstance(config.MF_Z.file, DotMap):
        if not isinstance(config.MF_Z.h, DotMap):
            print("=" * 50)
            print("⚠️  Note: Both h and file are set on MF_Z.")
            print("     Using file setting as the default value for magnetic field!")
            print("=" * 50)
        magnetic_field_Z_csv = pd.read_csv(config.MF_Z.file, delimiter=",", header=None)
        magnetic_field_Z_indices = magnetic_field_Z_csv.iloc[:, 0].values
        magnetic_field_Z = magnetic_field_Z_csv.iloc[:, 1].values

    ion_anisotropy_indices = None
    ion_anisotropy = None
    if not isinstance(config.SIA.D, DotMap):
        ion_anisotropy_indices = [i for i in range(config.N)]
        ion_anisotropy = [config.SIA.D] * config.N
    if not isinstance(config.SIA.file, DotMap):
        if not isinstance(config.SIA.D, DotMap):
            print("=" * 50)
            print("⚠️  Note: Both D and file are set on SIA.")
            print(
                "     Using file setting as the default value for single io anisotropy!"
            )
            print("=" * 50)
        ion_anisotropy_csv = pd.read_csv(config.SIA.file, delimiter=",", header=None)
        ion_anisotropy_indices = ion_anisotropy_csv.iloc[:, 0].values
        ion_anisotropy = ion_anisotropy_csv.iloc[:, 1].values

    # dzyaloshinskii_moriya_X
    dzyaloshinskii_moriya_X_indices = None
    dzyaloshinskii_moriya_X = None
    if not isinstance(config.DM_X.file, DotMap):
        dzyaloshinskii_moriya_X_csv = pd.read_csv(
            config.DM_X.file, delimiter=",", header=None
        )
        dzyaloshinskii_moriya_X_indices = dzyaloshinskii_moriya_X_csv.iloc[:, :2].values
        dzyaloshinskii_moriya_X = dzyaloshinskii_moriya_X_csv.iloc[:, 2:].values
        if dzyaloshinskii_moriya_X_indices.max() >= config.N:
            print("=" * 50)
            print(
                f"⚠️  Error: DM_X interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
            )
            print("=" * 50)
            exit()

    # dzyaloshinskii_moriya_Y
    dzyaloshinskii_moriya_Y_indices = None
    dzyaloshinskii_moriya_Y = None
    if not isinstance(config.DM_Y.file, DotMap):
        dzyaloshinskii_moriya_Y_csv = pd.read_csv(
            config.DM_Y.file, delimiter=",", header=None
        )
        dzyaloshinskii_moriya_Y_indices = dzyaloshinskii_moriya_Y_csv.iloc[:, :2].values
        dzyaloshinskii_moriya_Y = dzyaloshinskii_moriya_Y_csv.iloc[:, 2:].values
        if dzyaloshinskii_moriya_Y_indices.max() >= config.N:
            print("=" * 50)
            print(
                f"⚠️  Error: DM_Y interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
            )
            print("=" * 50)
            exit()

    # dzyaloshinskii_moriya_Y
    dzyaloshinskii_moriya_Z_indices = None
    dzyaloshinskii_moriya_Z = None
    if not isinstance(config.DM_Z.file, DotMap):
        dzyaloshinskii_moriya_Z_csv = pd.read_csv(
            config.DM_Z.file, delimiter=",", header=None
        )
        dzyaloshinskii_moriya_Z_indices = dzyaloshinskii_moriya_Z_csv.iloc[:, :2].values
        dzyaloshinskii_moriya_Z = dzyaloshinskii_moriya_Z_csv.iloc[:, 2:].values
        if dzyaloshinskii_moriya_Z_indices.max() >= config.N:
            print("=" * 50)
            print(
                f"⚠️  Error: DM_Z interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
            )
            print("=" * 50)
            exit()

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
            print("=" * 50)
            print("⚠️  Error: Please input two columns in model.file for XXZ model")
            print("=" * 50)
            exit()
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
            print("=" * 50)
            print("⚠️  Error: Please input three columns in model.file for XYZ model")
            print("=" * 50)
            exit()
    else:
        print("=" * 50)
        print("⚠️  Error: Please input the correct model type (XXZ or XYZ)")
        print("=" * 50)
        exit()

    return hamiltonian
