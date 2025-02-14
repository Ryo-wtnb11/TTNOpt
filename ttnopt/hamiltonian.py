from ttnopt.src import Hamiltonian
import pandas as pd
from dotmap import DotMap
from pathlib import Path


def hamiltonian(config: DotMap):
    """_summary_

    Args:
        config Dict: the configuration of the system
    """
    if isinstance(config.spin_size, DotMap) and isinstance(
        config.spin_size.file, DotMap
    ):
        print("=" * 50)
        print("⚠️  Error: Please input spin value in spin_size as S or Si_file")
        print("=" * 50)
        exit()

    spin_sizes = None
    if isinstance(config.spin_size, str):
        if Path(config.spin_size).suffix == ".dat":
            spin_csv = pd.read_csv(config.spin_size, delimiter=",", header=None)
            spin_csv_sorted = spin_csv.sort_values(by=0)
            spin_size = spin_csv_sorted.iloc[:, 1].values
            spin_sizes = [f"S={s}" for s in spin_size]
        else:
            spin_sizes = ["S=" + str(config.spin_size)] * config.N
    else:
        print("=" * 50)
        print(
            "⚠️ Error : spin_size is not specified correctly either number or .dat file"
        )
        print("=" * 50)
        exit()

    interaction_indices = None
    interaction_coefs = None
    if Path(config.model.file).suffix == ".dat":
        interaction_csv = pd.read_csv(config.model.file, delimiter=",", header=None)
        interaction_indices = interaction_csv.iloc[:, :2].values
        interaction_coefs = interaction_csv.iloc[:, 2:].values
    else:
        print("=" * 50)
        print("⚠️ Error : model.file is not specified correctly")
        print("=" * 50)
        exit()

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
    if not isinstance(config.MF_X, DotMap):
        if Path(str(config.MF_X)).suffix == ".dat":
            magnetic_field_X_csv = pd.read_csv(config.MF_X, delimiter=",", header=None)
            magnetic_field_X_indices = magnetic_field_X_csv.iloc[:, 0].values
            magnetic_field_X = magnetic_field_X_csv.iloc[:, 1].values
        else:
            if isinstance(float(config.MF_X), float):
                magnetic_field_X_indices = [i for i in range(config.N)]
                magnetic_field_X = [config.MF_X] * config.N
            else:
                print("=" * 50)
                print("⚠️  Error: MF_X should be .dat file or float.")
                print("=" * 50)
                exit()

    # magnetic_field_Y
    magnetic_field_Y_indices = None
    magnetic_field_Y = None
    if not isinstance(config.MF_Y, DotMap):
        if Path(str(config.MF_Y)).suffix == ".dat":
            magnetic_field_Y_csv = pd.read_csv(config.MF_Y, delimiter=",", header=None)
            magnetic_field_Y_indices = magnetic_field_Y_csv.iloc[:, 0].values
            magnetic_field_Y = magnetic_field_Y_csv.iloc[:, 1].values
        else:
            if isinstance(float(config.MF_Y), float):
                magnetic_field_Y_indices = [i for i in range(config.N)]
                magnetic_field_Y = [config.MF_Y] * config.N
            else:
                print("=" * 50)
                print("⚠️  Error: MF_Y should be .dat file or float.")
                print("=" * 50)
                exit()

    # magnetic_field_Z
    magnetic_field_Z_indices = None
    magnetic_field_Z = None
    if not isinstance(config.MF_Z, DotMap):
        if Path(str(config.MF_Z)).suffix == ".dat":
            magnetic_field_Z_csv = pd.read_csv(config.MF_Z, delimiter=",", header=None)
            magnetic_field_Z_indices = magnetic_field_Z_csv.iloc[:, 0].values
            magnetic_field_Z = magnetic_field_Z_csv.iloc[:, 1].values
        else:
            if isinstance(float(config.MF_Z), float):
                magnetic_field_Z_indices = [i for i in range(config.N)]
                magnetic_field_Z = [config.MF_Z] * config.N
            else:
                print("=" * 50)
                print("⚠️  Error: MF_Z should be .dat file or float.")
                print("=" * 50)
                exit()

    ion_anisotropy_indices = None
    ion_anisotropy = None
    if not isinstance(config.SIA, DotMap):
        if Path(str(config.SIA)).suffix == ".dat":
            ion_anisotropy_csv = pd.read_csv(config.SIA, delimiter=",", header=None)
            ion_anisotropy_indices = ion_anisotropy_csv.iloc[:, 0].values
            ion_anisotropy = ion_anisotropy_csv.iloc[:, 1].values
        else:
            if isinstance(float(config.SIA), float):
                ion_anisotropy_indices = [i for i in range(config.N)]
                ion_anisotropy = [config.SIA] * config.N
            else:
                print("=" * 50)
                print("⚠️  Error: SIA should be .dat file or float.")
                print("=" * 50)
                exit()

    # dzyaloshinskii_moriya_X
    dzyaloshinskii_moriya_X_indices = None
    dzyaloshinskii_moriya_X = None
    if not isinstance(config.DM_X, DotMap):
        if Path(config.DM_X).suffix == ".dat":
            dzyaloshinskii_moriya_X_csv = pd.read_csv(
                config.DM_X, delimiter=",", header=None
            )
            dzyaloshinskii_moriya_X_indices = dzyaloshinskii_moriya_X_csv.iloc[
                :, :2
            ].values
            dzyaloshinskii_moriya_X = dzyaloshinskii_moriya_X_csv.iloc[:, 2:].values
            if dzyaloshinskii_moriya_X_indices.max() >= config.N:
                print("=" * 50)
                print(
                    f"⚠️  Error: DM_X interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
                )
                print("=" * 50)
                exit()
        else:
            print("=" * 50)
            print("⚠️  Error: DM_X should be .dat file.")
            print("=" * 50)
            exit()

    # dzyaloshinskii_moriya_Y
    dzyaloshinskii_moriya_Y_indices = None
    dzyaloshinskii_moriya_Y = None
    if not isinstance(config.DM_Y, DotMap):
        if Path(config.DM_Y).suffix == ".dat":
            dzyaloshinskii_moriya_Y_csv = pd.read_csv(
                config.DM_Y, delimiter=",", header=None
            )
            dzyaloshinskii_moriya_Y_indices = dzyaloshinskii_moriya_Y_csv.iloc[
                :, :2
            ].values
            dzyaloshinskii_moriya_Y = dzyaloshinskii_moriya_Y_csv.iloc[:, 2:].values
            if dzyaloshinskii_moriya_Y_indices.max() >= config.N:
                print("=" * 50)
                print(
                    f"⚠️  Error: DM_Y interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
                )
                print("=" * 50)
                exit()
        else:
            print("=" * 50)
            print("⚠️  Error: DM_Y should be .dat file.")
            print("=" * 50)
            exit()

    # dzyaloshinskii_moriya_Y
    dzyaloshinskii_moriya_Z_indices = None
    dzyaloshinskii_moriya_Z = None
    if not isinstance(config.DM_Z, DotMap):
        if Path(config.DM_Z).suffix == ".dat":
            dzyaloshinskii_moriya_Z_csv = pd.read_csv(
                config.DM_Z, delimiter=",", header=None
            )
            dzyaloshinskii_moriya_Z_indices = dzyaloshinskii_moriya_Z_csv.iloc[
                :, :2
            ].values
            dzyaloshinskii_moriya_Z = dzyaloshinskii_moriya_Z_csv.iloc[:, 2:].values
            if dzyaloshinskii_moriya_Z_indices.max() >= config.N:
                print("=" * 50)
                print(
                    f"⚠️  Error: DM_Z interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
                )
                print("=" * 50)
                exit()
        else:
            print("=" * 50)
            print("⚠️  Error: DM_Y should be .dat file.")
            print("=" * 50)
            exit()

    # sod_x
    sod_X_indices = None
    sod_X = None
    if not isinstance(config.SOD_X, DotMap):
        if Path(config.SOD_X).suffix == ".dat":
            sod_x_csv = pd.read_csv(config.SOD_X, delimiter=",", header=None)
            sod_X_indices = sod_x_csv.iloc[:, :2].values
            sod_X = sod_x_csv.iloc[:, 2:].values
            if sod_X_indices.max() >= config.N:
                print("=" * 50)
                print(
                    f"⚠️  Error: SOD_X interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
                )
                print("=" * 50)
                exit()

    # sod_y
    sod_Y_indices = None
    sod_Y = None
    if not isinstance(config.SOD_Y, DotMap):
        if Path(config.SOD_Y).suffix == ".dat":
            sod_y_csv = pd.read_csv(config.SOD_Y, delimiter=",", header=None)
            sod_Y_indices = sod_y_csv.iloc[:, :2].values
            sod_Y = sod_y_csv.iloc[:, 2:].values
            if sod_Y_indices.max() >= config.N:
                print("=" * 50)
                print(
                    f"⚠️  Error: SOD_Y interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
                )
                print("=" * 50)
                exit()

    # sod_x
    sod_Z_indices = None
    sod_Z = None
    if not isinstance(config.SOD_Z, DotMap):
        if Path(config.SOD_Z).suffix == ".dat":
            sod_z_csv = pd.read_csv(config.SOD_Z, delimiter=",", header=None)
            sod_Z_indices = sod_z_csv.iloc[:, :2].values
            sod_Z = sod_z_csv.iloc[:, 2:].values
            if sod_Z_indices.max() >= config.N:
                print("=" * 50)
                print(
                    f"⚠️  Error: SOD_Z interaction indices exceed the allowed range. All indices must be less than N-1 (N={config.system.N})."
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
                sod_X_indices=sod_X_indices,
                sod_X=sod_X,
                sod_Y_indices=sod_Y_indices,
                sod_Y=sod_Y,
                sod_Z_indices=sod_Z_indices,
                sod_Z=sod_Z,
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
                sod_X_indices=sod_X_indices,
                sod_X=sod_X,
                sod_Y_indices=sod_Y_indices,
                sod_Y=sod_Y,
                sod_Z_indices=sod_Z_indices,
                sod_Z=sod_Z,
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
