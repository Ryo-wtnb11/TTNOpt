# TTNOpt: Tree Tensor Network Package for high-rank tensor compression
[![Documentation Status](https://readthedocs.org/projects/ttnopt/badge/?version=latest)](https://ttnopt.readthedocs.io/en/latest/)

TTNOpt is a software package that utilizes tree tensor networks (TTNs) for quantum spin systems and highdimensional data analysis.

TTNOpt provides efficient and powerful TTN computations by **locally optimizing the network structure**, guided by the entanglement pattern of the target tensors.

## Documentation
The documentation is avaibale [here](https://ttnopt.readthedocs.io/)

## Installation
One can install TTNOpt from GitHub (recommended)
```
pip install git+https://github.com/Ryo-wtnb11/TTNOpt
```
or from PyPI
```
pip install ttnopt
```

## Quick Start
Prepare a input file in the following format:
```
system:
   N: 8 # Number of spins
   spin_size: 1/2

   # Exchange coupling for the XXZ or XYZ model
   model:
      type: XYZ # Choose XXZ or XYZ
      file: XYZ.dat # Pair-variable file containing J_{i,j}, Δ_{i,j}  or Jx_{i,j}, Jy_{i,j}, Jz_{i,j}

numerics:
   init_tree: 0 # If 0, the initial structure is MPN
   opt_structure:
      type: 1 # 0: no optimization, 1: structural optimization
   initial_bond_dimension: 20
   max_bond_dimensions: [20, 40, 60, 80] # Maximum bond dimension for each repetition
   max_num_sweeps: [20, 10, 7, 5]
   energy_convergence_threshold: 1e-11
   entanglement_convergence_threshold: 1e-10
   energy_degeneracy_threshold: 1e-13
   entanglement_degeneracy_threshold: 0.1

output:
   dir: data
   single_site: 0
   two_site: 0
```

and the data file containing the exchange coupling parameters:
```
 0,  1, -0.436920921879,  0.089051481908,  0.114534548770
 1,  2,  0.438183752739,  0.051969784082, -0.352866682293
 2,  3,  0.091624296707, -0.308710064183, -0.485410291629
 3,  4,  0.162135533123, -0.339637726575, -0.132299037581
 4,  5, -0.135904201148,  0.440882034864,  0.310790500815
 5,  6, -0.104273755378,  0.423013058561, -0.352903122969
 6,  7, -0.338654007842,  0.199255537333, -0.200512881307
```

Then, run the following command:
```
gss input.yaml
```

The output will be saved in the `data` directory. The output files are:
- `basic.csv`: The EEs for all bonds, as well as the variational energies and truncation errors.
- `graph.dat`: The optimized TTN structure.

## Papers
When using TTNOpt for research, please cite:
