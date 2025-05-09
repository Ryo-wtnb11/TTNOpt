Quick Start: Ground State Search
=================================

In this quick start, we show the basic usage of the TTNOpt package for the ground state search of the quantum spin systems.


Creating Input Files
------------------------

First, specify the model and parameters of the Hamiltonian and set up the computational method.  
Each file should be written according to the paper.

Samples of the input files are provided in the `sample` directory. Input files are written in YAML format and looks like the following:

.. code-block:: yaml
   :caption: input.yml

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



Execution
------------------------

Use the name of the input file created above as an argument and execute the TTNOpt ground state search with the following command:

.. code-block:: bash

   gss input_file_name


Execution Results
------------------------

The execution results are output to the specified folder.  