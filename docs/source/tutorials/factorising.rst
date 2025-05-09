Factorising tensor
=================================

In this tutorial, we show how we can decompose a high-rank tensor into a tensor network representation using the TTNOpt package.

We assume that the input tensor is given as a $.npy$ file, which is a standard format for storing numpy arrays in Python.

Then, a input file for the factrizing tensor is created as follows:

.. code-block:: yaml
    :caption: factorising.yml

    target: tensor.npy
    numerics:
        initial_bond_dimension: 4
        fidelity:
            opt_structure:
                type: 1 # 0: no optimization, 1: stochastic optimization
                beta: [0.04, 0.02] # temperature for stochastic optimization
            init_random: 0
            max_bond_dimensions: [4, 8, 16] # Maximum bond dimension for each repetition
            max_num_sweeps: [100, 50, 25]
            truncation_error: 0.0
            fidelity_convergence_threshold: 1e-10
            entanglement_convergence_threshold: 1e-14

    output:
        dir: TTN_standard

Then, we run the decomposition by the following command:

.. code-block:: bash

    ft factorising.yml

Examples of the input files are provided in the `sample` directory, which reproduces the results of the paper.