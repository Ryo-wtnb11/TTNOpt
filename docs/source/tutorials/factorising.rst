Factorising tensor
=================================

TTNOpt also provides a tool for decomposing a high-rank tensor into a tensor network representation.

We assume that the input tensor is given as a `.npy` file.
Then, a input file for the factrizing tensor is created as follows:

.. code-block:: yaml
    :caption: factorising.yml

    target:
        tensor: tensor.npy
    numerics:
        initial_bond_dimension: 4
        fidelity:
            opt_structure:
                type: 1 # 0: no optimization, 1: structural optimization
            init_random: 0
            max_bond_dimensions: [4, 8, 16] # Maximum bond dimension for each repetition
            max_num_sweeps: [100, 50, 25]
            truncation_error: 0.0
            fidelity_convergence_threshold: 1e-10
            entanglement_convergence_threshold: 1e-14

    output:
        dir: TTN_
        tensors: 1

Then, we run the decomposition by the following command:

.. code-block:: bash

    ft factorising.yml

Then, the results including the decomposed tensors and the connectivity information of the tensor network are output to the specified folder.
Examples of the input files are provided in the `sample` directory, which reproduces the results of the paper.