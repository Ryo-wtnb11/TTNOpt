Ground State Search
=================================

In this tutorial, we show how we can optimize the ground state through the variational optimization combined with the automatic structure search.

We consider the ground state search for :math:`S=1/2` hierarchical chain model with system size :math:`N=2^d`:

.. math::
    H = \sum_{h=0}^{d-1} \sum_{i \in I(h)} J \alpha^h {\boldsymbol{s}}_i \cdot {\boldsymbol{s}}_{i+1}~,

where :math:`J` is the base coupling constant, :math:`0<\mathbf{\alpha}\leq 1.0` is the decay factor for the coupling strength,
:math:`h \in [0, d-1]` represents the height in the perfect binary tree (PBT) structure, and :math:`I(h)=\left\{i \mid i=2^h(2 k+1)-1,~ {\rm{where}}~k=0, 1, \ldots, 2^{d-h-1}-1\right\}` specifies pairs of adjacent sites :math:`(i, i+1)` according to the PBT structure.
This model is illustrated in the following figure:

In the case of :math:`\alpha=1.0`, the model is equivalent to the standard one-dimensional Heisenberg model. 
However, for :math:`\alpha<1.0`, the model have inhomogeneous interactions and its ground state is well-expressed as TTN, rather than MPS :cite:`hikihara_Automatic_2023`.

The TTNOpt package searchs the ground state of the quantum spin systems with non-trivial one-dimensional entanglement structures by simultaneously performing **variational optimization** and **network structure optimization**.

First, we set a input file for the Hamiltonian of the model.

.. code-block:: yaml
    :caption: input.yml

    system:
        N: 8 # Number of spins
        spin_size: 1/2 # Spin size (used only if uniform is 1)

        # Exchange coupling for the XXZ or XYZ model
        model:
            type: XXZ # Choose XXZ or XYZ
            file: XXZ.dat # Pair-variable file containing J_{i,j}, Δ_{i,j}  or Jx_{i,j}, Jy_{i,j}, Jz_{i,j}

    numerics:
        init_tree: 0
        opt_structure: 1
        initial_bond_dimension: 4
        max_bond_dimensions: [20] # Maximum bond dimension for each repetition
        max_num_sweeps: [50]
        energy_convergence_threshold: 1e-11
        entanglement_convergence_threshold: 1e-10
        energy_degeneracy_threshold: 1e-13
        entanglement_degeneracy_threshold: 0.1
        opt_structure:
            type: 1
            temperature: [0.01, 0.001] # if type is 1, temperature is set by this values and structure is chosen stochastically. that is always 0 by default (select always minimum EE)..

    output:
        dir: TTN
        single_site: 1
        two_site: 1

XXZ.dat is a file containing the interactions between two sites and their coefficients. In this case, the file contains the following:

.. code-block:: dat
    :caption: XXZ.dat

    0,1,1.0,1.0
    1,2,0.5,0.5
    2,3,1.0,1.0
    3,4,0.25,0.25
    4,5,1.0,1.0
    5,6,0.5,0.5
    6,7,1.0,1.0

Then we run the optimization by the following command:

.. code-block:: bash

    gss input.yml

The standard output is as follows:

.. code-block:: bash

    No initial tensors in TTN object.
    Initialize tensors with real space renormalization.
    Sweep count: 1
    -3.0592248189324436
    -3.059224818932444
    -3.0592249735506445
    -3.0592369142108216
    -3.059236914210821
    -3.0592369151984746
    Sweep count: 2
    -3.0592369151984764
    -3.059236915198476
    -3.0592369151984786
    -3.059236915198477
    -3.059236915198478
    -3.059236915198478
    Sweep count: 3
    -3.0592369151984737
    -3.0592369151984755
    -3.0592369151984746
    -3.0592369151984733
    -3.0592369151984733
    -3.059236915198475
    Sweep count: 4
    -3.059236915198475
    -3.059236915198477
    -3.0592369151984724
    -3.059236915198476
    -3.059236915198475
    -3.0592369151984746
    Sweep count: 5
    -3.0592369151984737
    -3.059236915198473
    -3.059236915198476
    -3.0592369151984737
    -3.0592369151984737
    -3.0592369151984777
    Converged
    Calculating the expectation values for the initial structure
    Sweep count: 1
    -3.0592369151984746
    -3.0592369151984755
    -3.059236915198479
    -3.0592369151984746
    -3.0592369151984737
    -3.059236915198477
    Converged

.. bibliography::
   :cited: