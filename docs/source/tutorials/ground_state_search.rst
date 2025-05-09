Ground State Search
=================================

In this tutorial, we show how we can optimize the ground state through the variational optimization combined with the automatic structure search.

We consider the ground state search for :math:`S=1/2` hierarchical chain model with system size :math:`N=2^d`:

.. math::
    H = \sum_{h=0}^{d-1} \sum_{i \in I(h)} J \alpha^h {\boldsymbol{s}}_i \cdot {\boldsymbol{s}}_{i+1}~,

where :math:`J` is the base coupling constant, :math:`0<\mathbf{\alpha}\leq 1.0` is the decay factor for the coupling strength,
:math:`h \in [0, d-1]` represents the height in the perfect binary tree (PBT) structure, and :math:`I(h)=\left\{i \mid i=2^h(2 k+1)-1,~ {\rm{where}}~k=0, 1, \ldots, 2^{d-h-1}-1\right\}` specifies pairs of adjacent sites :math:`(i, i+1)` according to the PBT structure.

In the case of :math:`\alpha=1.0`, the model is equivalent to the standard one-dimensional Heisenberg model. 
However, for :math:`\alpha<1.0`, the model have inhomogeneous interactions and its ground state is well-expressed as TTN, rather than MPS :cite:`hikihara_Automatic_2023`.

The TTNOpt package searchs the ground state of the quantum spin systems with non-trivial one-dimensional entanglement structures by simultaneously performing **variational optimization** and **network structure optimization**.

First, we set a input file for the Hamiltonian of the model.

.. code-block:: yaml
    :caption: input.yml

    system:
        N: 8
        spin_size: 1/2

        model:
            type: XXZ
            file: XXZ.dat

    numerics:
        init_tree: 0
        opt_structure:
            type: 1 # 0: no optimization, 1: structural optimization
            temperature: [0.01, 0.001]
        initial_bond_dimension: 4
        max_bond_dimensions: [20]
        max_num_sweeps: [50]
        energy_convergence_threshold: 1e-11
        entanglement_convergence_threshold: 1e-10
        energy_degeneracy_threshold: 1e-13
        entanglement_degeneracy_threshold: 0.1

    output:
        dir: data
        single_site: 1
        two_site: 1

XXZ.dat is a file containing the hierarchical interactions between two sites and their coefficients. In this case, the file contains the following:

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
    -3.040606434114664
    -3.0406064341146632
    -3.0406064830544235
    -3.0423226276018
    -3.0423226276018
    -3.042322627756622
    Sweep count: 2
    -3.0423226277566227
    -3.0423226277566227
    -3.042322627756622
    -3.042322627756622
    -3.04232262775662
    -3.042322627756622
    Sweep count: 3
    -3.042322627756622
    -3.042322627756623
    -3.042322627756621
    -3.042322627756623
    -3.042322627756621
    -3.0423226277566213
    Sweep count: 4
    -3.0423226277566218
    -3.042322627756621
    -3.0423226277566195
    -3.0423226277566213
    -3.042322627756622
    -3.0423226277566213
    Sweep count: 5
    -3.0423226277566218
    -3.042322627756623
    -3.042322627756623
    -3.0423226277566227
    -3.042322627756623
    -3.042322627756622
    Converged
    Calculating the expectation values for the initial structure
    Sweep count: 1
    -3.042322627756623
    -3.042322627756622
    -3.042322627756625
    -3.0423226277566218
    -3.0423226277566227
    -3.0423226277566218
    Converged

In the output, we can see the results of the variational optimization and the network structure optimization.
Specifically, the network structure after the optimization is contained in the file `graph.dat`. In this case, the file contains the following:

.. code-block:: dat
    :caption: graph.dat

    0,1,8
    3,2,9
    9,8,10
    4,5,12
    12,11,10
    6,7,11

which indicates the hierarchical structure of the optimized TTN.

.. bibliography::
   :cited: