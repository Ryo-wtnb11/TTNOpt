Reconstruction of TTNs
=================================

We can also optimize the network structure given a tree tensor network using the TTNOpt package.

We assume that the input is given as a TTN, local tensors and the .dat file that specify the connectivity of the network.
All input files should be included in the same directory such as:

.. code-block:: bash

    ├── data
    │   ├──edges.dat
    │   ├──isometry0.npy
    │   ├──isometry1.npy
    │   ├──isometry2.npy
    │   ├──isometry3.npy
    ...

A input file for the reconstruction should be written as follows:

.. code-block:: yaml
    :caption: reconstruction.yml

    target:
        dir: data
        tensors_name: isometry
        graph_file: edges.dat

    numerics:
        opt_structure:
            type: 1 # 0: no optimization, 1: structural optimization
        max_num_sweep: 20
        truncation_error: 1e-8

    output:
        dir: outout

Then, we run the reconstruction by the following command:

.. code-block:: bash

    ft reconstruction.yml

The output will be saved in the directory specified in the `output` section of the input file.

Examples of the input files are provided in the `sample` directory. Especially, by running the `run_reconstruction.sh` script, we can reproduce the results of the paper, that is, the reconstruction of the multivariate normal distribution.