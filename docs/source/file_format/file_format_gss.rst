.. _file_format_gss:

File Formats for Ground State Search
=================================

This section explains the file formats supported by the TTNOpt package for the ground state search.

Models
-------
The TTNOpt package has been developed for finite-size quantum spin systems, including XXZ and XYZ Hamiltonians:

.. math::

    \begin{align}
    \label{eq:xxz_hamiltonian}
    H_{\text{XXZ}} &= \sum_{i, j}  J_{ij}\left( s_i^x s_j^x + s_i^y s_j^y + \Delta^z_{ij} s_i^z s_j^z \right ) \\
    \label{eq:xyz_hamiltonian}
    H_{\text{XYZ}} &= \sum_{i, j}  J^x_{ij} s_i^x s_j^x + J^y_{ij} s_i^y s_j^y + J_{ij}^z  s_i^z s_j^z 
    \end{align}


.. _file_format_gss_output:

Outputs
-------