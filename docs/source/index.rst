.. ttnopt documentation master file, created by
   sphinx-quickstart on Thu Sep 19 20:57:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TTNOpt documentation
====================

TTNOpt is a software package that utilizes tree tensor networks (TTNs) for quantum spin systems and highdimensional data analysis. 

TTNOpt provides efficient and powerful TTN computations by **locally optimizing the network structure**,
guided by the entanglement pattern of the target tensors.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quick_start_gss

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/ground_state_search
   tutorials/factorising
   tutorials/reconstruction

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/TreeTensorNetwork
   api/Observable
   api/Hamiltonian
   api/GroundStateSearch

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`