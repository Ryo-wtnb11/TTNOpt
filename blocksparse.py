import tensornetwork as tn
import numpy as np
from scipy.linalg import eigh_tridiagonal
from src.Observable import bare_spin_operator, spin_dof
from src.TwoSiteUpdater import TwoSiteUpdater
from src.functionTTN import get_renormalization_sequence, get_bare_edges
from scipy.sparse.linalg import expm
from tensornetwork import U1Charge, Index, BlockSparseTensor

np.set_printoptions(precision=3, suppress=True)
tn.set_default_backend("symmetric")

c_i = U1Charge([3, 4, 4, 4, 4, 5])  # charges on leg i
c_j = U1Charge([0, 1, 2, 3])  # charges on leg j
c_k = U1Charge([3, 4, 4, 4, 4, 5])  # charges on leg i
c_l = U1Charge([0, 1, 2, 3])  # charges on leg j
c = U1Charge([8])
# use `Index` to bundle flow and charge information
i = Index(charges=c_i, flow=True)
j = Index(charges=c_j, flow=True)
k = Index(charges=c_k, flow=True)
l = Index(charges=c_l, flow=True)
a = Index(charges=c, flow=False)
bs = BlockSparseTensor.random(
    [i, j, k, l, a],
    dtype=np.complex128,
)
tensor = tn.Node(bs)
a = tensor[0]
b = tensor[1]
c = tensor[2]
d = tensor[3]
e = tensor[4]
(u, s, v, terr) = tn.split_node_full_svd(tensor, [e, a, b], [c, d])
print(s.tensor.todense())
print(s.tensor.flat_charges)

tensor = tn.Node(bs)
a = tensor[0]
b = tensor[1]
c = tensor[2]
d = tensor[3]
e = tensor[4]
(u, s, v, terr) = tn.split_node_full_svd(
    tensor, [e, a, b], [c, d], max_singular_values=5
)
print(s.tensor.todense())
print(s.tensor.flat_charges)
