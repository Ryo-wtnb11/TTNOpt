import tensornetwork as tn
from tensornetwork import U1Charge, BlockSparseTensor, Index
import numpy as np
import pprint
import tracemalloc

if __name__ == "__main__":
    print("Sparse Tensor Network")

    pauli_z = np.array([[1 / 2, 0], [0, -1 / 2]], dtype=np.complex128)
    pauli_x = np.array([[0, 1 / 2], [1 / 2, 0]], dtype=np.complex128)
    pauli_y = np.array([[0, -1j / 2], [1j / 2, 0]], dtype=np.complex128)
    pauli_p = pauli_x + 1j * pauli_y
    pauli_m = pauli_p.T

    zcharge = U1Charge([0, 1])
    xycharge = U1Charge([1, 0])
    pcharge = U1Charge([1, 2])
    mcharge = U1Charge([-1, 0])
    sparse_z = BlockSparseTensor.fromdense(
        [Index(zcharge, flow=False), Index(zcharge, flow=True)], pauli_z
    )
    sparse_p = BlockSparseTensor.fromdense(
        [Index(pcharge, flow=False), Index(zcharge, flow=True)], pauli_p
    )
    sparse_m = BlockSparseTensor.fromdense(
        [Index(mcharge, flow=False), Index(zcharge, flow=True)], pauli_m
    )
    sparse_x = BlockSparseTensor.fromdense(
        [Index(xycharge, flow=False), Index(zcharge, flow=True)], pauli_x
    )
    sparse_y = BlockSparseTensor.fromdense(
        [Index(xycharge, flow=False), Index(zcharge, flow=True)], pauli_y
    )

    u1 = U1Charge([0, 1])
    u2 = U1Charge([0, 1])
    u_in = U1Charge([0, 1, 1, 2])
    isometry_sparse = BlockSparseTensor.random(
        [Index(u1, flow=True), Index(u2, flow=True), Index(u_in, flow=False)]
    )
    isometry = isometry_sparse.todense()
    print(isometry)

    # Dense contraction
    print("Dense Contraction")
    print("Sz")
    bra = tn.Node(isometry)
    ket = bra.copy(conjugate=True)
    spin_z = tn.Node(pauli_z)

    bra[0] ^ spin_z[0]
    ket[0] ^ spin_z[1]
    bra[1] ^ ket[1]
    Sz_dense = tn.contractors.auto(
        [bra, spin_z, ket], output_edge_order=[bra[2], ket[2]]
    )
    print(Sz_dense.tensor)

    print("S+")
    bra = tn.Node(isometry)
    ket = bra.copy(conjugate=True)
    spin_p = tn.Node(pauli_p)

    bra[0] ^ spin_p[0]
    ket[0] ^ spin_p[1]
    bra[1] ^ ket[1]
    Sp_dense = tn.contractors.auto(
        [bra, spin_p, ket], output_edge_order=[bra[2], ket[2]]
    )
    print(Sp_dense.tensor)

    # Sparse contraction
    print("Sparse Contraction")
    print("Sz")
    bra = tn.Node(isometry_sparse, backend="symmetric")
    ket = bra.copy(conjugate=True)
    spin_z = tn.Node(sparse_z, backend="symmetric")
    u_in = U1Charge(U1Charge.fuse([0, 1], [0, 1]))

    bra[0] ^ spin_z[0]
    ket[0] ^ spin_z[1]
    bra[1] ^ ket[1]
    Sz = tn.contractors.auto([bra, spin_z, ket], output_edge_order=[bra[2], ket[2]])
    print(Sz.tensor.todense())

    print("S+")
    u1 = U1Charge([1, 2])
    u2 = U1Charge([0, 1])
    u_in_p = U1Charge(U1Charge.fuse([1, 2], [0, 1]))
    print(isometry_sparse.charges[0])
    print(isometry_sparse.charges[1])
    print(isometry_sparse.charges[2])
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024:.2f} KB")
    print(f"Peak memory usage: {peak / 1024:.2f} KB")

    tracemalloc.stop()
    new_tensor = BlockSparseTensor(
        data=isometry_sparse.data,
        flows=[True, True, False],
        charges=[u1, u2, u_in_p],
    )
    isometry_sparse_ket = BlockSparseTensor.fromdense(
        [Index(u1, flow=True), Index(u2, flow=True), Index(u_in_p, flow=False)],
        isometry_sparse.todense(),
    )
    print(isometry_sparse_ket.charges[0])
    print(isometry_sparse_ket.charges[1])
    print(isometry_sparse_ket.charges[2])
    bra = tn.Node(isometry_sparse_ket, backend="symmetric")
    ket = tn.Node(isometry_sparse, backend="symmetric").copy(conjugate=True)
    spin_p = tn.Node(sparse_p, backend="symmetric")
    bra[0] ^ spin_p[0]
    ket[0] ^ spin_p[1]
    bra[1] ^ ket[1]
    Sp = tn.contractors.auto([bra, spin_p, ket], output_edge_order=[bra[2], ket[2]])
    print(Sp.tensor.todense())

    # # lanczos test
    # print("lanczos test")
    # u1 = U1Charge([0, 1])
    # u_in = U1Charge(U1Charge.fuse([0, 1], [0, 1]))
    # magnetic = U1Charge([2])
    # m_index = Index(magnetic, flow=False)
    # psi = BlockSparseTensor.random([
    #     Index(u1, flow=True),
    #     Index(u_in, flow=True),
    #     Index(u_in, flow=True),
    #     Index(u1, flow=True),
    #     m_index
    # ])

    # np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
    # print(np.array(psi.todense().reshape([8, 8])))

    # psi_dense = psi.todense()

    # # Dense Test
    # print("Dense Test")
    # psi_node = tn.Node(psi_dense)
    # # S+S-
    # psi_node[1] ^ Sp_dense[0]
    # psi_node = tn.contractors.auto([psi_node, Sp_dense], output_edge_order=[psi_node[0], Sp_dense[1], psi_node[2], psi_node[3], psi_node[4]])
    # psi_node[2] ^ Sm_dense[0]
    # psi_node = tn.contractors.auto([psi_node, Sm_dense], output_edge_order=[psi_node[0], psi_node[1], Sm_dense[1], psi_node[3], psi_node[4]])
    # # SzSz
    # psi_node[1] ^ Sz_dense[0]
    # psi_node = tn.contractors.auto([psi_node, Sz_dense], output_edge_order=[psi_node[0], Sz_dense[1], psi_node[2], psi_node[3], psi_node[4]])
    # psi_node[2] ^ Sz_dense[0]
    # psi_node = tn.contractors.auto([psi_node, Sz_dense], output_edge_order=[psi_node[0], psi_node[1], Sz_dense[1], psi_node[3], psi_node[4]])

    # print(np.real(np.array(psi_node.tensor.reshape([8, 8]))))

    # #
    # print("Sparse Test")
    # # S+S-
    # psi_ = BlockSparseTensor.fromdense(
    #     [Index(u1, flow=True), Index(u_in_p, flow=True), Index(u_in_m, flow=True), Index(u1, flow=True), m_index], psi_dense
    # )
    # psi_node = tn.Node(psi_, backend="symmetric")
    # psi_node[1] ^ Sp[0]
    # psi_node = tn.contractors.auto([psi_node, Sp], output_edge_order=[psi_node[0], Sp[1], psi_node[2], psi_node[3], psi_node[4]])
    # psi_node[2] ^ Sm[0]
    # psi_node = tn.contractors.auto([psi_node, Sm], output_edge_order=[psi_node[0], psi_node[1], Sm[1], psi_node[3], psi_node[4]])
    # # SzSz
    # psi_node[1] ^ Sz[0]
    # psi_node = tn.contractors.auto([psi_node, Sz], output_edge_order=[psi_node[0], Sz[1], psi_node[2], psi_node[3], psi_node[4]])
    # psi_node[2] ^ Sz[0]
    # psi_node = tn.contractors.auto([psi_node, Sz], output_edge_order=[psi_node[0], psi_node[1], Sz[1], psi_node[3], psi_node[4]])
    # print(np.real(np.array(psi_node.tensor.todense().reshape([8, 8]))))

    # #
    # print("Sum Test")

    # spin_operators = [pauli_z, pauli_z]
    # block_zz = tn.ncon(
    #     spin_operators,
    #     [["-b0", "-k0"], ["-b1", "-k1"]],
    #     out_order=["-b0", "-b1", "-k0", "-k1"],
    # )
    # block_zz = block_zz.reshape([4, 4])

    # spin_operators = [pauli_p, pauli_m]
    # block_pm = tn.ncon(
    #     spin_operators,
    #     [["-b0", "-k0"], ["-b1", "-k1"]],
    #     out_order=["-b0", "-b1", "-k0", "-k1"],
    # )
    # block_pm = block_pm.reshape([4, 4])

    # block = block_zz + block_pm
    # print(block)

    # spin_operators = [sparse_z, sparse_z]
    # block_zz = tn.ncon(
    #     spin_operators,
    #     [["-b0", "-k0"], ["-b1", "-k1"]],
    #     out_order=["-b0", "-b1", "-k0", "-k1"],
    #     backend="symmetric"
    # )
    # block_zz = block_zz.reshape([4, 4])
    # zz_node = tn.Node(block_zz, backend="symmetric")

    # spin_operators = [sparse_p, sparse_m]
    # block_pm = tn.ncon(
    #     spin_operators,
    #     [["-b0", "-k0"], ["-b1", "-k1"]],
    #     out_order=["-b0", "-b1", "-k0", "-k1"],
    #     backend="symmetric"
    # )
    # block_pm = block_pm.reshape([4, 4])
    # pm_node = tn.Node(block_pm, backend="symmetric")

    # node = zz_node + pm_node
    # print(node.tensor.todense())
