from typing import Dict, Optional
import numpy as np
import tensornetwork as tn
from tensornetwork import U1Charge, Index, BlockSparseTensor
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import expm

from ttnopt.src.TTN import TreeTensorNetwork
from ttnopt.src.Hamiltonian import Hamiltonian
from ttnopt.src.Observable import bare_spin_operator, spin_dof
from ttnopt.src.TwoSiteUpdaterSparse import TwoSiteUpdaterSparse
from ttnopt.src.functionTTN import get_renormalization_sequence, get_bare_edges, inner_product_sparse

from scipy.sparse.linalg import expm
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import expm

class PhysicsEngineSparse(TwoSiteUpdaterSparse):
    def __init__(
        self,
        psi: TreeTensorNetwork,
        hamiltonian: Hamiltonian,
        u1_num: int,
        init_bond_dim: int,
        max_bond_dim: int,
        truncation_error: float,
        edge_spin_operators: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        block_hamiltonians: Optional[Dict[int, Dict[str, np.ndarray]]] = None
    ):
        """Initialize a PhysicsEngineSparse object.

        Args:
            psi (TreeTensorNetwork): The quantum state.
            hamiltonians (Hamiltonian): Hamiltonian which has a list of Observables.
            u1_num (int): The number of U1 charges.
            init_bond_dim (int): Initial bond dimension.
            max_bond_dim (int): Maximum bond dimension.
            truncation_error (float): Maximum truncation error.
            edge_spin_operators (Optional(Dict[int, Dict[str, np.ndarray]]): Spin operators at each edge. Defaults to None.
            block_hamiltonians (Optional(Dict[int, Dict[str, np.ndarray]]): Block_hamiltonian at each edge. Defaults to None.
            edge_u1_charges (Optional[Dict[int, U1Charge]]): U1 charges at each edge. Defaults to None.
        """

        super().__init__(psi)
        self.hamiltonian = hamiltonian
        self.u1_num = u1_num
        self.init_bond_dim = init_bond_dim
        self.max_bond_dim = max_bond_dim
        self.truncation_err = truncation_error
        if edge_spin_operators is None:
            self.edge_spin_operators = self._init_spin_operator()
        else:
            self.edge_spin_operators = edge_spin_operators
        if block_hamiltonians is None:
            self.block_hamiltonians = self._init_block_hamiltonians()
        else:
            self.block_hamiltonians = block_hamiltonians

        init_tensors_flag = False
        if (
            self.psi.tensors is None
        ):  # if there is no initial tensors, we need to generate it
            print("No initial tensors in TTN object.")
            self.psi.tensors = []
            for _ in self.psi.edges:
                self.psi.tensors.append(None)
            init_tensors_flag = True
        else:
            for k in self.hamiltonian.spin_size.keys():
                if spin_dof(self.hamiltonian.spin_size[k]) != self.psi.edge_dims[k]:
                    print("Initial tensors is not valid for given hamiltonian.")
                    init_tensors_flag = True
                    break

        if init_tensors_flag:
            print("Initialize tensors with real space renormalization.")

            for k in self.hamiltonian.spin_size.keys():
                self.psi.edge_dims[k] = spin_dof(self.hamiltonian.spin_size[k])
            self.edge_u1_charges = self._init_edge_u1_charge()
            self.init_tensors_by_block_hamiltonian()

    def calculate_expval(self, indices, operators):
        def _calculate_single_expval(index, central_tensor_id, operator):
            bra = tn.Node(self.psi.tensors[central_tensor_id], backend=self.backend)
            gauge = tn.Node(self.psi.gauge_tensors, backend=self.backend)
            bra[2] ^ gauge[0]
            bra = tn.contractors.auto(
                [bra, gauge], output_edge_order=[bra[0], bra[1], gauge[1]]
            )
            ket = bra.copy(conjugate=True)
            spin = tn.Node(self._spin_operator_at_edge(index, index, operator), backend=self.backend)

            if index == self.psi.edges[central_tensor_id][0]:
                bra[0] ^ spin[0]
                ket[0] ^ spin[1]
                bra[1] ^ ket[1]
            elif index == self.psi.edges[central_tensor_id][1]:
                bra[1] ^ spin[0]
                ket[1] ^ spin[1]
                bra[0] ^ ket[0]

            bra[2] ^ ket[2]
            return tn.contractors.auto([bra, spin, ket])

        def _calculate_double_expval(central_tensor_ids, indices, operators, apply_ids):
            psi1 = tn.Node(self.psi.tensors[central_tensor_ids[0]], backend=self.backend)
            psi2 = tn.Node(self.psi.tensors[central_tensor_ids[1]], backend=self.backend)
            gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)
            psi1[2] ^ gauge[0]
            psi2[2] ^ gauge[1]
            bra = tn.contractors.auto(
                [psi1, psi2, gauge],
                output_edge_order=[psi1[0], psi1[1], psi2[0], psi2[1]],
            )
            ket = bra.copy()

            spin1 = tn.Node(
                self._spin_operator_at_edge(
                    central_tensor_ids[0][apply_ids[0]], indices[0], operators[0]
                )
                , backend=self.backend
            )

            spin2 = tn.Node(
                self._spin_operator_at_edge(
                    central_tensor_ids[1][apply_ids[1]], indices[1], operators[1]
                )
                , backend=self.backend
            )

            output_edge_order = bra.get_all_edges()

            bra[apply_ids[0]] ^ spin1[0]
            output_edge_order[apply_ids[0]] = spin1[1]
            bra = tn.contractors.auto([bra, spin1], output_edge_order=output_edge_order)
            output_edge_order = bra.get_all_edges()
            bra[apply_ids[1]] ^ spin2[0]
            bra = tn.contractors.auto([bra, spin2], output_edge_order=output_edge_order)
            exp_val = inner_product_sparse(bra, ket)
            return exp_val

        self.distance = self.initial_distance()
        self.flag = self.initial_flag()

        if len(indices) == 1:  # one-site expectation value
            index = indices[0]
            while self.candidate_edge_ids() != []:
                central_tensor_ids = self.psi.central_tensor_ids()
                if (
                    index == self.psi.edges[central_tensor_ids[0]][0]
                    or index == self.psi.edges[central_tensor_ids[0]][1]
                ):
                    expval = _calculate_single_expval(
                        index, central_tensor_ids[0], operators[0]
                    )
                elif (
                    index == self.psi.edges[central_tensor_ids[1]][0]
                    or index == self.psi.edges[central_tensor_ids[1]][1]
                ):
                    expval = _calculate_single_expval(
                        index, central_tensor_ids[1], operators[0]
                    )
                self.move_center()
        else:  # two-site expectation value

            def check_and_calculate(apply_ids, indices_order):
                return _calculate_double_expval(
                    central_tensor_ids, indices_order, operators, apply_ids
                )

            while self.candidate_edge_ids() != []:
                central_tensor_ids = self.psi.central_tensor_ids()

                conditions = [
                    ((0, 0), [indices, indices[::-1]]),
                    ((1, 0), [indices, indices[::-1]]),
                    ((0, 1), [indices, indices[::-1]]),
                    ((1, 1), [indices, indices[::-1]]),
                ]

                for apply_ids, indices_orders in conditions:
                    if indices[0] in get_bare_edges(
                        central_tensor_ids[apply_ids[0]],
                        self.psi.edges,
                        self.psi.physical_edges,
                    ) and indices[1] in get_bare_edges(
                        central_tensor_ids[apply_ids[1]],
                        self.psi.edges,
                        self.psi.physical_edges,
                    ):
                        expval = check_and_calculate(apply_ids, indices_orders[0])
                    elif indices[1] in get_bare_edges(
                        central_tensor_ids[apply_ids[0]],
                        self.psi.edges,
                        self.psi.physical_edges,
                    ) and indices[0] in get_bare_edges(
                        central_tensor_ids[apply_ids[1]],
                        self.psi.edges,
                        self.psi.physical_edges,
                    ):
                        expval = check_and_calculate(apply_ids, indices_orders[1])

                self.move_center()

        return expval

    def energy(self):
        central_tensor_ids = self.psi.central_tensor_ids()
        psi_ = self.contract_central_tensors()
        psi_h = psi_.copy()
        psi_h = self._apply_ham_psi(psi_h, central_tensor_ids)
        return inner_product_sparse(psi_, psi_h)

    def lanczos(self, central_tensor_ids, tol=1e-10):
        psi_1 = tn.Node(self.psi.tensors[central_tensor_ids[0]], backend=self.backend)
        psi_2 = tn.Node(self.psi.tensors[central_tensor_ids[1]], backend=self.backend)
        psi_1[2] ^ psi_2[2]
        psi = tn.contractors.auto(
            [psi_1, psi_2],
            output_edge_order=[psi_1[0], psi_1[1], psi_2[0], psi_2[1], psi_1[3]],
        )
        # normalization
        psi = psi / np.sqrt(inner_product_sparse(psi, psi))
        psi_ = psi.copy()
        psi_0 = psi.copy()
        dim_n = np.prod(psi.shape)
        alpha = np.zeros(dim_n, dtype=np.float64)
        beta = np.zeros(dim_n, dtype=np.float64)

        psi_w = self._apply_ham_psi(psi, central_tensor_ids)

        alpha[0] = np.real(inner_product_sparse(psi_w, psi))
        omega = psi_w.tensor - alpha[0] * psi.tensor

        if dim_n == 1:
            eigen_vectors = psi
            return eigen_vectors
        else:
            e_old = 0
            for j in range(1, dim_n):
                beta[j] = np.sqrt(inner_product_sparse(tn.Node(omega, backend=self.backend), tn.Node(omega, backend=self.backend)))
                if j == 1 and beta[j] < tol:
                    eigen_vectors = psi
                    return eigen_vectors
                elif j > 1 and beta[j] < tol:
                    break
                psi = tn.Node(omega / beta[j], backend=self.backend)
                psi_w = self._apply_ham_psi(psi, central_tensor_ids)
                alpha[j] = np.real(inner_product_sparse(psi_w, psi))
                omega = psi_w.tensor - alpha[j] * psi.tensor - beta[j] * psi_.tensor
                psi_ = psi

                if j >= 1:
                    e, v_tilda = eigh_tridiagonal(
                        np.real(alpha[: j + 1]),
                        np.real(beta[1 : j + 1]),
                        select="i",
                        select_range=(0, 0),
                    )
                    if np.abs(e - e_old) < tol:
                        break
                    e_old = e

        v_tilda = np.array(v_tilda.flatten(), dtype=np.complex128)
        v = v_tilda[0] * psi_0.tensor
        psi = psi_0
        psi_ = psi_0
        psi_w = self._apply_ham_psi(psi, central_tensor_ids)
        a = np.real(inner_product_sparse(psi_w, psi))
        omega = psi_w.tensor - a * psi.tensor
        for k in range(1, len(v_tilda)):
            b = np.sqrt(inner_product_sparse(tn.Node(omega, backend=self.backend), tn.Node(omega, backend=self.backend)))
            psi = tn.Node(omega / b, backend=self.backend)
            v += v_tilda[k] * psi.tensor
            psi_w = self._apply_ham_psi(psi, central_tensor_ids)
            a = np.real(inner_product_sparse(psi_w, psi))
            omega = psi_w.tensor - a * psi.tensor - b * psi_.tensor
            psi_ = psi

        # check convergence
        v = tn.Node(v, backend=self.backend)
        v_ = self._apply_ham_psi(v, central_tensor_ids)
        v_ = v_ / np.sqrt(inner_product_sparse(v_, v_))
        delta_v = v_.tensor - np.sign(e)[0] * v.tensor
        while np.sqrt(inner_product_sparse(tn.Node(delta_v, backend=self.backend), tn.Node(delta_v, backend=self.backend))) > tol:
            v = self._apply_ham_psi(v, central_tensor_ids)
            v = v / np.sqrt(inner_product_sparse(v, v))
            delta_v = v.tensor - np.sign(e)[0] * v_.tensor
            v_ = v

        eigen_vectors = v
        return eigen_vectors

    def lanczos_exp_multiply(self, central_tensor_ids, dt, tol=1e-10):
        psi_1 = tn.Node(self.psi.tensors[central_tensor_ids[0]], backend=self.backend)
        psi_2 = tn.Node(self.psi.tensors[central_tensor_ids[1]], backend=self.backend)
        psi_1[2] ^ psi_2[2]
        psi = tn.contractors.auto(
            [psi_1, psi_2], output_edge_order=[psi_1[0], psi_1[1], psi_2[0], psi_2[1]]
        )
        psi = psi / np.linalg.norm(psi.tensor)
        psi_ = psi.copy()
        psi_0 = psi.copy()
        dim_n = np.prod(psi.shape)
        alpha = np.zeros(dim_n, dtype=np.float64)
        beta = np.zeros(dim_n, dtype=np.float64)

        psi_w = self._apply_ham_psi(psi, central_tensor_ids)

        alpha[0] = np.real(inner_product_sparse(psi_w, psi))
        omega = psi_w.tensor - np.array(alpha[0]) * psi.tensor

        if dim_n == 1:
            eigen_vectors = psi
            return eigen_vectors
        else:
            e_old = 0
            for j in range(1, dim_n):
                beta[j] = np.linalg.norm(omega)
                if j > 1 and beta[j] < tol:
                    break
                psi = tn.Node(omega / beta[j], backend=self.backend)
                psi_w = self._apply_ham_psi(psi, central_tensor_ids)
                alpha[j] = np.real(inner_product_sparse(psi_w, psi))
                omega = psi_w.tensor - alpha[j] * psi.tensor - beta[j] * psi_.tensor
                psi_ = psi
                if j >= 1:
                    e, v_tilda = eigh_tridiagonal(
                        np.real(alpha[: j + 1]),
                        np.real(beta[1 : j + 1]),
                        select="i",
                        select_range=(0, 0),
                    )
                    if np.abs(e - e_old) < tol:
                        tri = (
                            np.diag(np.real(alpha[: j + 1]))
                            + np.diag(np.real(beta[1 : j + 1]), 1)
                            + np.diag(np.real(beta[1 : j + 1]), -1)
                        )
                        break
                    e_old = e
        v0 = np.zeros(tri.shape[0])
        v0[0] = 1.0
        v_tilda = expm(-1.0j * dt * tri) @ v0
        v = v_tilda[0] * psi_0.tensor
        psi = psi_0
        psi_ = psi_0
        psi_w = self._apply_ham_psi(psi, central_tensor_ids)
        a = np.real(inner_product_sparse(psi_w, psi))
        omega = psi_w.tensor - a * psi.tensor
        for k in range(1, len(v_tilda)):
            b = np.linalg.norm(omega)
            psi = tn.Node(omega / b, backend=self.backend)
            v += v_tilda[k] * psi.tensor
            psi_w = self._apply_ham_psi(psi, central_tensor_ids)
            a = np.real(inner_product_sparse(psi_w, psi))
            omega = psi_w.tensor - a * psi.tensor - b * psi_.tensor
            psi_ = psi

        # check convergence
        v = tn.Node(v, backend=self.backend)
        v_ = self._apply_ham_psi(v, central_tensor_ids)
        v_ = v_ / np.linalg.norm(v_.tensor)
        delta_v = v_.tensor - np.sign(e)[0] * v.tensor
        while np.linalg.norm(delta_v) > tol:
            v = self._apply_ham_psi(v, central_tensor_ids)
            v = v / np.linalg.norm(v.tensor)
            delta_v = v.tensor - np.sign(e)[0] * v_.tensor
            v_ = v

        eigen_vectors = v
        return eigen_vectors

    def init_tensors_by_block_hamiltonian(self):
        sequence = get_renormalization_sequence(self.psi.edges, self.psi.top_edge_id)
        for tensor_id in sequence:
            ham = self._get_block_hamiltonian(tensor_id)
            self._set_psi_tensor_with_ham(tensor_id, ham)
            self._set_psi_edge_dim(tensor_id)
            self._set_edge_u1_charge(tensor_id)
            self._set_edge_spin(tensor_id)
            self._set_block_hamiltonian(tensor_id)
        # gauge_tensor
        central_tensor_ids = self.psi.central_tensor_ids()
        c0 = self.edge_u1_charges[self.psi.edges[central_tensor_ids[0]][2]]
        c1 = self.edge_u1_charges[self.psi.edges[central_tensor_ids[1]][2]]
        i0 = Index(U1Charge(c0), flow=True)
        i1 = Index(U1Charge(c1), flow=True)
        u = Index(U1Charge([self.psi.size // 2 + self.u1_num]), flow=False)
        self.psi.gauge_tensor = BlockSparseTensor.random([i0, i1, u])
        self.psi.gauge_tensor = self.psi.gauge_tensor / np.linalg.norm(
            self.psi.gauge_tensor.todense()
        )
        (_, selected_tensor_id, _, _) = self.local_two_tensor()
        iso = tn.Node(self.psi.tensors[selected_tensor_id], backend=self.backend)
        gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)
        if iso.tensor.flat_flows[2]:
            out = gauge[1]
            iso[2] ^ gauge[0]
        else:
            out = gauge[0]
            iso[2] ^ gauge[1]
        iso = tn.contractors.auto(
            [iso, gauge], output_edge_order=[iso[0], iso[1], out, gauge[2]]
        )
        self.psi.tensors[selected_tensor_id] = iso.get_tensor()
        ground_state = self.lanczos(central_tensor_ids)
        u, s, v, _, _ = self.decompose_two_tensors(
            ground_state, self.init_bond_dim,
        )
        self.psi.tensors[central_tensor_ids[0]] = u
        self.psi.tensors[central_tensor_ids[1]] = v
        self.psi.gauge_tensor = s

        self._set_edge_u1_charge(central_tensor_ids[0])
        self._set_edge_u1_charge(central_tensor_ids[1])


    def _apply_ham_psi(self, psi, central_tensor_ids):
        psi_tensor = psi.get_tensor().copy()
        i_ll = Index(psi_tensor.flat_charges[0], flow=psi_tensor.flat_flows[0])
        i_l = Index(psi_tensor.flat_charges[1], flow=psi_tensor.flat_flows[1])
        i_r = Index(psi_tensor.flat_charges[2], flow=psi_tensor.flat_flows[2])
        i_rr = Index(psi_tensor.flat_charges[3], flow=psi_tensor.flat_flows[3])
        i = Index(psi_tensor.flat_charges[4], flow=psi_tensor.flat_flows[4])

        psi_tensor = BlockSparseTensor.zeros([i_ll, i_l, i_r, i_rr, i])
        if self.psi.edges[central_tensor_ids[0]][0] in self.block_hamiltonians.keys():
            psi_tensor += self._block_ham_psi(
                psi, self.psi.edges[central_tensor_ids[0]][0], 0
            )

        if self.psi.edges[central_tensor_ids[0]][1] in self.block_hamiltonians.keys():
            psi_tensor += self._block_ham_psi(
                psi, self.psi.edges[central_tensor_ids[0]][1], 1
            )

        if self.psi.edges[central_tensor_ids[1]][0] in self.block_hamiltonians.keys():
            psi_tensor += self._block_ham_psi(
                psi, self.psi.edges[central_tensor_ids[1]][0], 2
            )

        if self.psi.edges[central_tensor_ids[1]][1] in self.block_hamiltonians.keys():
            psi_tensor += self._block_ham_psi(
                psi, self.psi.edges[central_tensor_ids[1]][1], 3
            )

        psi_tensor += self._ham_psi(
            psi,
            self.psi.edges[central_tensor_ids[0]][:2],
            [0, 1],
        )
        psi_tensor += self._ham_psi(
            psi,
            self.psi.edges[central_tensor_ids[1]][:2],
            [2, 3],
        )
        psi_tensor += self._ham_psi(
            psi,
            [
                self.psi.edges[central_tensor_ids[0]][0],
                self.psi.edges[central_tensor_ids[1]][0],
            ],
            [0, 2],
        )
        psi_tensor += self._ham_psi(
            psi,
            [
                self.psi.edges[central_tensor_ids[0]][0],
                self.psi.edges[central_tensor_ids[1]][1],
            ],
            [0, 3],
        )
        psi_tensor += self._ham_psi(
            psi,
            [
                self.psi.edges[central_tensor_ids[0]][1],
                self.psi.edges[central_tensor_ids[1]][0],
            ],
            [1, 2],
        )
        psi_tensor += self._ham_psi(
            psi,
            [
                self.psi.edges[central_tensor_ids[0]][1],
                self.psi.edges[central_tensor_ids[1]][1],
            ],
            [1, 3],
        )
        return tn.Node(psi_tensor, backend=self.backend)

    def _block_ham_psi(self, psi, edge_id, apply_id):
        h = tn.Node(self.block_hamiltonians[edge_id], backend=self.backend)
        psi_ = psi.copy()
        psi_[apply_id] ^ h[0]

        output_edge_order = psi_.get_all_edges()
        output_edge_order[apply_id] = h[1]

        psi_tensor = tn.contractors.auto(
            [psi_, h], output_edge_order=output_edge_order
        ).get_tensor()
        return psi_tensor

    def _ham_psi(self, psi, edge_ids, apply_ids):
        if len(psi.shape) == 5:
            psi_tensor = psi.get_tensor().copy()
            i_ll = Index(psi_tensor.flat_charges[0], flow=psi_tensor.flat_flows[0])
            i_l = Index(psi_tensor.flat_charges[1], flow=psi_tensor.flat_flows[1])
            i_r = Index(psi_tensor.flat_charges[2], flow=psi_tensor.flat_flows[2])
            i_rr = Index(psi_tensor.flat_charges[3], flow=psi_tensor.flat_flows[3])
            i = Index(psi_tensor.flat_charges[4], flow=psi_tensor.flat_flows[4])
            psi_tensor = BlockSparseTensor.zeros([i_ll, i_l, i_r, i_rr, i])
            out_order = [0, 1, 2, 3, 4]
        if len(psi.shape) == 3:
            psi_tensor = psi.get_tensor().copy()
            i_l = Index(psi_tensor.flat_charges[0], flow=psi_tensor.flat_flows[0])
            i_r = Index(psi_tensor.flat_charges[1], flow=psi_tensor.flat_flows[1])
            i = Index(psi_tensor.flat_charges[2], flow=psi_tensor.flat_flows[2])
            psi_tensor = BlockSparseTensor.zeros([i_l, i_r, i])
            out_order = [0, 1, 2]

        ham_infos = self._get_ham_infos(edge_ids)
        for ham_info in ham_infos:
            edge_order = ham_info["edge_order"]
            observable = ham_info["observable"]
            ham_indices = observable.indices
            for n in range(observable.operators_num):
                operators = observable.operators_list[n]
                charge_indexs = [Index(psi.tensor.flat_charges[i], flow=psi.tensor.flat_flows[i]) for i in range(len(psi_tensor.shape))]
                for i, j in enumerate(edge_order):
                    edge_id = edge_ids[i]
                    bare_edge_id = ham_indices[j]
                    if operators[i] == "S+":
                        c = 1 if psi.tensor.flat_flows[apply_ids[i]] else -1
                        u = psi.tensor.flat_charges[apply_ids[i]].charges.flatten() + c
                        charge_indexs[apply_ids[i]] = Index(U1Charge(u), flow=psi.tensor.flat_flows[apply_ids[i]])
                    if operators[i] == "S-":
                        c = 1 if not psi.tensor.flat_flows[apply_ids[i]] else -1
                        u = psi.tensor.flat_charges[apply_ids[i]].charges.flatten() + c
                        charge_indexs[apply_ids[i]] = Index(U1Charge(u), flow=psi.tensor.flat_flows[apply_ids[i]])
                psi_ = BlockSparseTensor.fromdense(charge_indexs, psi.tensor.todense())
                psi_ = tn.Node(psi_, backend=self.backend)
                output_edge_order = psi_.get_all_edges()
                for i, j in enumerate(edge_order):
                    edge_id = edge_ids[i]
                    bare_edge_id = ham_indices[j]
                    spin_operator = self._spin_operator_at_edge(
                        edge_id, bare_edge_id, operators[i]
                    )
                    spin_operator = tn.Node(spin_operator, backend=self.backend)
                    psi_[apply_ids[i]] ^ spin_operator[0]
                    output_edge_order[apply_ids[i]] = spin_operator[1]
                    psi_ = tn.contractors.auto(
                        [psi_, spin_operator], output_edge_order=output_edge_order
                    )
                psi_tensor += psi_.get_tensor() * observable.coef_list[n]
        return psi_tensor


    def _get_block_hamiltonian(self, tensor_id):
        def fuse_ham(block_ham, u1_charges):
            dims = block_ham.shape[: len(block_ham.shape) // 2]
            dim = np.prod(dims)
            b = Index(U1Charge(U1Charge.fuse(u1_charges[0], u1_charges[1])), flow=False)
            k = b.copy().flip_flow()
            block_ham = BlockSparseTensor.fromdense([b, k], block_ham.reshape([dim] * 2).todense())
            return block_ham
        block_hams = []
        edge_ids = self.psi.edges[tensor_id][:2]
        u1_charges = [self.edge_u1_charges[self.psi.edges[tensor_id][0]], self.edge_u1_charges[self.psi.edges[tensor_id][1]]]
        ham_infos = self._get_ham_infos(edge_ids)
        for ham_info in ham_infos:
            edge_order = ham_info["edge_order"]
            observable = ham_info["observable"]
            ham_indices = observable.indices
            for n in range(observable.operators_num):
                spin_operators = [None, None]
                operators = observable.operators_list[n]
                for i, j in enumerate(edge_order):
                    edge_id = edge_ids[i]

                    bare_edge_id = ham_indices[j]
                    spin_operators[i] = self._spin_operator_at_edge(
                        edge_id, bare_edge_id, operators[i]
                    )
                block_ham = tn.ncon(
                    spin_operators,
                    [["-b0", "-k0"], ["-b1", "-k1"]],
                    out_order=["-b0", "-b1", "-k0", "-k1"],
                    backend=self.backend
                )
                block_ham *= observable.coef_list[n]
                block_ham = fuse_ham(block_ham, u1_charges)
                block_hams.append(block_ham)


        if self.psi.tensors[tensor_id] is None:
            b_l = Index(
                U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][0]]), flow=False
            )
            b_r = Index(
                U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][1]]), flow=False
            )
        else:
            b_l = Index(
                self.psi.tensors[tensor_id].flat_charges[0],
                flow=self.psi.tensors[tensor_id].flat_flows[0],
            )
            b_r = Index(
                self.psi.tensors[tensor_id].flat_charges[1],
                flow=self.psi.tensors[tensor_id].flat_flows[1],
            )
        k_l = b_l.copy().flip_flow()
        k_r = b_r.copy().flip_flow()

        eye_l = np.eye(self.psi.edge_dims[self.psi.edges[tensor_id][0]])
        eye_l = BlockSparseTensor.fromdense([b_l, k_l], eye_l)
        eye_r = np.eye(self.psi.edge_dims[self.psi.edges[tensor_id][1]])
        eye_r = BlockSparseTensor.fromdense([b_r, k_r], eye_r)

        # left block ham
        if self.psi.edges[tensor_id][0] in self.block_hamiltonians.keys():
            block_ham_left = self.block_hamiltonians[self.psi.edges[tensor_id][0]]
            block_ham = tn.ncon(
                [block_ham_left, eye_r],
                [["-b0", "-k0"], ["-b1", "-k1"]],
                out_order=["-b0", "-b1", "-k0", "-k1"],
                backend=self.backend
            )
            block_ham = fuse_ham(block_ham, u1_charges)

            block_hams.append(block_ham)
        # right block ham
        if self.psi.edges[tensor_id][1] in self.block_hamiltonians.keys():
            block_ham_right = self.block_hamiltonians[self.psi.edges[tensor_id][1]]
            block_ham = tn.ncon(
                [eye_l, block_ham_right],
                [["-b0", "-k0"], ["-b1", "-k1"]],
                out_order=["-b0", "-b1", "-k0", "-k1"],
                backend=self.backend
            )
            block_ham = fuse_ham(block_ham, u1_charges)
            block_hams.append(block_ham)

        # if there is no hamiltonian within this block
        if block_hams == []:
            block_hams.append(fuse_ham(np.kron(eye_l, eye_r, backend=self.backend), u1_charges))

        block_hams = np.sum(block_hams, axis=0)
        return block_hams


    def _get_ham_infos(self, edge_ids):
        ham_info_list = []
        l_bare_edges = get_bare_edges(
            edge_ids[0],
            self.psi.edges,
            self.psi.physical_edges,
        )
        r_bare_edges = get_bare_edges(
            edge_ids[1],
            self.psi.edges,
            self.psi.physical_edges,
        )
        for ham in self.hamiltonian.observables:
            output_edge_order, edges = _output_edges_order(
                ham.indices, l_bare_edges, r_bare_edges
            )
            if set(output_edge_order) == set([0, 1]):
                ham_info_list.append(
                    {
                        "edge_order": output_edge_order,
                        "edges": edges,
                        "observable": ham,
                    }
                )
        return ham_info_list

    def _set_psi_tensor_with_ham(self, tensor_id, ham, delta=1e-11):
        lower_edge_dims = [len(self.edge_u1_charges[i]) for i in self.psi.edges[tensor_id][:2]]
        bond_dim = ham.shape[0]
        eta, iso = tn.block_sparse.eigh(ham)
        ind = np.min([self.init_bond_dim, bond_dim])
        eigenvalues = eta.data
        indices = np.argsort(eigenvalues)
        if ind < len(eigenvalues):
            while ind > 1:
                if (
                    np.abs(eigenvalues[indices[ind]] - eigenvalues[indices[ind- 1]])
                    < delta
                ):
                    ind -= 1
                else:
                    break
        # 最終的な固有値と固有ベクトルを選択
        selected_indices = indices[:ind]
        selected_eigenvectors = iso.todense()[:, selected_indices]

        # ここで対応するチャージ数でソートする処理
        charges = eta.charges[0][0].charges.flatten()
        charges = charges[selected_indices]
        sorted_by_charges = np.argsort(charges)

        # チャージに従って並べ直した固有ベクトル
        final_eigenvectors = selected_eigenvectors[:, sorted_by_charges]
        final_eigenvectors = final_eigenvectors.reshape(lower_edge_dims + [ind])

        # 結果を返す、もしくは次の処理に使用
        charges = U1Charge(charges[sorted_by_charges])
        c0 = U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][0]])
        c1 = U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][1]])

        i = Index(charges, flow=False)
        i0 = Index(c0, flow=True)
        i1 = Index(c1, flow=True)
        self.psi.tensors[tensor_id] = BlockSparseTensor.fromdense(
            [i0, i1, i], final_eigenvectors
        )

    def _set_edge_u1_charge(self, tensor_id):
        if self.psi.tensors[tensor_id] is not None:
            self.edge_u1_charges[self.psi.edges[tensor_id][2]] = (
                self.psi.tensors[tensor_id].charges[2][0].charges.flatten()
            )

    def _set_psi_edge_dim(self, tensor_id):
        if self.psi.tensors[tensor_id] is not None:
            self.psi.edge_dims[self.psi.edges[tensor_id][2]] = self.psi.tensors[
                tensor_id
            ].shape[2]

    def _set_edge_spin(self, tensor_id):
        new_spin_operators = {}
        # left edge
        edge_left = self.psi.edges[tensor_id][0]
        edge_right = self.psi.edges[tensor_id][1]
        edge_top = self.psi.edges[tensor_id][2]

        bare_edges = get_bare_edges(edge_left, self.psi.edges, self.psi.physical_edges)
        spin_operators = self.edge_spin_operators[edge_left]
        for bare_edge in bare_edges:
            renormalized_spin_operators = {}
            for key, value in spin_operators[bare_edge].items():
                spin = tn.Node(value, backend=self.backend)
                ket = tn.Node(self.psi.tensors[tensor_id], backend=self.backend).copy(conjugate=True)
                if key == "Sz":
                    bra = ket.copy(conjugate=True)
                elif key == "S+":
                    c = 1 if self.psi.tensors[tensor_id].flat_flows[0] else - 1
                    u_left = self.psi.tensors[tensor_id].flat_charges[0].charges.flatten() + c
                    u_right = self.psi.tensors[tensor_id].flat_charges[1].charges.flatten()
                    c = 1 if not self.psi.tensors[tensor_id].flat_flows[2] else - 1
                    u_up = self.psi.tensors[tensor_id].flat_charges[2].charges.flatten() + c
                    u_left = U1Charge(u_left)
                    u_right = U1Charge(u_right)
                    u_up = U1Charge(u_up)
                    iso_bra = BlockSparseTensor.fromdense([Index(u_left, flow=self.psi.tensors[tensor_id].flat_flows[0]), Index(u_right, flow=self.psi.tensors[tensor_id].flat_flows[1]), Index(u_up, flow=self.psi.tensors[tensor_id].flat_flows[2])], self.psi.tensors[tensor_id].todense())
                    bra = tn.Node(iso_bra, backend=self.backend)
                elif key == "S-":
                    c = 1 if not self.psi.tensors[tensor_id].flat_flows[0] else - 1
                    u_left = self.psi.tensors[tensor_id].flat_charges[0].charges.flatten() + c
                    u_right = self.psi.tensors[tensor_id].flat_charges[1].charges.flatten()
                    c = 1 if self.psi.tensors[tensor_id].flat_flows[2] else - 1
                    u_up = self.psi.tensors[tensor_id].flat_charges[2].charges.flatten() + c
                    u_left = U1Charge(u_left)
                    u_right = U1Charge(u_right)
                    u_up = U1Charge(u_up)
                    iso_bra = BlockSparseTensor.fromdense([Index(u_left, flow=self.psi.tensors[tensor_id].flat_flows[0]), Index(u_right, flow=self.psi.tensors[tensor_id].flat_flows[1]), Index(u_up, flow=self.psi.tensors[tensor_id].flat_flows[2])], self.psi.tensors[tensor_id].todense())
                    bra = tn.Node(iso_bra, backend=self.backend)

                bra[0] ^ spin[0]
                ket[0] ^ spin[1]
                bra[1] ^ ket[1]
                spin = tn.contractors.auto(
                    [bra, spin, ket], output_edge_order=[bra[2], ket[2]]
                )
                renormalized_spin_operators[key] = spin.tensor
            new_spin_operators[bare_edge] = renormalized_spin_operators
        # right edge
        bare_edges = get_bare_edges(edge_right, self.psi.edges, self.psi.physical_edges)
        spin_operators = self.edge_spin_operators[edge_right]
        for bare_edge in bare_edges:
            renormalized_spin_operators = {}
            for key, value in spin_operators[bare_edge].items():
                spin = tn.Node(value, backend=self.backend)
                ket = tn.Node(self.psi.tensors[tensor_id], backend=self.backend).copy(conjugate=True)
                if key == "Sz":
                    bra = ket.copy(conjugate=True)
                elif key == "S+":
                    u_left = self.psi.tensors[tensor_id].flat_charges[0].charges.flatten()
                    c = 1 if self.psi.tensors[tensor_id].flat_flows[1] else - 1
                    u_right = self.psi.tensors[tensor_id].flat_charges[1].charges.flatten() + c
                    c = 1 if not self.psi.tensors[tensor_id].flat_flows[2] else - 1
                    u_up = self.psi.tensors[tensor_id].flat_charges[2].charges.flatten() + c
                    u_left = U1Charge(u_left)
                    u_right = U1Charge(u_right)
                    u_up = U1Charge(u_up)
                    iso_bra = BlockSparseTensor.fromdense([Index(u_left, flow=self.psi.tensors[tensor_id].flat_flows[0]), Index(u_right, flow=self.psi.tensors[tensor_id].flat_flows[1]), Index(u_up, flow=self.psi.tensors[tensor_id].flat_flows[2])], self.psi.tensors[tensor_id].todense())
                    bra = tn.Node(iso_bra, backend=self.backend)
                elif key == "S-":
                    u_left = self.psi.tensors[tensor_id].flat_charges[0].charges.flatten()
                    c = 1 if not self.psi.tensors[tensor_id].flat_flows[1] else - 1
                    u_right = self.psi.tensors[tensor_id].flat_charges[1].charges.flatten() + c
                    c = 1 if self.psi.tensors[tensor_id].flat_flows[2] else - 1
                    u_up = self.psi.tensors[tensor_id].flat_charges[2].charges.flatten() + c
                    u_left = U1Charge(u_left)
                    u_right = U1Charge(u_right)
                    u_up = U1Charge(u_up)
                    iso_bra = BlockSparseTensor.fromdense([Index(u_left, flow=self.psi.tensors[tensor_id].flat_flows[0]), Index(u_right, flow=self.psi.tensors[tensor_id].flat_flows[1]), Index(u_up, flow=self.psi.tensors[tensor_id].flat_flows[2])], self.psi.tensors[tensor_id].todense())
                    bra = tn.Node(iso_bra, backend=self.backend)
                bra[1] ^ spin[0]
                ket[1] ^ spin[1]
                bra[0] ^ ket[0]
                spin = tn.contractors.auto(
                    [bra, spin, ket], output_edge_order=[bra[2], ket[2]]
                )
                renormalized_spin_operators[key] = spin.tensor
            new_spin_operators[bare_edge] = renormalized_spin_operators
        self.edge_spin_operators[self.psi.edges[tensor_id][2]] = new_spin_operators


    def _set_block_hamiltonian(self, tensor_id):
        bra = self.psi.tensors[tensor_id]
        bra_tensor = BlockSparseTensor.zeros(
            [
                Index(bra.flat_charges[0], flow=bra.flat_flows[0]),
                Index(bra.flat_charges[1], flow=bra.flat_flows[1]),
                Index(bra.flat_charges[2], flow=bra.flat_flows[2]),
            ]
        )
        bra = tn.Node(bra, backend=self.backend)
        ket = bra.copy(conjugate=True)
        if self.psi.edges[tensor_id][0] in self.block_hamiltonians.keys():
            bra_tensor += self._block_ham_psi(bra, self.psi.edges[tensor_id][0], 0)
        if self.psi.edges[tensor_id][1] in self.block_hamiltonians.keys():
            bra_tensor += self._block_ham_psi(bra, self.psi.edges[tensor_id][1], 1)
        bra_tensor += self._ham_psi(
            bra, self.psi.edges[tensor_id][:2], [0, 1]
        )
        bra_h = tn.Node(bra_tensor, backend=self.backend)
        bra_h[0] ^ ket[0]
        bra_h[1] ^ ket[1]
        block_ham = tn.contractors.auto(
            [bra_h, ket], output_edge_order=[bra_h[2], ket[2]]
        )
        self.block_hamiltonians[self.psi.edges[tensor_id][2]] = (
            block_ham.get_tensor()
        )


    def _spin_operator_at_edge(self, edge_id, bare_edge_id, operator):
        if operator == "S+":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["S+"]
        elif operator == "S-":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["S-"]
        elif operator == "Sz":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["Sz"]
        elif operator == "Sx":
            sp = self.edge_spin_operators[edge_id][bare_edge_id]["S+"]
            sm = self.edge_spin_operators[edge_id][bare_edge_id]["S-"]
            op = (sp + sm) / 2
        elif operator == "Sy":
            sp = self.edge_spin_operators[edge_id][bare_edge_id]["S+"]
            sm = self.edge_spin_operators[edge_id][bare_edge_id]["S-"]
            op = (sp - sm) / 2.0j
        return op

    def _init_spin_operator(self):
        edge_spin_operators = {}
        for key, value in self.hamiltonian.spin_size.items():
            z_charge = U1Charge([i for i in range(spin_dof(value))])
            plus_charge = U1Charge([i + 1 for i in range(spin_dof(value))])
            minus_charge = U1Charge([i - 1 for i in range(spin_dof(value))])
            edge_spin_operators[key] = {
                key: {
                    "Sz": BlockSparseTensor.fromdense([Index(z_charge, flow=False), Index(z_charge, flow=True)], bare_spin_operator("Sz", value)),
                    "S+": BlockSparseTensor.fromdense([Index(plus_charge, flow=False), Index(z_charge, flow=True)], bare_spin_operator("S+", value)),
                    "S-": BlockSparseTensor.fromdense([Index(minus_charge, flow=False), Index(z_charge, flow=True)], bare_spin_operator("S-", value)),
                }
            }
        return edge_spin_operators

    def _init_block_hamiltonians(self):
        block_hamiltonians = {}
        for key in self.hamiltonian.spin_size.keys():
            for ham in self.hamiltonian.observables:
                if np.array_equal(ham.indices, [key]):
                    spin_operators = []
                    for n in range(ham.operators_num):
                        operators = ham.operators_list[n]
                        spin_operator = self._spin_operator_at_edge(key, key, operators)
                        spin_operator *= ham.coef_list[n]
                        spin_operators.append(spin_operator)
                    block_ham = np.sum(spin_operators, axis=0)
                    block_hamiltonians[key] = block_ham
        return block_hamiltonians

    def _init_edge_u1_charge(self):
        edge_u1_charges = {}
        for key, _ in self.hamiltonian.spin_size.items():
            edge_u1_charges[key] = [0, 1]
        return edge_u1_charges

    def _contraction_rule(
        self,
        initial_edge_ids,
    ):
        causal_cone_ids = {}
        remain_rule = {}
        connect_rule = {}

        for n, initial_edge_id in enumerate(initial_edge_ids):
            if initial_edge_id is None:
                continue
            for i, edge_id in enumerate(self.psi.edges):
                if initial_edge_id in edge_id[:2]:
                    causal_cone_ids[n] = i
                    for ii in range(2):
                        if initial_edge_id == edge_id[ii]:
                            connect_rule[i] = ii
                        else:
                            remain_rule[i] = ii

        return causal_cone_ids, connect_rule, remain_rule

    def contract_central_tensors(self):
        central_tensor_ids = self.psi.central_tensor_ids()

        psi1 = tn.Node(self.psi.tensors[central_tensor_ids[0]], backend=self.backend)
        psi2 = tn.Node(self.psi.tensors[central_tensor_ids[1]], backend=self.backend)
        gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)

        if psi1.tensor.flows[2][0] and not psi2.tensor.flows[2][0]:
            psi1[2] ^ gauge[0]
            gauge[1] ^ psi2[2]
        if not psi1.tensor.flows[2][0] and psi2.tensor.flows[2][0]:
            psi2[2] ^ gauge[0]
            gauge[1] ^ psi1[2]

        psi = tn.contractors.auto(
            [psi1, gauge, psi2],
            output_edge_order=[psi1[0], psi1[1], psi2[0], psi2[1], gauge[2]],
        )
        return psi


def _output_edges_order(bare_edges, left_bare_edges, right_bare_edges):
    output_edges_order = []
    edges = []
    for edge in bare_edges:
        if edge in left_bare_edges:
            output_edges_order.append(0)
            edges.append(edge)
        if edge in right_bare_edges:
            output_edges_order.append(1)
            edges.append(edge)
    return output_edges_order, edges
