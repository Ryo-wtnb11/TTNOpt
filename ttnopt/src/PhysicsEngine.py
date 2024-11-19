from typing import Dict, Optional
import tensornetwork as tn
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import expm

from ttnopt.src.TTN import TreeTensorNetwork
from ttnopt.src.Hamiltonian import Hamiltonian
from ttnopt.src.Observable import bare_spin_operator, spin_dof
from ttnopt.src.TwoSiteUpdater import TwoSiteUpdater
from ttnopt.src.functionTTN import (
    get_renormalization_sequence,
    get_bare_edges,
    inner_product,
)

tn.set_default_backend("numpy")


class PhysicsEngine(TwoSiteUpdater):
    def __init__(
        self,
        psi: TreeTensorNetwork,
        hamiltonian: Hamiltonian,
        init_bond_dim: int,
        max_bond_dim: int,
        truncation_error: float,
        edge_spin_operators: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
        block_hamiltonians: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    ):
        """Initialize a PhysicsEngine object.

        Args:
            psi (TreeTensorNetwork): The quantum state.
            hamiltonians (Hamiltonian): Hamiltonian which has a list of Observables.
            init_bond_dim (int): Initial bond dimension.
            max_bond_dim (int): Maximum bond dimension.
            truncation_error (float): Maximum truncation error.
            edge_spin_operators (Optional(Dict[int, Dict[str, np.ndarray]]): Spin operators at each edge. Defaults to None.
            block_hamiltonians (Optional(Dict[int, Dict[str, np.ndarray]]): Block_hamiltonian at each edge. Defaults to None.
        """

        super().__init__(psi)
        self.hamiltonian = hamiltonian
        self.init_bond_dim = init_bond_dim
        self.max_bond_dim = max_bond_dim
        self.truncation_error = truncation_error
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
            self.init_tensors_by_block_hamiltonian()

    def calculate_expval(self, indices, operators):
        def _calculate_single_expval(index, central_tensor_id, operator):
            bra = tn.Node(self.psi.tensors[central_tensor_id])
            gauge = tn.Node(self.psi.gauge_tensor)
            bra[2] ^ gauge[0]
            bra = tn.contractors.auto(
                [bra, gauge], output_edge_order=[bra[0], bra[1], gauge[1]]
            )
            ket = bra.copy(conjugate=True)
            spin = tn.Node(self._spin_operator_at_edge(index, index, operator))

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

        def _calculate_double_expval(tensor_id, indices, operators, order):
            psi = tn.Node(self.psi.tensors[tensor_id])
            gauge = tn.Node(self.psi.gauge_tensor)
            psi[2] ^ gauge[0]
            bra = tn.contractors.auto(
                [psi, gauge],
                output_edge_order=[psi[0], psi[1], gauge[1]],
            )
            ket = bra.copy()

            spin1 = tn.Node(
                self._spin_operator_at_edge(
                    self.psi.edges[tensor_id][order[0]], indices[0], operators[0]
                )
            )

            spin2 = tn.Node(
                self._spin_operator_at_edge(
                    self.psi.edges[tensor_id][order[1]], indices[1], operators[1]
                )
            )

            output_edge_order = bra.get_all_edges()
            bra[order[0]] ^ spin1[0]
            output_edge_order[order[0]] = spin1[1]
            bra = tn.contractors.auto([bra, spin1], output_edge_order=output_edge_order)
            output_edge_order = bra.get_all_edges()
            bra[order[1]] ^ spin2[0]
            output_edge_order[order[1]] = spin2[1]
            bra = tn.contractors.auto([bra, spin2], output_edge_order=output_edge_order)
            bra[0] ^ ket[0]
            bra[1] ^ ket[1]
            bra[2] ^ ket[2]
            exp_val = tn.contractors.auto([bra, ket])
            return exp_val

        self.distance = self.initial_distance()
        self.flag = self.initial_flag()

        if isinstance(indices, int):  # one-site expectation value
            index = indices
            while self.candidate_edge_ids() != []:
                central_tensor_ids = self.psi.central_tensor_ids()
                if (
                    index == self.psi.edges[central_tensor_ids[0]][0]
                    or index == self.psi.edges[central_tensor_ids[0]][1]
                ):
                    expval = _calculate_single_expval(
                        index, central_tensor_ids[0], operators
                    ).tensor
                elif (
                    index == self.psi.edges[central_tensor_ids[1]][0]
                    or index == self.psi.edges[central_tensor_ids[1]][1]
                ):
                    expval = _calculate_single_expval(
                        index, central_tensor_ids[1], operators
                    ).tensor
                self.move_center()
            return np.real(expval)
        else:  # two-site expectation value

            def check_and_calculate(tensor_id, order):
                return _calculate_double_expval(
                    tensor_id,
                    indices,
                    operators,
                    order,
                )

            while self.candidate_edge_ids() != []:
                central_tensor_ids = self.psi.central_tensor_ids()
                for k in [0, 1]:
                    for order in [(0, 1), (1, 0)]:
                        if indices[order[0]] in get_bare_edges(
                            self.psi.edges[central_tensor_ids[k]][0],
                            self.psi.edges,
                            self.psi.physical_edges,
                        ) and indices[order[1]] in get_bare_edges(
                            self.psi.edges[central_tensor_ids[k]][1],
                            self.psi.edges,
                            self.psi.physical_edges,
                        ):
                            expval = check_and_calculate(
                                central_tensor_ids[k], list(order)
                            ).tensor

                self.move_center()
            return np.real(expval)

    def energy(self):
        central_tensor_ids = self.psi.central_tensor_ids()
        psi_ = self.contract_central_tensors()
        psi_h = psi_.copy()
        psi_h = self._apply_ham_psi(psi_h, central_tensor_ids)
        return np.real(inner_product(psi_, psi_h))

    def move_center(self):
        (
            edge_id,
            selected_tensor_id,
            connected_tensor_id,
            not_selected_tensor_id,
        ) = self.local_two_tensor()

        # absorb gauge tensor
        iso = tn.Node(self.psi.tensors[selected_tensor_id])
        gauge = tn.Node(self.psi.gauge_tensor)
        iso[2] ^ gauge[0]
        iso = tn.contractors.auto(
            [iso, gauge], output_edge_order=[iso[0], iso[1], gauge[1]]
        )
        self.psi.tensors[selected_tensor_id] = iso.get_tensor()

        self.set_flag(not_selected_tensor_id)

        self.set_ttn_properties_at_one_tensor(edge_id, selected_tensor_id)

        self._set_edge_spin(not_selected_tensor_id)
        self._set_block_hamiltonian(not_selected_tensor_id)

        psi1 = tn.Node(self.psi.tensors[selected_tensor_id])
        psi2 = tn.Node(self.psi.tensors[connected_tensor_id])
        psi1[2] ^ psi2[2]
        psi = tn.contractors.auto(
            [psi1, psi2], output_edge_order=[psi1[0], psi1[1], psi2[0], psi2[1]]
        )
        u, s, v, _, _, edge_order = self.decompose_two_tensors(
            psi, self.max_bond_dim, operate_degeneracy=True
        )
        psi_edges = (
            self.psi.edges[selected_tensor_id][:2]
            + self.psi.edges[connected_tensor_id][:2]
        )

        self.psi.tensors[selected_tensor_id] = u
        self.psi.tensors[connected_tensor_id] = v
        self.psi.gauge_tensor = s
        (
            self.psi.edges[selected_tensor_id][0],
            self.psi.edges[selected_tensor_id][1],
        ) = (
            psi_edges[edge_order[0]],
            psi_edges[edge_order[1]],
        )
        (
            self.psi.edges[connected_tensor_id][0],
            self.psi.edges[connected_tensor_id][1],
        ) = (
            psi_edges[edge_order[2]],
            psi_edges[edge_order[3]],
        )
        self.distance = self.initial_distance()

    def lanczos(
        self, central_tensor_ids, lanczos_tol=1e-13, inverse_tol=1e-6, init_random=False
    ):
        if (
            self.psi.tensors[central_tensor_ids[0]].shape[2]
            != self.psi.tensors[central_tensor_ids[1]].shape[2]
        ):
            psi_1_shape = self.psi.tensors[central_tensor_ids[0]].shape
            psi_2_shape = self.psi.tensors[central_tensor_ids[1]].shape
            if psi_2_shape[2] < psi_1_shape[2]:
                self.psi.tensors[central_tensor_ids[0]] = self.psi.tensors[
                    central_tensor_ids[0]
                ][:, :, : psi_2_shape[2]]
            elif psi_1_shape[2] < psi_2_shape[2]:
                self.psi.tensors[central_tensor_ids[1]] = self.psi.tensors[
                    central_tensor_ids[1]
                ][:, :, : psi_1_shape[2]]

        psi_1 = tn.Node(self.psi.tensors[central_tensor_ids[0]])
        psi_2 = tn.Node(self.psi.tensors[central_tensor_ids[1]])
        psi_1[2] ^ psi_2[2]

        psi = tn.contractors.auto(
            [psi_1, psi_2], output_edge_order=[psi_1[0], psi_1[1], psi_2[0], psi_2[1]]
        )
        if init_random:
            psi = tn.Node(np.random.rand(*psi.shape))

        # normalization
        psi = psi / np.linalg.norm(psi.tensor)
        psi_ = psi.copy()
        psi_0 = psi.copy()
        dim_n = np.prod(psi.shape)
        alpha = np.zeros(dim_n, dtype=np.float64)
        beta = np.zeros(dim_n, dtype=np.float64)

        psi_w = self._apply_ham_psi(psi, central_tensor_ids)

        alpha[0] = np.real(inner_product(psi_w, psi))
        omega = psi_w.tensor - np.array(alpha[0]) * psi.tensor

        d = 0
        if dim_n == 1:
            eigen_vectors = psi
            return eigen_vectors, self.energy()
        else:
            e_old = 0.0
            for j in range(1, dim_n):
                beta[j] = np.linalg.norm(omega)
                if j == 1 and beta[j] < 1e-14:
                    eigen_vectors = psi
                    return eigen_vectors, self.energy()
                elif j > 1 and beta[j] < 1e-14:
                    break
                psi = tn.Node(omega / beta[j])
                psi_w = self._apply_ham_psi(psi, central_tensor_ids)
                alpha[j] = np.real(inner_product(psi_w, psi))
                omega = psi_w.tensor - alpha[j] * psi.tensor - beta[j] * psi_.tensor
                psi_ = psi
                if j >= 1:
                    e, v_tilda = eigh_tridiagonal(
                        np.real(alpha[: j + 1]),
                        np.real(beta[1 : j + 1]),
                        select="i",
                        select_range=(0, 0),
                    )
                    energy = e[0]
                    if np.abs(e - e_old) < np.max([1.0, np.abs(e)]) * lanczos_tol:
                        d += 1
                    if d > 5:
                        break
                    e_old = energy

        v_tilda = np.array(v_tilda.flatten(), dtype=np.complex128)
        v = v_tilda[0] * psi_0.tensor
        psi = psi_0
        psi_ = psi_0
        psi_w = self._apply_ham_psi(psi, central_tensor_ids)
        a = np.real(inner_product(psi_w, psi))
        omega = psi_w.tensor - a * psi.tensor
        for k in range(1, len(v_tilda)):
            b = np.linalg.norm(omega)
            psi = tn.Node(omega / b)
            v += v_tilda[k] * psi.tensor
            psi_w = self._apply_ham_psi(psi, central_tensor_ids)
            a = np.real(inner_product(psi_w, psi))
            omega = psi_w.tensor - a * psi.tensor - b * psi_.tensor
            psi_ = psi

        # check convergence
        v = tn.Node(v)
        v_ = self._apply_ham_psi(v, central_tensor_ids)
        v_ = v_ / np.linalg.norm(v_.tensor)
        delta_v = v_.tensor - np.sign(e)[0] * v.tensor
        while np.linalg.norm(delta_v) > inverse_tol:
            v = self._apply_ham_psi(v, central_tensor_ids)
            v = v / np.linalg.norm(v.tensor)
            delta_v = v.tensor - np.sign(e)[0] * v_.tensor
            v_ = v

        eigen_vectors = v
        return eigen_vectors, energy

    def lanczos_exp_multiply(self, central_tensor_ids, dt, tol=1e-10):
        psi_1 = tn.Node(self.psi.tensors[central_tensor_ids[0]])
        psi_2 = tn.Node(self.psi.tensors[central_tensor_ids[1]])
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

        alpha[0] = np.real(inner_product(psi_w, psi))
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
                psi = tn.Node(omega / beta[j])
                psi_w = self._apply_ham_psi(psi, central_tensor_ids)
                alpha[j] = np.real(inner_product(psi_w, psi))
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
        a = np.real(inner_product(psi_w, psi))
        omega = psi_w.tensor - a * psi.tensor
        for k in range(1, len(v_tilda)):
            b = np.linalg.norm(omega)
            psi = tn.Node(omega / b)
            v += v_tilda[k] * psi.tensor
            psi_w = self._apply_ham_psi(psi, central_tensor_ids)
            a = np.real(inner_product(psi_w, psi))
            omega = psi_w.tensor - a * psi.tensor - b * psi_.tensor
            psi_ = psi

        # check convergence
        v = tn.Node(v)
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
        sequence = get_renormalization_sequence(
            self.psi.edges, self.psi.canonical_center_edge_id
        )
        for tensor_id in sequence:
            ham = self._get_block_hamiltonian(tensor_id)
            self._set_psi_tensor_with_ham(tensor_id, ham)
            self._set_psi_edge_dim(tensor_id)
            self._set_edge_spin(tensor_id)
            self._set_block_hamiltonian(tensor_id, ham)
        # gauge_tensor
        central_tensor_ids = self.psi.central_tensor_ids()
        ground_state, _ = self.lanczos(central_tensor_ids)
        u, s, v, _, _, _ = self.decompose_two_tensors(
            ground_state, self.max_bond_dim, operate_degeneracy=True
        )
        self.psi.tensors[central_tensor_ids[0]] = u
        self.psi.tensors[central_tensor_ids[1]] = v
        self.psi.gauge_tensor = s

    def _apply_ham_psi(self, psi, central_tensor_ids):
        psi_tensor = np.zeros(psi.shape, dtype=np.complex128)

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
            psi, self.psi.edges[central_tensor_ids[0]][:2], [0, 1]
        )
        psi_tensor += self._ham_psi(
            psi, self.psi.edges[central_tensor_ids[1]][:2], [2, 3]
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
        return tn.Node(psi_tensor)

    def _block_ham_psi(self, psi, edge_id, apply_id):
        h = tn.Node(self.block_hamiltonians[edge_id])
        psi_ = psi.copy()
        psi_[apply_id] ^ h[0]

        output_edge_order = psi_.get_all_edges()
        output_edge_order[apply_id] = h[1]

        psi_tensor = tn.contractors.auto(
            [psi_, h], output_edge_order=output_edge_order
        ).get_tensor()
        return psi_tensor

    def _ham_psi(self, psi, edge_ids, apply_ids):
        psi_tensor = np.zeros(psi.shape, dtype=np.complex128)
        ham_infos = self._get_ham_infos(edge_ids)
        for ham_info in ham_infos:
            edge_order = ham_info["edge_order"]
            observable = ham_info["observable"]
            ham_indices = observable.indices
            for n in range(observable.operators_num):
                operators = observable.operators_list[n]
                psi_ = psi.copy()
                output_edge_order = psi_.get_all_edges()
                for i, j in enumerate(edge_order):
                    edge_id = edge_ids[i]
                    bare_edge_id = ham_indices[j]
                    spin_operator = self._spin_operator_at_edge(
                        edge_id, bare_edge_id, operators[i]
                    )
                    spin_operator = tn.Node(spin_operator)
                    psi_[apply_ids[i]] ^ spin_operator[0]
                    output_edge_order[apply_ids[i]] = spin_operator[1]
                    psi_ = tn.contractors.auto(
                        [psi_, spin_operator], output_edge_order=output_edge_order
                    )
                psi_tensor += psi_.get_tensor() * observable.coef_list[n]
        return psi_tensor

    def _get_block_hamiltonian(self, tensor_id):
        block_hams = []

        edge_ids = self.psi.edges[tensor_id][:2]
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
                )
                block_ham *= observable.coef_list[n]
                block_hams.append(block_ham)

        # left block ham
        if self.psi.edges[tensor_id][0] in self.block_hamiltonians.keys():
            block_ham_left = self.block_hamiltonians[self.psi.edges[tensor_id][0]]
            eye = np.eye(self.psi.edge_dims[self.psi.edges[tensor_id][1]])
            block_ham = tn.ncon(
                [block_ham_left, eye],
                [["-b0", "-k0"], ["-b1", "-k1"]],
                out_order=["-b0", "-b1", "-k0", "-k1"],
            )
            block_hams.append(block_ham)
        # right block ham
        if self.psi.edges[tensor_id][1] in self.block_hamiltonians.keys():
            block_ham_right = self.block_hamiltonians[self.psi.edges[tensor_id][1]]
            eye = np.eye(self.psi.edge_dims[self.psi.edges[tensor_id][0]])
            block_ham = tn.ncon(
                [eye, block_ham_right],
                [["-b0", "-k0"], ["-b1", "-k1"]],
                out_order=["-b0", "-b1", "-k0", "-k1"],
            )
            block_hams.append(block_ham)

        # if there is no hamiltonian within this block
        if block_hams == []:
            eye_l = np.eye(
                self.psi.edge_dims[self.psi.edges[tensor_id][0]], dtype=np.complex128
            )
            eye_r = np.eye(
                self.psi.edge_dims[self.psi.edges[tensor_id][1]], dtype=np.complex128
            )
            block_hams.append(np.kron(eye_l, eye_r))

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
            output_edge_order = _output_edges_order(
                ham.indices, l_bare_edges, r_bare_edges
            )
            if set(output_edge_order) == set([0, 1]):
                ham_info_list.append(
                    {"edge_order": output_edge_order, "observable": ham}
                )
        return ham_info_list

    def _set_psi_tensor_with_ham(self, tensor_id, ham, delta=1e-11):
        lower_edge_dims = ham.shape[: len(ham.shape) // 2]
        bond_dim = np.prod(lower_edge_dims)
        ind = np.min([self.init_bond_dim, bond_dim])
        ham = ham.reshape(bond_dim, bond_dim)
        eigenvalues, eigenvectors = scipy.linalg.eigh(ham)
        if ind < len(eigenvalues):
            while ind > 1:
                if np.abs(eigenvalues[ind] - eigenvalues[ind - 1]) < delta:
                    ind -= 1
                else:
                    break
        if ind == 1:
            # if the bond dimension is too small, we need to increase it
            # stop program execution
            raise ValueError("initial bond dimension is too small.")
        isometry = eigenvectors[:, :ind]
        isometry = eigenvectors[:, :ind].reshape(lower_edge_dims + (ind,))
        self.psi.tensors[tensor_id] = isometry

    def _set_psi_edge_dim(self, tensor_id):
        if self.psi.tensors[tensor_id] is not None:
            for i, d in enumerate(self.psi.tensors[tensor_id].shape):
                self.psi.edge_dims[self.psi.edges[tensor_id][i]] = d

    def _set_edge_spin(self, tensor_id):
        new_spin_operators = {}
        # left edge
        edge_left = self.psi.edges[tensor_id][0]
        bare_edges = get_bare_edges(edge_left, self.psi.edges, self.psi.physical_edges)
        spin_operators = self.edge_spin_operators[edge_left]
        for bare_edge in bare_edges:
            renormalized_spin_operators = {}
            for key, value in spin_operators[bare_edge].items():
                bra = tn.Node(self.psi.tensors[tensor_id])
                ket = bra.copy(conjugate=True)
                spin = tn.Node(value)
                bra[0] ^ spin[0]
                ket[0] ^ spin[1]
                bra[1] ^ ket[1]
                spin = tn.contractors.auto(
                    [bra, spin, ket], output_edge_order=[bra[2], ket[2]]
                )
                renormalized_spin_operators[key] = spin.get_tensor()
            new_spin_operators[bare_edge] = renormalized_spin_operators

        # right edge
        edge_right = self.psi.edges[tensor_id][1]
        bare_edges = get_bare_edges(edge_right, self.psi.edges, self.psi.physical_edges)
        spin_operators = self.edge_spin_operators[edge_right]
        for bare_edge in bare_edges:
            renormalized_spin_operators = {}
            for key, value in spin_operators[bare_edge].items():
                bra = tn.Node(self.psi.tensors[tensor_id])
                ket = bra.copy(conjugate=True)
                spin = tn.Node(value)
                bra[1] ^ spin[0]
                ket[1] ^ spin[1]
                bra[0] ^ ket[0]
                spin = tn.contractors.auto(
                    [bra, spin, ket], output_edge_order=[bra[2], ket[2]]
                )
                renormalized_spin_operators[key] = spin.get_tensor()
            new_spin_operators[bare_edge] = renormalized_spin_operators
        self.edge_spin_operators[self.psi.edges[tensor_id][2]] = new_spin_operators

    def _set_block_hamiltonian(self, tensor_id, ham=None):
        if ham is not None:
            bra = self.psi.tensors[tensor_id]
            bra = tn.Node(bra)
            ket = bra.copy(conjugate=True)
            ham = tn.Node(ham)
            ham[0] ^ bra[0]
            ham[1] ^ bra[1]
            ham[2] ^ ket[0]
            ham[3] ^ ket[1]
            block_ham = tn.contractors.auto(
                [bra, ham, ket], output_edge_order=[bra[2], ket[2]]
            )
            self.block_hamiltonians[self.psi.edges[tensor_id][2]] = (
                block_ham.get_tensor()
            )
        else:
            bra = self.psi.tensors[tensor_id]
            bra_tensor = np.zeros(bra.shape, dtype=np.complex128)
            bra = tn.Node(bra)
            ket = bra.copy(conjugate=True)
            if self.psi.edges[tensor_id][0] in self.block_hamiltonians.keys():
                bra_tensor += self._block_ham_psi(bra, self.psi.edges[tensor_id][0], 0)
            if self.psi.edges[tensor_id][1] in self.block_hamiltonians.keys():
                bra_tensor += self._block_ham_psi(bra, self.psi.edges[tensor_id][1], 1)
            bra_tensor += self._ham_psi(bra, self.psi.edges[tensor_id][:2], [0, 1])
            bra_h = tn.Node(bra_tensor)
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
            op = self.edge_spin_operators[edge_id][bare_edge_id]["S+"].conj().T
        elif operator == "Sz":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["Sz"]
        elif operator == "Sx":
            sp = self.edge_spin_operators[edge_id][bare_edge_id]["S+"]
            sm = self.edge_spin_operators[edge_id][bare_edge_id]["S+"].conj().T
            op = (sp + sm) / 2
        elif operator == "Sy":
            sp = self.edge_spin_operators[edge_id][bare_edge_id]["S+"]
            sm = self.edge_spin_operators[edge_id][bare_edge_id]["S+"].conj().T
            op = (sp - sm) / 2.0j
        elif operator == "Sz^2":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["Sz"]
            op = np.dot(op, op)
        return op

    def _init_spin_operator(self):
        edge_spin_operators = {}
        for key, value in self.hamiltonian.spin_size.items():
            edge_spin_operators[key] = {
                key: {
                    "S+": bare_spin_operator("S+", value),
                    "Sz": bare_spin_operator("Sz", value),
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
                        spin_operator = self._spin_operator_at_edge(
                            key, key, operators[0]
                        )
                        spin_operator *= ham.coef_list[n]
                        spin_operators.append(spin_operator)
                    block_ham = np.sum(spin_operators, axis=0)
                    block_hamiltonians[key] = block_ham
        return block_hamiltonians


def _output_edges_order(bare_edges, left_bare_edges, right_bare_edges):
    output_edges_order = []
    for edge in bare_edges:
        if edge in left_bare_edges:
            output_edges_order.append(0)
        if edge in right_bare_edges:
            output_edges_order.append(1)
    return output_edges_order
