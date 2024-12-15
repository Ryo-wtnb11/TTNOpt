from typing import Union
import numpy as np
import tensornetwork as tn
from tensornetwork import U1Charge, Index, BlockSparseTensor
from scipy.linalg import eigh_tridiagonal
from copy import deepcopy
from collections import defaultdict

from ttnopt.src.TTN import TreeTensorNetwork
from ttnopt.src.Hamiltonian import Hamiltonian
from ttnopt.src.Observable import bare_spin_operator, spin_dof, spin_ind
from ttnopt.src.TwoSiteUpdaterSparse import TwoSiteUpdaterSparse
from ttnopt.src.functionTTN import get_renormalization_sequence, get_bare_edges


class PhysicsEngineSparse(TwoSiteUpdaterSparse):

    def __init__(
        self,
        psi: TreeTensorNetwork,
        hamiltonian: Hamiltonian,
        u1_num: Union[int, str],
        init_bond_dim: int,
        max_bond_dim: int,
        energy_degeneracy_threshold: float = 1e-13,
        entanglement_degeneracy_threshold: float = 0.1,
    ):
        """Initialize a PhysicsEngineSparse object.

        Args:
            psi (TreeTensorNetwork): The quantum state.
            hamiltonians (Hamiltonian): Hamiltonian which has a list of Observables.
            u1_num (int or str): The number of U1 charges.
            init_bond_dim (int): Initial bond dimension.
            max_bond_dim (int): Maximum bond dimension.
            edge_spin_operators (Optional(Dict[int, Dict[str, np.ndarray]]): Spin operators at each edge. Defaults to None.
        """

        super().__init__(psi)
        self.hamiltonian = hamiltonian
        self.u1_num = int(2 * spin_ind("S=" + str(u1_num)))
        self.init_bond_dim = init_bond_dim
        self.max_bond_dim = max_bond_dim
        self.edge_u1_charges = self._init_edge_u1_charge()
        self.edge_spin_operators = self._init_spin_operator()
        self.block_hamiltonians = self._init_block_hamiltonians()
        self.previous_id = 0
        self.energy_degeneracy_threshold = energy_degeneracy_threshold
        self.entanglement_degeneracy_threshold = entanglement_degeneracy_threshold

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

    def expval_onesite(self, tensor_id):
        """Calculate the expectation values of the one-site operators.
        Returns:
            The expectation values of the one-site operators of dict.
        """
        one_site_expvals = {}
        indices = [(tensor_id, i) for i in self.psi.edges[tensor_id][:2]]

        bra = tn.Node(self.psi.tensors[tensor_id], backend=self.backend)
        gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)
        if self.previous_id == tensor_id:
            out = gauge[1]
            bra[2] ^ gauge[0]
        else:
            out = gauge[0]
            bra[2] ^ gauge[1]
        bra_tensor = tn.contractors.auto(
            [bra, gauge], output_edge_order=[bra[0], bra[1], out, gauge[2]]
        )
        ket_tensor = bra_tensor.copy(conjugate=True)

        for index in indices:
            tensor_id, edge_id = index
            if edge_id in self.psi.physical_edges:
                expvals = {}
                for operator in ["Sz"]:
                    bra = bra_tensor.copy()
                    ket = ket_tensor.copy()
                    spin = tn.Node(
                        self._spin_operator_at_edge(edge_id, edge_id, operator),
                        backend=self.backend,
                    )
                    if edge_id == self.psi.edges[tensor_id][0]:
                        bra[0] ^ spin[0]
                        bra = tn.contractors.auto(
                            [bra, spin],
                            output_edge_order=[spin[1], bra[1], bra[2], bra[3]],
                        )
                    if edge_id == self.psi.edges[tensor_id][1]:
                        bra[1] ^ spin[0]
                        bra = tn.contractors.auto(
                            [bra, spin],
                            output_edge_order=[bra[0], spin[1], bra[2], bra[3]],
                        )
                    bra[0] ^ ket[0]
                    bra[1] ^ ket[1]
                    bra[2] ^ ket[2]
                    bra[3] ^ ket[3]
                    expvals[operator] = np.real(
                        tn.contractors.auto([bra, ket]).tensor.data
                    )[0]
                one_site_expvals[edge_id] = expvals
        return one_site_expvals

    def expval_twosite(self, tensor_id):
        two_site_expvals = {}
        l_bare_edges = get_bare_edges(
            self.psi.edges[tensor_id][0],
            self.psi.edges,
            self.psi.physical_edges,
        )
        r_bare_edges = get_bare_edges(
            self.psi.edges[tensor_id][1],
            self.psi.edges,
            self.psi.physical_edges,
        )
        psi = tn.Node(self.psi.tensors[tensor_id], backend=self.backend)
        gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)
        if self.previous_id == tensor_id:
            out = gauge[1]
            psi[2] ^ gauge[0]
        else:
            out = gauge[0]
            psi[2] ^ gauge[1]
        bra_tensor = tn.contractors.auto(
            [psi, gauge],
            output_edge_order=[psi[0], psi[1], out, gauge[2]],
        )
        ket_tensor = bra_tensor.copy(conjugate=True)

        pairs = [(i, j) for i in l_bare_edges for j in r_bare_edges]
        for pair in pairs:
            expvals = {}
            for operators in [
                ["Sz", "Sz"],
                ["S+", "S-"],
                ["S-", "S+"],
            ]:
                bra = bra_tensor.copy()
                ket = ket_tensor.copy()
                spin1 = tn.Node(
                    self._spin_operator_at_edge(
                        self.psi.edges[tensor_id][0], pair[0], operators[0]
                    ),
                    backend=self.backend,
                )
                spin2 = tn.Node(
                    self._spin_operator_at_edge(
                        self.psi.edges[tensor_id][1], pair[1], operators[1]
                    ),
                    backend=self.backend,
                )
                if operators != ["Sz", "Sz"]:
                    bra_ = bra.tensor
                    charges = [
                        bra_.flat_charges[i].charges.flatten()
                        for i in range(len(bra.shape))
                    ]
                    charges[0] = spin1.tensor.flat_charges[0].charges.flatten()
                    charges[1] = spin2.tensor.flat_charges[0].charges.flatten()
                    bra_.contiguous(inplace=True)
                    bra_ = BlockSparseTensor(
                        data=bra_.data,
                        charges=[U1Charge(c) for c in charges],
                        flows=bra_.flat_flows,
                    )
                    bra = tn.Node(bra_, backend=self.backend)

                bra[0] ^ spin1[0]
                bra = tn.contractors.auto(
                    [bra, spin1], output_edge_order=[spin1[1], bra[1], bra[2], bra[3]]
                )
                bra[1] ^ spin2[0]
                bra = tn.contractors.auto(
                    [bra, spin2], output_edge_order=[bra[0], spin2[1], bra[2], bra[3]]
                )
                bra[0] ^ ket[0]
                bra[1] ^ ket[1]
                bra[2] ^ ket[2]
                bra[3] ^ ket[3]
                exp_val = np.real(tn.contractors.auto([bra, ket]).tensor.data)[0]
                op_key = (
                    operators[0] + operators[1]
                    if pair[0] > pair[1]
                    else operators[1] + operators[0]
                )
                expvals[op_key] = exp_val
            key = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
            two_site_expvals[key] = expvals
        return two_site_expvals

    def expval_twosite_origin(self, keys):
        tensor_ids = self.psi.central_tensor_ids()
        two_site_expvals = {}
        l_bare_edges = get_bare_edges(
            self.psi.edges[tensor_ids[0]][0],
            self.psi.edges,
            self.psi.physical_edges,
        )
        l_bare_edges += get_bare_edges(
            self.psi.edges[tensor_ids[0]][1],
            self.psi.edges,
            self.psi.physical_edges,
        )
        r_bare_edges = get_bare_edges(
            self.psi.edges[tensor_ids[1]][0],
            self.psi.edges,
            self.psi.physical_edges,
        )
        r_bare_edges += get_bare_edges(
            self.psi.edges[tensor_ids[1]][1],
            self.psi.edges,
            self.psi.physical_edges,
        )
        gauge_tensor = tn.Node(self.psi.gauge_tensor, backend=self.backend)

        left_spins = self._set_edge_spin(tensor_ids[0], save=False)
        right_spins = self._set_edge_spin(tensor_ids[1], save=False)

        pairs = [(i, j) for i in l_bare_edges for j in r_bare_edges]
        pairs = [pair for pair in pairs if pair not in keys]
        for pair in pairs:
            expvals = {}
            for operators in [
                ["Sz", "Sz"],
                ["S+", "S-"],
                ["S-", "S+"],
            ]:
                gauge = gauge_tensor.copy()
                ket_gauge = gauge_tensor.copy(conjugate=True)
                if operators[0] == "S-":
                    left_spin = left_spins[pair[0]]["S+"]
                    left_spin = left_spin.H
                    left_spin.contiguous(inplace=True)
                    left_spin = BlockSparseTensor(
                        data=left_spin.data,
                        charges=[
                            U1Charge(c.charges.flatten() - 2)
                            for c in left_spin.flat_charges
                        ],
                        flows=np.logical_not(left_spin.flat_flows),
                    )
                else:
                    left_spin = left_spins[pair[0]][operators[0]]
                if operators[1] == "S-":
                    right_spin = right_spins[pair[1]]["S+"]
                    right_spin = right_spin.H
                    right_spin.contiguous(inplace=True)
                    right_spin = BlockSparseTensor(
                        data=right_spin.data,
                        charges=[
                            U1Charge(c.charges.flatten() - 2)
                            for c in right_spin.flat_charges
                        ],
                        flows=np.logical_not(right_spin.flat_flows),
                    )
                else:
                    right_spin = right_spins[pair[1]][operators[1]]
                if operators != ["Sz", "Sz"]:
                    gauge_ = gauge.tensor
                    charges = [
                        gauge_.flat_charges[i].charges.flatten()
                        for i in range(len(gauge.shape))
                    ]
                    if tensor_ids[0] == self.previous_id:
                        charges[0] = left_spin.flat_charges[0].charges.flatten()
                        charges[1] = right_spin.flat_charges[0].charges.flatten()
                    else:
                        charges[1] = left_spin.flat_charges[0].charges.flatten()
                        charges[0] = right_spin.flat_charges[0].charges.flatten()

                    gauge_.contiguous(inplace=True)
                    gauge_ = BlockSparseTensor(
                        data=gauge_.data,
                        charges=[U1Charge(c) for c in charges],
                        flows=gauge_.flat_flows,
                    )
                    gauge = tn.Node(gauge_, backend=self.backend)
                spin1 = tn.Node(left_spin, backend=self.backend)
                spin2 = tn.Node(right_spin, backend=self.backend)
                if tensor_ids[0] == self.previous_id:
                    gauge[0] ^ spin1[0]
                    gauge = tn.contractors.auto(
                        [gauge, spin1], output_edge_order=[spin1[1], gauge[1], gauge[2]]
                    )
                    gauge[1] ^ spin2[0]
                    gauge = tn.contractors.auto(
                        [gauge, spin2], output_edge_order=[gauge[0], spin2[1], gauge[2]]
                    )
                else:
                    gauge[1] ^ spin1[0]
                    gauge = tn.contractors.auto(
                        [gauge, spin1], output_edge_order=[gauge[0], spin1[1], gauge[2]]
                    )
                    gauge[0] ^ spin2[0]
                    gauge = tn.contractors.auto(
                        [gauge, spin2], output_edge_order=[spin2[1], gauge[1], gauge[2]]
                    )
                gauge[0] ^ ket_gauge[0]
                gauge[1] ^ ket_gauge[1]
                gauge[2] ^ ket_gauge[2]
                exp_val = np.real(tn.contractors.auto([gauge, ket_gauge]).tensor.data)[
                    0
                ]
                op_key = (
                    operators[0] + operators[1]
                    if pair[0] > pair[1]
                    else operators[1] + operators[0]
                )
                expvals[op_key] = exp_val
            key = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
            two_site_expvals[key] = expvals
        return two_site_expvals

    def lanczos(self, central_tensor_ids, lanczos_tol=1e-13, inverse_tol=1e-7):
        # todo: deal with different bond dimensions

        psi_1 = tn.Node(self.psi.tensors[central_tensor_ids[0]], backend=self.backend)
        psi_2 = tn.Node(self.psi.tensors[central_tensor_ids[1]], backend=self.backend)
        psi_1[2] ^ psi_2[2]
        psi = tn.contractors.auto(
            [psi_1, psi_2],
            output_edge_order=[psi_1[0], psi_1[1], psi_2[0], psi_2[1], psi_1[3]],
        )
        # normalization
        psi = psi / np.linalg.norm(psi.tensor.data)
        psi_ = psi.copy()
        psi_0 = psi.copy()
        dim_n = np.prod(psi.shape)
        alpha = np.zeros(dim_n, dtype=np.float64)
        beta = np.zeros(dim_n, dtype=np.float64)

        psi_w = self._apply_ham_psi(psi, central_tensor_ids)

        alpha[0] = np.real(np.dot(psi_w.tensor.data, psi.tensor.data))
        omega = psi_w.tensor - alpha[0] * psi.tensor

        d = 0
        if dim_n == 1:
            print("-" * 50)
            print("Fail on lanczos: All bond dimensions in canonical center are 1.")
            print(
                "Set more larger bond dimension to run correctly on numerics.max_bond_dimensions."
            )
            print("-" * 50)
            exit()
        else:
            e_old = 0.0
            for j in range(1, dim_n):
                beta[j] = np.linalg.norm(omega.data)
                psi = tn.Node(omega / beta[j], backend=self.backend)
                psi_w = self._apply_ham_psi(psi, central_tensor_ids)
                alpha[j] = np.real(np.dot(psi_w.tensor.data, psi.tensor.data))
                omega = psi_w.tensor - alpha[j] * psi.tensor - beta[j] * psi_.tensor
                psi_ = psi

                if j >= 1:
                    if j > dim_n:
                        break
                    e, v_tilda = eigh_tridiagonal(
                        np.real(alpha[: j + 1]),
                        np.real(beta[1 : j + 1]),
                        select="i",
                        select_range=(0, 0),
                    )
                    energy = e[0]
                    if np.abs(e - e_old) < np.max([1.0, np.abs(e)[0]]) * lanczos_tol:
                        d += 1
                    if j > dim_n or d > 5:
                        max_e, _ = eigh_tridiagonal(
                            np.real(alpha[: j + 1]),
                            np.real(beta[1 : j + 1]),
                            select="a",
                        )
                        max_e = max_e[-1]
                        break
                    e_old = energy

        v_tilda = np.array(v_tilda.flatten(), dtype=np.complex128)
        v = v_tilda[0] * psi_0.tensor
        psi = psi_0
        psi_ = psi_0
        psi_w = self._apply_ham_psi(psi, central_tensor_ids)
        a = np.real(np.dot(psi_w.tensor.data, psi.tensor.data))
        omega = psi_w.tensor - a * psi.tensor
        for k in range(1, len(v_tilda)):
            b = np.linalg.norm(omega.data)
            psi = tn.Node(omega / b, backend=self.backend)
            v += v_tilda[k] * psi.tensor
            psi_w = self._apply_ham_psi(psi, central_tensor_ids)
            a = np.real(np.dot(psi_w.tensor.data, psi.tensor.data))
            omega = psi_w.tensor - a * psi.tensor - b * psi_.tensor
            psi_ = psi

        # check convergence
        v = tn.Node(v, backend=self.backend)
        v_ = self._apply_ham_psi(v, central_tensor_ids)
        e = np.real(np.dot(v_.tensor.data, v.tensor.data)) / np.real(
            np.linalg.norm(v.tensor.data) ** 2
        )
        delta_v = v_.tensor - max_e * v.tensor
        v_tensor = v_.tensor / np.linalg.norm(v_.tensor.data)
        v = tn.Node(v_tensor, backend=self.backend)
        while np.linalg.norm(delta_v.data) > inverse_tol:
            v_ = self._apply_ham_psi(v, central_tensor_ids)
            e = np.real(np.dot(v_.tensor.data, v.tensor.data)) / np.real(
                np.linalg.norm(v.tensor.data) ** 2
            )
            delta_v = v_.tensor - e * v.tensor
            v_tensor = v_.tensor - max_e * v.tensor
            v_tensor = v_tensor / np.linalg.norm(v_tensor.data)
            v = tn.Node(v_tensor, backend=self.backend)

        eigen_vectors = v
        energy = e
        return eigen_vectors, energy

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
        (_, selected_tensor_id, _, not_selected_tensor_id) = self.local_two_tensor()
        c0 = self.psi.tensors[selected_tensor_id].flat_charges[2].charges.flatten()
        c1 = self.psi.tensors[not_selected_tensor_id].flat_charges[2].charges.flatten()
        u = U1Charge.fuse(c0, c1)
        cc = np.count_nonzero(u == self.u1_num)
        if cc == 0:
            print("-" * 50)
            print(f"Fail on RG: There is no sector with M={self.u1_num}.")
            print(
                "Set correct U1 charge to run correctly on numerics.U1_symmetry.magnitude."
            )
            print("Or they need more larger initial bond dimension.")
            print("To rifer see U1 charges sectors on canonical center following:")
            unique_values, counts = np.unique(u, return_counts=True)
            for val, count in zip(unique_values, counts):
                print(f"S: {val}/2, Count: {count}")
            print("-" * 50)
            exit()
        c0 = U1Charge(c0)
        c1 = U1Charge(c1)
        c = U1Charge([self.u1_num])
        self.psi.gauge_tensor = BlockSparseTensor(
            data=np.ones(cc) / np.sum(np.ones(cc)),
            charges=[c0, c1, c],
            flows=[True, True, False],
        )
        self.psi.gauge_tensor = self.psi.gauge_tensor / np.linalg.norm(
            self.psi.gauge_tensor.data
        )
        iso = tn.Node(self.psi.tensors[selected_tensor_id], backend=self.backend)
        gauge = tn.Node(self.psi.gauge_tensor, backend=self.backend)
        out = gauge[1]
        iso[2] ^ gauge[0]
        iso = tn.contractors.auto(
            [iso, gauge], output_edge_order=[iso[0], iso[1], out, gauge[2]]
        )
        self.psi.tensors[selected_tensor_id] = iso.tensor
        ground_state, _ = self.lanczos([selected_tensor_id, not_selected_tensor_id])
        u, s, v, _, _, _ = self.decompose_two_tensors(
            ground_state,
            self.max_bond_dim,
            delta=self.entanglement_degeneracy_threshold,
        )
        self.psi.tensors[selected_tensor_id] = u
        self.psi.tensors[not_selected_tensor_id] = v
        self.psi.gauge_tensor = s
        self.previous_id = selected_tensor_id

    def _apply_ham_psi(self, psi, central_tensor_ids):
        indices = [
            Index(psi.tensor.flat_charges[i], flow=psi.tensor.flat_flows[i])
            for i in range(len(psi.shape))
        ]
        psi_tensor = BlockSparseTensor.zeros(indices, dtype=np.complex128)

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
        def get_psi_tensor(psi, spins, other_spins, keys, apply_ids):
            indices = [
                Index(psi.tensor.flat_charges[i], flow=psi.tensor.flat_flows[i])
                for i in range(len(psi.shape))
            ]
            psi_tensor = BlockSparseTensor.zeros(indices, dtype=np.complex128)
            for pair1, pair2 in keys:
                psi_ = psi.copy().tensor
                spin1 = spins[pair1][pair2]
                spin2 = other_spins[pair1][pair2]
                charges = [
                    psi_.flat_charges[i].charges.flatten()
                    for i in range(len(psi_.shape))
                ]
                if pair2 != "Sz":
                    psi_.contiguous(inplace=True)
                    charges[apply_ids[0]] = spin1.flat_charges[0].charges.flatten()
                    charges[apply_ids[1]] = spin2.flat_charges[0].charges.flatten()
                    psi_ = BlockSparseTensor(
                        data=psi_.data,
                        charges=[U1Charge(c) for c in charges],
                        flows=psi_.flat_flows,
                    )

                psi_ = tn.Node(psi_, backend=self.backend)

                spin_op1 = tn.Node(spin1, backend=self.backend)
                psi_[apply_ids[0]] ^ spin_op1[0]
                output_edge_order = psi_.get_all_edges()
                output_edge_order[apply_ids[0]] = spin_op1[1]
                psi_ = tn.contractors.auto(
                    [psi_, spin_op1], output_edge_order=output_edge_order
                )

                spin_op2 = tn.Node(spin2, backend=self.backend)
                psi_[apply_ids[1]] ^ spin_op2[0]
                output_edge_order = psi_.get_all_edges()
                output_edge_order[apply_ids[1]] = spin_op2[1]
                psi_ = tn.contractors.auto(
                    [psi_, spin_op2], output_edge_order=output_edge_order
                )
                psi_tensor += psi_.tensor
            return psi_tensor

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

        spins = defaultdict(lambda: defaultdict(lambda: None))
        other_spins = defaultdict(lambda: defaultdict(lambda: None))
        keys = []
        if len(l_bare_edges) > len(r_bare_edges):
            for ham in self.hamiltonian.observables:
                if len(ham.indices) == 2:
                    if (
                        ham.indices[0] in l_bare_edges
                        and ham.indices[1] in r_bare_edges
                    ):
                        for i, op_list in enumerate(ham.operators_list):
                            key = (
                                (ham.indices[1], op_list[1]),
                                (op_list[0]),
                            )
                            if spins[key[0]][key[1]] is None:
                                keys.append(key)
                                spins[key[0]][key[1]] = ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[0], ham.indices[0], op_list[0]
                                    )
                                )
                                other_spins[key[0]][key[1]] = (
                                    self._spin_operator_at_edge(
                                        edge_ids[1], ham.indices[1], op_list[1]
                                    )
                                )
                            else:
                                spins[key[0]][key[1]] += ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[0], ham.indices[0], op_list[0]
                                    )
                                )
                    elif (
                        ham.indices[1] in l_bare_edges
                        and ham.indices[0] in r_bare_edges
                    ):
                        for i, op_list in enumerate(ham.operators_list):
                            key = (
                                (ham.indices[0], op_list[0]),
                                (op_list[1]),
                            )
                            if spins[key[0]][key[1]] is None:
                                keys.append(key)
                                spins[key[0]][key[1]] = ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[0], ham.indices[1], op_list[1]
                                    )
                                )
                                other_spins[key[0]][key[1]] = (
                                    self._spin_operator_at_edge(
                                        edge_ids[1], ham.indices[0], op_list[0]
                                    )
                                )
                            else:
                                spins[key[0]][key[1]] += ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[0], ham.indices[1], op_list[1]
                                    )
                                )
            psi_tensor = get_psi_tensor(psi, spins, other_spins, keys, apply_ids)
            return psi_tensor
        else:
            for ham in self.hamiltonian.observables:
                if len(ham.indices) == 2:
                    if (
                        ham.indices[0] in l_bare_edges
                        and ham.indices[1] in r_bare_edges
                    ):
                        for i, op_list in enumerate(ham.operators_list):
                            key = (
                                (ham.indices[0], op_list[0]),
                                (op_list[1]),
                            )
                            if spins[key[0]][key[1]] is None:
                                keys.append(key)
                                spins[key[0]][key[1]] = ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[1], ham.indices[1], op_list[1]
                                    )
                                )
                                other_spins[key[0]][key[1]] = (
                                    self._spin_operator_at_edge(
                                        edge_ids[0], ham.indices[0], op_list[0]
                                    )
                                )
                            else:
                                spins[key[0]][key[1]] += ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[1], ham.indices[1], op_list[1]
                                    )
                                )
                    elif (
                        ham.indices[1] in l_bare_edges
                        and ham.indices[0] in r_bare_edges
                    ):
                        for i, op_list in enumerate(ham.operators_list):
                            key = (
                                (ham.indices[1], op_list[1]),
                                (op_list[0]),
                            )
                            if spins[key[0]][key[1]] is None:
                                keys.append(key)
                                spins[key[0]][key[1]] = ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[1], ham.indices[0], op_list[0]
                                    )
                                )
                                other_spins[key[0]][key[1]] = (
                                    self._spin_operator_at_edge(
                                        edge_ids[0], ham.indices[1], op_list[1]
                                    )
                                )
                            else:
                                spins[key[0]][key[1]] += ham.coef_list[i] * deepcopy(
                                    self._spin_operator_at_edge(
                                        edge_ids[1], ham.indices[0], op_list[0]
                                    )
                                )
            psi_tensor = get_psi_tensor(psi, spins, other_spins, keys, apply_ids[::-1])
            return psi_tensor

    def _get_block_hamiltonian(self, tensor_id):
        def fuse_ham(block_ham, u1_charges):
            dims = block_ham.shape[: len(block_ham.shape) // 2]
            dim = np.prod(dims)
            u = U1Charge(U1Charge.fuse(u1_charges[0], u1_charges[1]))
            block_ham.reshape([dim] * 2)
            block_ham.contiguous(inplace=True)
            block_ham = BlockSparseTensor(
                data=block_ham.data,
                charges=[u, u],
                flows=[False, True],
            )
            return block_ham

        block_hams = []
        edge_ids = self.psi.edges[tensor_id][:2]
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
        if self.psi.tensors[tensor_id] is None:
            u1_charges = [
                self.edge_u1_charges[self.psi.edges[tensor_id][0]],
                self.edge_u1_charges[self.psi.edges[tensor_id][1]],
            ]
        else:
            u1_charges = [
                self.psi.tensors[tensor_id].flat_charges[0],
                self.psi.tensors[tensor_id].flat_charges[1],
            ]
        for ham in self.hamiltonian.observables:
            spin_operators = [None, None]
            if len(ham.indices) == 2:
                if ham.indices[0] in l_bare_edges and ham.indices[1] in r_bare_edges:
                    for n in range(ham.operators_num):
                        operators = ham.operators_list[n]
                        spin_operators[0] = deepcopy(
                            self._spin_operator_at_edge(
                                edge_ids[0], ham.indices[0], operators[0]
                            )
                        )
                        spin_operators[1] = deepcopy(
                            self._spin_operator_at_edge(
                                edge_ids[1], ham.indices[1], operators[1]
                            )
                        )
                        block_ham = tn.ncon(
                            spin_operators,
                            [["-b0", "-k0"], ["-b1", "-k1"]],
                            out_order=["-b0", "-b1", "-k0", "-k1"],
                            backend=self.backend,
                        )
                        block_ham *= complex(ham.coef_list[n])
                        block_ham = fuse_ham(block_ham, u1_charges)
                        block_hams.append(block_ham)

                if ham.indices[1] in l_bare_edges and ham.indices[0] in r_bare_edges:
                    for n in range(ham.operators_num):
                        operators = ham.operators_list[n]

                        spin_operators[0] = deepcopy(
                            self._spin_operator_at_edge(
                                edge_ids[0], ham.indices[1], operators[1]
                            )
                        )
                        spin_operators[1] = deepcopy(
                            self._spin_operator_at_edge(
                                edge_ids[1], ham.indices[0], operators[0]
                            )
                        )

                        block_ham = tn.ncon(
                            spin_operators,
                            [["-b0", "-k0"], ["-b1", "-k1"]],
                            out_order=["-b0", "-b1", "-k0", "-k1"],
                            backend=self.backend,
                        )
                        block_ham *= ham.coef_list[n]
                        block_ham = fuse_ham(block_ham, u1_charges)
                        block_hams.append(block_ham)

        if self.psi.tensors[tensor_id] is None:
            b_l = U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][0]])
            b_r = U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][1]])
        else:
            b_l = U1Charge(self.psi.tensors[tensor_id].flat_charges[0])
            b_r = U1Charge(self.psi.tensors[tensor_id].flat_charges[1])

        b_l = Index(b_l, flow=False)
        b_r = Index(b_r, flow=False)
        k_l = b_l.copy().flip_flow()
        k_r = b_r.copy().flip_flow()

        eye_l = np.eye(
            self.psi.edge_dims[self.psi.edges[tensor_id][0]], dtype=np.complex128
        )

        eye_l = BlockSparseTensor.fromdense([b_l, k_l], eye_l)
        eye_r = np.eye(
            self.psi.edge_dims[self.psi.edges[tensor_id][1]], dtype=np.complex128
        )
        eye_r = BlockSparseTensor.fromdense([b_r, k_r], eye_r)

        # left block ham
        if self.psi.edges[tensor_id][0] in self.block_hamiltonians.keys():
            block_ham_left = self.block_hamiltonians[self.psi.edges[tensor_id][0]]
            block_ham = tn.ncon(
                [block_ham_left, eye_r],
                [["-b0", "-k0"], ["-b1", "-k1"]],
                out_order=["-b0", "-b1", "-k0", "-k1"],
                backend=self.backend,
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
                backend=self.backend,
            )
            block_ham = fuse_ham(block_ham, u1_charges)
            block_hams.append(block_ham)

        # if there is no hamiltonian within this block
        if block_hams == []:
            block_hams.append(
                fuse_ham(np.kron(eye_l, eye_r, backend=self.backend), u1_charges)
            )

        block_hams = np.sum(block_hams, axis=0)
        return block_hams

    def _set_psi_tensor_with_ham(self, tensor_id, ham):
        if self.psi.tensors[tensor_id] is None:
            lower_edge_dims = [
                len(self.edge_u1_charges[i]) for i in self.psi.edges[tensor_id][:2]
            ]
        else:
            lower_edge_dims = [
                len(i) for i in self.psi.tensors[tensor_id].flat_charges[:2]
            ]
        bond_dim = ham.shape[0]
        eta, iso = tn.block_sparse.eigh(ham)
        ind = np.min([self.init_bond_dim, bond_dim])
        eigenvalues = eta.data
        indices = np.argsort(eigenvalues)
        if ind < len(eigenvalues):
            while ind > 1:
                if (
                    np.abs(eigenvalues[indices[ind]] - eigenvalues[indices[ind - 1]])
                    < self.energy_degeneracy_threshold
                ):
                    ind -= 1
                else:
                    break
        selected_indices = indices[:ind]
        selected_eigenvectors = iso.todense()[:, selected_indices]
        charges = eta.flat_charges[0].charges.flatten()
        charges = charges[selected_indices]
        sorted_by_charges = np.argsort(charges)

        final_eigenvectors = selected_eigenvectors[:, sorted_by_charges]
        final_eigenvectors = final_eigenvectors.reshape(lower_edge_dims + [ind])

        charges = U1Charge(charges[sorted_by_charges])
        c0 = U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][0]])
        c1 = U1Charge(self.edge_u1_charges[self.psi.edges[tensor_id][1]])

        i = Index(charges, flow=False)
        i0 = Index(c0, flow=True)
        i1 = Index(c1, flow=True)

        self.psi.tensors[tensor_id] = BlockSparseTensor.fromdense(
            [i0, i1, i], final_eigenvectors
        )
        self.psi.tensors[tensor_id].contiguous(inplace=True)

    def _set_edge_u1_charge(self, tensor_id):
        if self.psi.tensors[tensor_id] is not None:
            self.edge_u1_charges[self.psi.edges[tensor_id][2]] = (
                self.psi.tensors[tensor_id].flat_charges[2].charges.flatten()
            )

    def _set_psi_edge_dim(self, tensor_id):
        if self.psi.tensors[tensor_id] is not None:
            self.psi.edge_dims[self.psi.edges[tensor_id][2]] = self.psi.tensors[
                tensor_id
            ].shape[2]

    def _set_edge_spin(self, tensor_id, save=True):
        new_spin_operators = {}
        # left edge
        edge_left = self.psi.edges[tensor_id][0]
        edge_right = self.psi.edges[tensor_id][1]

        bare_edges = get_bare_edges(edge_left, self.psi.edges, self.psi.physical_edges)
        spin_operators = self.edge_spin_operators[edge_left]
        for bare_edge in bare_edges:
            renormalized_spin_operators = {}
            for key, value in spin_operators[bare_edge].items():
                spin = tn.Node(value, backend=self.backend)
                bra = tn.Node(self.psi.tensors[tensor_id], backend=self.backend)
                ket = bra.copy(conjugate=True)
                bra_tensor = bra.tensor
                charges = [
                    bra_tensor.flat_charges[i].charges.flatten()
                    for i in range(len(bra.shape))
                ]
                charges[0] = spin.tensor.flat_charges[0].charges.flatten()
                if key == "S+":
                    charges[2] = charges[2] + 2
                bra_tensor.contiguous(inplace=True)
                bra = tn.Node(
                    BlockSparseTensor(
                        data=bra_tensor.data,
                        charges=[U1Charge(c) for c in charges],
                        flows=bra_tensor.flat_flows,
                    ),
                    backend=self.backend,
                )
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
                bra = tn.Node(self.psi.tensors[tensor_id], backend=self.backend)
                ket = bra.copy(conjugate=True)
                charges = [
                    bra.tensor.flat_charges[i].charges.flatten()
                    for i in range(len(bra.shape))
                ]
                bra_tensor = bra.tensor
                charges = [
                    bra_tensor.flat_charges[i].charges.flatten()
                    for i in range(len(bra.shape))
                ]
                charges[1] = spin.tensor.flat_charges[0].charges.flatten()
                if key == "S+":
                    charges[2] = charges[2] + 2
                bra_tensor.contiguous(inplace=True)
                bra = tn.Node(
                    BlockSparseTensor(
                        data=bra_tensor.data,
                        charges=[U1Charge(c) for c in charges],
                        flows=bra_tensor.flat_flows,
                    ),
                    backend=self.backend,
                )
                bra[1] ^ spin[0]
                ket[1] ^ spin[1]
                bra[0] ^ ket[0]
                spin = tn.contractors.auto(
                    [bra, spin, ket], output_edge_order=[bra[2], ket[2]]
                )
                renormalized_spin_operators[key] = spin.tensor
            new_spin_operators[bare_edge] = renormalized_spin_operators
        if save:
            self.edge_spin_operators[self.psi.edges[tensor_id][2]] = new_spin_operators
        else:
            return new_spin_operators

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
        bra_tensor += self._ham_psi(bra, self.psi.edges[tensor_id][:2], [0, 1])

        bra_h = tn.Node(bra_tensor, backend=self.backend)
        bra_h[0] ^ ket[0]
        bra_h[1] ^ ket[1]
        block_ham = tn.contractors.auto(
            [bra_h, ket], output_edge_order=[bra_h[2], ket[2]]
        )
        self.block_hamiltonians[self.psi.edges[tensor_id][2]] = block_ham.get_tensor()

    def _spin_operator_at_edge(self, edge_id, bare_edge_id, operator):
        if operator == "S+":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["S+"]
        elif operator == "S-":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["S+"]
            op = op.H
            op.contiguous(inplace=True)
            op = BlockSparseTensor(
                data=op.data,
                charges=[U1Charge(c.charges.flatten() - 2) for c in op.flat_charges],
                flows=np.logical_not(op.flat_flows),
            )
        elif operator == "Sz":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["Sz"]
        elif operator == "Sz2":
            op = self.edge_spin_operators[edge_id][bare_edge_id]["Sz"]
            op = BlockSparseTensor(
                data=np.array(op.data) ** 2,
                charges=op.flat_charges,
                flows=op.flat_flows,
            )
        return op

    def _init_spin_operator(self):
        edge_spin_operators = {}
        for key, value in self.hamiltonian.spin_size.items():
            z_charge = U1Charge(self.edge_u1_charges[key])
            plus_charge = U1Charge(np.array(self.edge_u1_charges[key]) + 2)
            edge_spin_operators[key] = {
                key: {
                    "Sz": BlockSparseTensor.fromdense(
                        [Index(z_charge, flow=False), Index(z_charge, flow=True)],
                        bare_spin_operator("Sz", value),
                    ),
                    "S+": BlockSparseTensor.fromdense(
                        [Index(plus_charge, flow=False), Index(z_charge, flow=True)],
                        bare_spin_operator("S+", value),
                    ),
                }
            }
        return edge_spin_operators

    def _init_block_hamiltonians(self):
        block_hamiltonians = {}
        for ham in self.hamiltonian.observables:
            for key in self.hamiltonian.spin_size.keys():
                if np.array_equal(ham.indices, [key]):
                    spin_operators = []
                    for n in range(ham.operators_num):
                        operators = ham.operators_list[n]
                        spin_operator = deepcopy(
                            self._spin_operator_at_edge(key, key, operators[0])
                        )
                        spin_operator *= ham.coef_list[n]

                        spin_operators.append(spin_operator)
                    block_ham = np.sum(spin_operators, axis=0)
                    if block_hamiltonians.get(key) is None:
                        block_hamiltonians[key] = block_ham
                    else:
                        block_hamiltonians[key] += block_ham
        return block_hamiltonians

    def _init_edge_u1_charge(self):
        edge_u1_charges = {}
        for key, spin in self.hamiltonian.spin_size.items():
            spin_num = spin_dof(spin)
            spin_value = spin_ind(spin)
            s = int(2 * spin_value)
            edge_u1_charges[key] = [-s + int(i * 2) for i in range(spin_num)]
        return edge_u1_charges

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
