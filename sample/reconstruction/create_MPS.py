import numpy as np
from scipy.stats import multivariate_normal
import os
import random
import networkx
import matplotlib.pyplot as plt
import copy
import tqdm
import opt_einsum as oe

def calculate_f(index, cov, l):
    """Calculate f(x) = exp(-x^T cov x).

    Args:
        x (list[int]): The index of the point.
        D (int): The number of dimensions.
        cov (np.array): The covariance matrix of the gaussian distribution.
    """
    x_min = -l/2
    x_max = l/2
    space = (x_max - x_min) / 2**n
    x = np.array(index) * space + x_min
    return np.exp(-np.dot(x, np.dot(np.linalg.inv(cov), x))/2.0)

def create_candidates(I_before, pdim, add_last=True):
    candidates = []
    for index in I_before:
        for i in range(pdim):
            if add_last:
                candidates.append(index + [i])
            else:
                candidates.append([i] + index)
    return candidates

def sample_from_candidates(I_before, pdim, bdim, add_last=True):
    candidates = create_candidates(I_before, pdim, add_last)
    if len(candidates) <= bdim:
        return candidates
    else:
        return random.sample(candidates, bdim)
    
def maxvol(Q, max_iter=100, threshold=1e-5):
    """Given n time r matrix Q, find r times r dominant submatrix.
    Reference: https://www.worldscientific.com/doi/abs/10.1142/9789812836021_0015

    Args:
        Q (np.ndarray) : n times r matrix
        max_iter (int) : maximum iteration for maxvol algorithm
        threshold (float) : threshold for maxvol algorithm
    """
    # Q should be orthogonal matrix earned by QR decomposition
    n, r = Q.shape
    if n <= r:
        return [i for i in range(n)]
    I = [i for i in range(n)]
    J = [i for i in range(r)]
    iter = 0
    while iter < max_iter:
        Qdom = Q[np.ix_(I[:r],J[:r])]
        B = np.dot(Q, np.linalg.inv(Qdom))
        i, j = np.unravel_index(np.argmax(np.abs(B)), B.shape)
        if np.abs(B[i,j]) < 1.0 + threshold:
            break
        else:
            # swap row i and j
            I[i], I[j] = I[j], I[i]
        iter += 1
    return I[:r]

def TCI(get_amplitude, p_dim, D, tci_chi, max_iter=10, progress_bar=True):
    """
    Args:
        get_amplitude (Callable) : return amplitude of the state given tuple (i1,...,in)
        p_dim (int) : dimension of physical index
        D (int) : number of indices, > 2
        tci_chi (int) : bond dimension
        apply_maxvol (bool) : if True, apply ALS_maxvol algorithm
        max_iter (int) : maximum iteration for ALS_maxvol algorithm, >= 1

    Return:
        mps (BinaryMPS) : MPS of the state
    """

    assert D > 2

    bdim_list = []
    for i in range(D-1):
        bdim_list.append(min(tci_chi, p_dim**(i+1), p_dim**(D-i-1)))
    
    Is = [[[]]]
    for i in range(D-1):
        Is.append(sample_from_candidates(Is[-1], p_dim, bdim_list[i]))
    Is += [[]]

    Js = [[[]]]
    for i in range(D-1):
        Js.append(sample_from_candidates(Js[-1], p_dim, bdim_list[D-i-2], add_last=False))
    Js = [[], []] + list(reversed(Js))

    # apply ALS_maxvol
    past_Is, past_Js = copy.deepcopy(Is), copy.deepcopy(Js)
    iteration = 0
    mps_tensors = [None for _ in range(D)]
    for _ in tqdm.trange(max_iter, disable=not progress_bar):
        # sweep from left to right and update Is
        for i in range(1, D):
            # create Talpha from get_amplitude
            T = np.zeros([len(Is[i-1])*p_dim, len(Js[i+1])], dtype=np.complex128)
            for b in range(T.shape[0]):
                for c in range(T.shape[1]):
                    T[b,c] = get_amplitude(Is[i-1][b//p_dim] + [b%p_dim] + Js[i+1][c])

            # decompose Talpha and get new Is
            Q, R = np.linalg.qr(T)
            maxvol_idx_list = maxvol(Q)
            Is[i] = [Is[i-1][idx//p_dim] + [idx%p_dim] for idx in maxvol_idx_list]

            # calculate Q_submatrix
            Qt = np.zeros([Q.shape[1], Q.shape[1]], dtype=np.complex128)
            for b in range(Qt.shape[0]):
                for c in range(Qt.shape[1]):
                    Qt[b,c] = Q[maxvol_idx_list[b], c]
            assert np.allclose(Qt, Q[np.ix_(maxvol_idx_list, range(Q.shape[1]))])

            # calculate mps_tensors using Q
            Qt = np.linalg.pinv(Qt)
            mps_tensors[i-1] = np.dot(Q, Qt).reshape(len(Is[i-1]), p_dim, len(Js[i+1])).transpose(1,0,2)

        # add last mps tensor
        T = np.zeros([p_dim, bdim_list[-1], 1], dtype=np.complex128)
        for a in range(T.shape[0]):
            for b in range(T.shape[1]):
                T[a,b,0] = get_amplitude(Is[D-1][b] + [a])
        mps_tensors[-1] = T
        
        # sweep from right to left and update Js
        for i in range(D, 1, -1):
            # create Talpa from get_amplitude
            T = np.zeros([len(Is[i-1]), len(Js[i+1])*p_dim], dtype=np.complex128)
            for b in range(T.shape[0]):
                for c in range(T.shape[1]):
                    T[b,c] = get_amplitude(Is[i-1][b] + [c%p_dim] + Js[i+1][c//p_dim])

            # decompose Talpa and get new Js
            Q, R = np.linalg.qr(T.T)
            maxvol_idx_list = maxvol(Q)
            Js[i] = [[idx%p_dim] + Js[i+1][idx//p_dim] for idx in maxvol_idx_list]

            # calculate Q_submatrix
            Qt = np.zeros([Q.shape[1], Q.shape[1]], dtype=np.complex128)
            for b in range(Qt.shape[0]):
                for c in range(Qt.shape[1]):
                    Qt[b,c] = Q[maxvol_idx_list[b], c]
            assert np.allclose(Qt, Q[np.ix_(maxvol_idx_list, range(Q.shape[1]))])

            # calculate mps_tensors using Q
            Qt = np.linalg.pinv(Qt)
            mps_tensors[i-1] = np.dot(Q, Qt).reshape(len(Js[i+1]), p_dim, len(Is[i-1])).transpose(1,2,0)

        # add first mps tensor
        T = np.zeros([p_dim, 1, bdim_list[0]], dtype=np.complex128)
        for a in range(T.shape[0]):
            for b in range(T.shape[2]):
                T[a,0,b] = get_amplitude([a] + Js[2][b])
        mps_tensors[0] = T

        if Is == past_Is and Js == past_Js:
            break
        else:
            past_Is, past_Js = copy.deepcopy(Is), copy.deepcopy(Js)
        iteration += 1

    return mps_tensors

def create_TCI_mps_tensors(n, D, l, tci_chi):
    callable = lambda x: calculate_f(x, cov, l)
    TCI_mps_tensors = TCI(callable, 2**n, D, tci_chi)
    return TCI_mps_tensors

def format_mps_tensors(mps_tensors, n):
    new_mps_tensors = []
    tensor = oe.contract("ab,cbd->acd", mps_tensors[0].reshape(mps_tensors[0].shape[0],-1), mps_tensors[1])
    U, s, Vh = np.linalg.svd(tensor.reshape(-1, tensor.shape[-1]), full_matrices=False)
    new_mps_tensors.append(U.reshape(tensor.shape[0], tensor.shape[1], -1))
    for i in range(2, len(mps_tensors)-2):
        tensor = oe.contract("ab,bc,dce->ade", np.diag(s), Vh, mps_tensors[i])
        U, s, Vh = np.linalg.svd(tensor.reshape(-1, tensor.shape[-1]), full_matrices=False)
        new_mps_tensors.append(U.reshape(tensor.shape[0], tensor.shape[1], -1))
    tensor = oe.contract("ab,bc,dce,fe->adf", np.diag(s), Vh, mps_tensors[-2], mps_tensors[-1].reshape(mps_tensors[-1].shape[0], -1))
    U, s, Vh = np.linalg.svd(tensor.reshape(-1, tensor.shape[-2] * tensor.shape[-1]), full_matrices=False)
    S = np.dot(U, np.diag(s))
    new_mps_tensors.append(Vh.T.reshape(2**n, 2**n, -1))
    return new_mps_tensors, S

def save_edges_file(D):
    # create edge list
    edges = []
    edges.append(["0", "1", str(D)])
    for idx in range(2, D-2):
        edges.append([str(D+idx-2), str(idx), str(D+idx-1)])
    edges.append([str(D-2), str(D-1), str(2*D-4)])
    
    with open("normal_distribution_data/edges.dat", "w") as f:
        for e in edges:
            string = ",".join(e)
            f.write(string + "\n")


n = 4
D = 16
l = 10
chi = 16

# load covariance matrix
cov = np.load("normal_distribution_data/cov/cov.npy")

# create MPS that represents f(x) using TCI
print("creating TCI_mps_tensors...")
mps_tensors = create_TCI_mps_tensors(n, D, l, chi)

# convert MPS to the TreeTensorNetwork that can be used for the TTNOpt
print("formatting mps_tensors...")
new_mps_tensors, S = format_mps_tensors(mps_tensors, n)
norm = np.sqrt(oe.contract("ab,ab", S, S.conj()))

save_edges_file(D)
for i in range(len(new_mps_tensors)):
    np.save(f"normal_distribution_data/isometry{i}.npy", new_mps_tensors[i])
    np.save(f"normal_distribution_data/singular_values.npy", S)
    np.save(f"normal_distribution_data/norm.npy", norm)