import numpy as np
# tools for testing the inferred weights from the covariance matrix
import tensorly
# tensorly.set_backend('numpy')
# from tensorly import tensor as toTensor
from tensorly.decomposition import parafac
from tensorly import kruskal_to_tensor



# ==================================================
# iterative method for inferring rank one tensor
# ==================================================

def infer_tensor(T, max_iter, tol, return_iters = False):
    '''
    Algorithm for inferring entries (all entries excluding i != j != k) that when substituted into the third central moment tensor results in a tensor with only one non-zero singular value.

    Input
    -----
    T : (M, M, M) ndarray
        The third central moment tensor
    tol : float
        criterion for exiting loop.  The difference of successively inferred singular values must be smaller than the tolerance
    max_iter : integer
        the maximum number of iterations of algorithm
    return_items : True or False
        (default False), should the values of each iteration be returned

    Return
    ------
    Case 1, return_items = False:
        singular value, factors, and convergence message
    Case 2, return_items = True:
        singular value, factors, the inferred singular value at each iteration, and a convergence message
    '''
    Q_t = T.copy()
    # get idx to replace
    idx = get_idx(T.shape[0])
    # first two iterations
    svalues = []
    for j in range(2):
        # CP decomposition
        factors = parafac(Q_t, rank=1)
        # store singular values
        svalues.append(get_singular_value(factors))
        # update Q_t
        Q_t = _update_Q(Q_t, kruskal_to_tensor(factors), idx)
    # compute difference in singular_values
    epsilon = np.abs(svalues[1] - svalues[0])
    # iterate until convergence or max iter is reached, whichever comes first
    while (j < max_iter) & (epsilon > tol):
        # CP decomp
        factors = parafac(Q_t, rank=1)
        # store singular value
        svalues.append(get_singular_value(factors))
        # update Q_t
        Q_t = _update_Q(Q_t, kruskal_to_tensor(factors), idx)
        epsilon = np.abs(svalues[-1] - svalues[-2])
        j += 1
    if (j == max_iter):
        raise ValueError("Tensor decomposition did not converge, try:\n  a) Increase the maximum number of iterations above {},\n  b) input the prevalence, or\n  c) increase the minimum tolerance above {}.".format(max_iter, tol))
        raise ValueError("Tensor decomposition did not converge.\nIncrease the maximum number of iterations (max_iter), input the prevalence, turn off prevalence inference, or change minimum tolerance (tol = 1e-3).")
    # self.core = core
    singular_value_sign = 1
    for w in range(len(factors)):
        factors[w] = factors[w] / norm(factors[w].squeeze())
        if np.sum(factors[w] > 0) > factors[w].size/2:
            factors[w] = -factors[w]
            singular_value_sign = -singular_value_sign
    singular_value = singular_value_sign * svalues[-1]
    if return_iters == True:
        return [singular_value, factors, svalues,
                "Tensor decomposition converged in {} steps".format(j)]
    else:
        return [singular_value, factors, 
                "Tensor decomposition converged in {} steps".format(j)]

# ==================================================
# vector norm
# ==================================================

def norm(vector):
    """
    Compute the norm of a vector
    Input
    -----
    (M,) ndarray

    Return
    ------
    float representing the norm
    """
    return np.sqrt(np.sum(vector**2))

# ==================================================
# compute singular value from tensor factors
# ==================================================

def get_singular_value(factors):
    """
    Compute the singular value from the factors

    Input
    -----
    factors : ndarray
        list of factors produced by tensorly parafac

    Return
    ------
    sv : float
        singular value
    """
    sv = 1
    for wfactor in factors:
        sv = sv * norm(wfactor.squeeze())
    return sv

# ==================================================
# get indecies of values to replace in tensor
# ==================================================

def get_idx(M):
    """
    Get the indexes in that need to be replace by the algorithm, specifically all entries excluding those in which i \neq j \neq k.

    Input
    -----
    M : integer
        The number of elements per row, column, and depth.

    Return
    idx : python list
        Each list entry is a tuple consisting of three indexes that need to be replaced by the algorithm
    """
    idx = []
    for j in range(M):
        for k in range(M):
            for l in range(M):
                if len(set([j,k,l])) < 3:
                    idx += [(j,k,l)]
    return idx

# ==================================================
# update Q tensor
# ==================================================

def _update_Q(Qt, Q1, idx):
    """
    Using the tensor generalization of SVD, replace entries of the third central moment tensor with those of the inferred tensor that has on one non-zero singular value.

    Input
    -----
    Qt : (M, M, M) ndarray
        Updated third central moment tensor
    Q1 : (M, M, M) ndarray
        The tensor resulting from the tensor SVD decomposition
    idx : python list
        Entries are tuples of length three indicating the elements of Qt that need to be replaced by those of Q1.

    Return
    ------
    Qt : (M, M, M) ndarray
        updated Qt tensor
    """
    for widx in idx:
        Qt[widx[0], widx[1], widx[2]] = Q1[widx[0], widx[1], widx[2]]
    return Qt


# ==================================================
# tensor class
# ==================================================

class tensor:
    def __init__(self, max_iter=2500, tol=1e-3, return_iters=False):
        '''
        Implementation of the iterative approach for estimating the approximate singular value of the third central moment tensor of conditionally independent method predictions.

        Input
        -----
        - max_iter : (default 5000) max number of iterations
        - tol : (default 1e-5) stopping criterion

        '''
        self.max_iter = max_iter
        self.iters = return_iters
        self.tol = tol

    # ==================================
    # Fit
    # ==================================

    def fit(self, T):
        """
        Find the singular value and factors from the third central moment tensor

        Input
        -----
        T : (M, M, M) ndarray
            The third central moment
        """
        if self.iters:
            self.singular_value, self.factors, self.evals, self.msg = infer_tensor(T, self.max_iter, self.tol, return_iters=self.iters)
        else:
            self.singular_value, self.factors, self.msg = infer_tensor(T, self.max_iter, self.tol)
