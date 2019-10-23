import numpy as np

# =====================================================
# Infer rank one matrix
# =====================================================

def infer_matrix(Q, tol, max_iter, return_iters = False):
    """
    Algorithm for inferring the diagonal entries which would make the covariance matrix Q, of full rank, a rank one matrix.

    Input
    -----
    Q : (M, M) ndarray
        The covariance matrix
    tol : float
        criterion for exiting loop.  The difference of successively inferred eigenvalues must be smaller than the tolerance
    max_iter : integer
        the maximum number of iterations of algorithm
    return_items : True or False
        (default False), should the values of each iteration be returned

    Return
    ------
    Case 1, return_items = False:
        rank one matrix eigenvalue, eigenvector and convergence message
    Case 2, return_items = True:
        rank one matrix eigenvalue, eigenvector, the inferred eigenvalue at each iteration, and a convergence message
    """
    Q_off_diagonal = Q - np.diag(np.diag(Q))
    R = Q.copy()
    j = 0
    # sum of eigenvalues is the first estimate
    epsilon = np.sum(2*np.diag(Q))
    eig_values = [epsilon]
    while (epsilon > tol) & (j < max_iter):
        # decompose the rank one approximation
        R, eig_value, eig_vector = _update_r(Q_off_diagonal, R)
        # compute the difference beween the j+1 and j eigen value
        epsilon = np.abs(eig_values[-1] - eig_value)
        # store eigen_value
        eig_values += [eig_value]
        j += 1
    # if the algorithm does not converge throw an error
    if (j == max_iter):
        raise ValueError("Matrix decomposition did not converge, try:\n  a) Increase the maximum number of iterations above {}, or \n  b) increase the minimum tolerance above {}.".format(max_iter, tol))
    # assume that the majority of methods correctly rank samples according to latent class
    if np.sum(eig_vector < 0) > Q.shape[0]/2:
        eig_vector = -eig_vector
    # if true return each computed eigen value
    if return_iters == True:
        return [eig_value, eig_vector, eig_values[1:], 
                "Matrix decomposition converged in {} steps".format(j)]
    else:
        return [eig_value, eig_vector,
                "Matrix decomposition converged in {} steps".format(j)]

# =====================================================
# Update matrix, eigenvalue and eigenvector
# =====================================================

def _update_r(Q_off_diagonal, R):
    '''
    Update the estimate of the rank one matrix.

    Input
    -----
    Q_off_diagonal : (M, M) ndarray
        in which the diagonal entries are zero
    - R : (M, M) ndarray
        estimated Rank One matrix

    Return
    ------
    Python list [
        (M, M) ndarray
            Updated estimate of the rank one matrix R,
        float
            eigenvalue of R
        (M,) ndarray
            eigenvector of R
    ]
    '''
    # spectral decomposition of a hermitian matrix
    l, v = np.linalg.eigh(R)
    # compute the diaganol of a rank one matrix
    rdiag = np.diag(l[-1] * v[:, -1]**2)
    # update the rank one matrix by replacing the diagonal entries of the covariance matrix with those from the previously estimated rank one matrix
    return [Q_off_diagonal + rdiag, l[-1], v[:, -1]]


# =====================================================
# MATRIX CLASS
# =====================================================

class Matrix:
    def __init__(self, max_iter = 5000, tol=1e-5):
        '''
        Implementation of the iterative approach for estimating a rank one matrix from the covariance of conditionally independent method predictions.

        Input
        -----
        max_iter : int
            (default 5000) max number of iterations
        tol : float
            (default 1e-5) stopping criterion
        '''
        self.max_iter = max_iter
        self.tol = tol

    # ==================================
    # Fit
    # ==================================

    def fit(self, Q):
        """
        Find the diagonal entries that make Q a rank one matrix.

        Input
        -----
        Q : (M, M) ndarray
            The covariance matrix
        """
        self.eig_value, self.eig_vector, self.msg = infer_matrix(Q, self.tol, self.max_iter)
