"""Test tensor inference and associated tools within 5 decimal places.

It has been shown that the covariance matrix of conditionally independent 
binary [1] and rank [2] base classifier predictions has a special structure.
This is that the off-diagonal elements i,j are proportional to the 
product of the i^th and j^th base classifier's performance as measured by
the balanced accuracy [1] or the difference of sample class conditioned average
rank predictions [2].  Consequently, these entries are those of a rank one 
matrix formed by the outer product of a vector encoding base classifier 
performances.

In this module we infer the performance vector from the empirical covariance
matrix by implementing the iterative procedue of Ahsen et al. [2].

References:
    1. Fabio Parisi, Francesco Strino, Boaz Nadler, and Yuval Kluger.
    Ranking and combining multiple predictors without labeled data.
    Proceedings of the National Academy of Sciences,
    111(4):1253--1258, 2014.

    2. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
    Unsupervised evaluation and weighted aggregation of ranked predictions.
    arXiv preprint arXiv:1802.04684, 2018.

Available classes:
- Matrix
"""
import numpy as np

def infer_matrix(Q, tol, max_iter, return_iters = False):
    """Algorithm for inferring performance vector.

    Algorithm for inferring the diagonal entries which would make
    the covariance matrix Q, of full rank, a rank one matrix.

    Args:
        Q: The covariance matrix, ((M, M) ndarray)
        tol: The tolerance for convergence, convergence occurs
            when succesive eigenvalues are smaller than tol (float).
        max_iter: the maximum number of iterations of algorithm (integer)
        return_items: should the values of each iteration be returned 
            (True or False, default False)

    Returns:
        Case 1, return_items = False:
            rank one matrix eigenvalue (float), 
            eigenvector (ndarray)
            convergence message (str)
        Case 2, return_items = True:
            rank one matrix eigenvalue (float),
            eigenvector (ndarray),
            the inferred eigenvalue at each iteration, 
            and a convergence message
            
    Raises:
        RuntimeError: raised when convergence criteria not met.
    """
    Q_off_diagonal = Q - np.diag(np.diag(Q))
    R = Q.copy()
    j = 0
    
    epsilon = np.sum(2*np.diag(Q))
    eig_values = [epsilon]
    while epsilon > tol and j < max_iter:
        # decompose the rank one approximation
        R, eig_value, eig_vector = _update_r(Q_off_diagonal, R)
        epsilon = np.abs(eig_values[-1] - eig_value)
        eig_values += [eig_value]
        j += 1
    
    if j == max_iter:
        raise RuntimeError(("Matrix decomposition did not converge, try:\n"
                            "a) Increase the maximum number of"
                            " iterations above {:d}, or \n"
                            "b) increase the minimum"
                            " tolerance above {:.4f}.").format(max_iter, tol))

    # Assume that the majority of methods 
    # correctly rank samples according to latent class
    # consequently the majority of Eigenvector
    # elements should be positive.
    if np.sum(eig_vector < 0) > Q.shape[0]/2:
        eig_vector = -eig_vector
    
    if return_iters:
        return (eig_value, eig_vector, eig_values[1:], 
                "Matrix decomposition converged in {} steps".format(j))
    else:
        return (eig_value, eig_vector,
                "Matrix decomposition converged in {} steps".format(j))

def _update_r(Q_off_diagonal, R):
    '''Update the estimate of the rank one matrix.

    Args:
        Q_off_diagonal: Covariance matrix with diagonal 
            entries set to zero ((M, M) ndarray)
        R: estimated Rank one matrix ((M, M) ndarray)

    Returns:
        tuple with 3 entries
            0. Updated estimate of the rank one matrix R, ((M, M) ndarray)
            1. eigenvalue of R (float)
            2. eigenvector of R ((M,) ndarray)
    '''
    # spectral decomposition of a hermitian matrix
    l, v = np.linalg.eigh(R)
    
    # compute the diagonal of a rank one matrix
    rdiag = np.diag(l[-1] * v[:, -1]**2)
    
    # update the rank one matrix by replacing the diagonal 
    # entries of the covariance matrix with those from the
    # previously estimated rank one matrix
    return (Q_off_diagonal + rdiag, l[-1], v[:, -1])


class Matrix:
    def __init__(self, max_iter = 5000, tol=1e-6):
        '''Implement the iterative inference from Ahsen et al. [2].

        Args:
        max_iter: max number of iterations (int, default 5000)
        tol: stopping threshold (float, default 1e-6)

        Methods:
            fit: infer Eigenvector and Eigenvalue of rank one matrix
                corresponding to the empirical covariance matrix.
        '''
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, Q):
        """Find the diagonal entries that make Q a rank one matrix.

        Args:
            Q: The covariance matrix ((M, M) ndarray)
        """
        if Q.shape[0] < 3:
            raise ValueError("The minimum required number of base classifiers is 3.")
        if Q.ndim != 2:
            raise ValueError("Input ndarray must be a matrix (ndim == 2).")
        if Q.shape[0] != Q.shape[1]:
            raise ValueError("Covariance matrix is square, check input array.")
        self.eig_value, self.eig_vector, self.msg = infer_matrix(Q, self.tol, self.max_iter)
