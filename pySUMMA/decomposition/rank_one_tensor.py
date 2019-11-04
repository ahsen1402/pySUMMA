"""Find the singular value of the rank one third central moment tensor.

The third central moment of 3rd order conditionally independent base classifier
predictions has a special structure [1,2].  This is, the elements T_{ijk} for
i \neq j \neq k, are those of a rank one tensor,

T_{ijk} = l  v[i] v[j] v[k]

where l is the singular value and v is a unit norm vector.  In the case of SML and
SUMMA the unit norm vector consists fo values proportional to the performance
of each base classifier.  The peformance metric differs between problem statements.
In SML the base classifier predictions are binary (-1, 1), and the elements are proportional
to the balanced accuracy [1,2].  In SUMMA the base classifier predictions are sample rank
and the vector elements are proportional to Delta, the difference between the class
conditioned average rank values [3]. 

In this implementation we use the method from Jaffe et al. [1] to estimate the 
singular value (l) by linear regression.

References:
    1. Ariel Jaffe, Boaz Nadler, and Yuval Kluger.
    Estimating the accuracies of multiple classifiers without labeled data.
    Artificial Intelligence and Statistics, pages 407--415, 2015.
    
    2. Fabio Parisi, Francesco Strino, Boaz Nadler, and Yuval Kluger.
    Ranking and combining multiple predictors without labeled data.
    Proceedings of the National Academy of Sciences,
    111(4):1253--1258, 2014.

    3. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
    Unsupervised evaluation and weighted aggregation of ranked predictions.
    arXiv preprint arXiv:1802.04684, 2018.

Available classes:
- Tensor
"""
from warnings import warn
import numpy as np
from scipy.special import comb

def check_inputs(tensor_shape, vector_shape):
    """Warnings and errors for data with 3 or less base classifiers."""
    if len(vector_shape) != 1:
        raise ValueError("Input vector should be an M length ndarray")
    M = vector_shape[0]
    if tensor_shape != (M, M, M):
        raise ValueError("Input tensor should be an MxMxM and"
                         "input vector M length ndarray")
    if M < 3:
        raise ValueError("The minimum required number of base classifiers is 3.")
    elif M == 3:
        warn("3 Base Classifiers may result in unreliable\n"
             "estimation of the positive class sample\n"
             "prevalence and the magnitude of each base\n"
             "classifiers performance.  The inferred weights used for\n"
             "computing the sample scores is unaffected.")

def get_tensor_idx(M):
    """Get indecies i \neq j \neq k of the 3rd order tensor (M, M, M) tensor T.

    Args:
        M: The number of entries of the tensor of interest (integer)

    Returns:
        idx : list of tuples containing indexes (i, j, k) such that i \neq j \neq k (list)
    """
    idx = list(range(comb(M, 3, exact=True)))
    
    l = 0
    for i in range(M-2):
        for j in range(i+1, M-1):
            for k in range(j+1, M):
                idx[l] = (i, j, k)
                l += 1
    return idx


class Tensor:
    """ Fit singular value of third central moment tensor.
    
    Fit the singular value (l) for the rank one tensor whose elements
    T_{i, j, k} = l * v[i] * v[j] * v[k] where i \neq j \neq k and v is 
    the Eigenvector from the covariane decomposition.
    
    Args:
        T: third central moment tensor ((M, M, M) ndarray)
        v: Eigenvector of base classifier performances,
            from the matrix decomposition class ((M,) ndarray)

    Methods:
        tensor_and_eigenvector_elements: extract tensor values into array
        fit_singular_value: infer tensor singular by linear regression.

    Raises:
        ValueError: when the number of base classifiers 
            is less than 3, tensor is not MxMxM, that 
            T.shape = (M,M,M) and v.shape = (M,)
        UserWarning: when the number of base classifiers
            is equal to 3.
    """
    def __init__(self, T, v):
        check_inputs(T.shape, v.shape)
        self.tensorIndex = get_tensor_idx(T.shape[0])
        self.eigenvectorData, self.tensorData = self.tensor_and_eigenvector_elements(T, v)
        self.singular_value = self.fit_singular_value()
    
    def tensor_and_eigenvector_elements(self, T, v):
        """Extract tensor elements T_{ijk} for i \neq j \neq k.
        
        Args:
            T: third central moment tensor ((M, M , M) ndarray).
            v: Vector of unit norm. ((M,) ndarray).
        
        Returns:
            two element list [eigData, tData] as defined below (list)
                eigData : the product of vector entries v[i] * v[j] * v[k]
                    ((len(self.tensorIndex),) ndarray).
                tData : tensor elements i \neq j \neq k ((len(self.tensorIndex),) ndarray).
        """
        tData = np.zeros(len(self.tensorIndex))
        eigData = np.zeros(len(self.tensorIndex))

        # store tensor elements and the product of Eigenvector elements,
        # from each set of indices
        j = 0
        for widx in self.tensorIndex:
            tData[j] = T[widx[0], widx[1], widx[2]]
            eigData[j] = v[widx[0]] * v[widx[1]] * v[widx[2]]
            j += 1
        
        return (eigData, tData)

    def fit_singular_value(self):
        """Fit singular value by linear regression."""
        if len(self.eigenvectorData) == 1:
            return self.tensorData[0] / self.eigenvectorData[0]
        else:
            c = np.cov(self.eigenvectorData, self.tensorData)
            return c[0, 1] / c[0, 0]
