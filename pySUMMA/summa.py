"""Implement the Strategy for Unsupervised Multiple Method Aggregation (SUMMA) classifier.

SUMMA infers base classifier performance from the empirical covariance matrix
and third central moment tensor of base classifier predictions.  Predictions of
base classifiers are assumed to be sample ranks, with low ranks representing positive
class samples (1), and high ranks negative class samples (0).  Moreover, SUMMA
assumes that the base classifier predictions are conditionally independent.  

THE RESULTS OF SUMMA DEPEND ON THE VALIDITY OF THE CONDITIONAL INDEPENDENCE
ASSUMPTION.  IF YOUR DATA DOES NOT SATISFY THE CONDITIONAL INDEPENDENCE 
ASSUMPTION, DO NOT USE SUMMA.

In this implementatino of SUMMA [1], we estimate the rank one Eigenvalue and Eigenvector
from the empirical covariance by the iteration strategy developed by Ahsen et al. [1].
The inferred elements are then used as the aggregation weights for computing the SUMMA
sample scores and inferring sample class labels.

We estimate the tensor singular value using a method introduced by Jaffe et al [2].
Using tensor singular value and the Eigenvalue from the covariance matrix
decomposition we estimate the AUC of each base classifier, and the prevalence 
of postive class samples (class 1), without using labeled data.

References:
    1. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
    Unsupervised evaluation and weighted aggregation of ranked predictions.
    arXiv preprint arXiv:1802.04684, 2018.
    
    2. Ariel Jaffe, Boaz Nadler, and Yuval Kluger. 
    Estimating the accuracies of multiple classifiers without labeled data.
    Artificial Intelligence and Statistics, pages 407--415, 2015.
    
Available classes:
- summa: Implementatin of the SUMMA classifier
"""

import numpy as np
from .decomposition import Matrix
from .decomposition import Tensor
from .utilities.check_data import check_rank_data
from .utilities import third

class Summa:
    """Apply SUMMA ensemble to data.

    Args:
        prevalence: the fraction of samples from the positive class.
            When None, infer prevalence from data. (float [0,1] or None, default None)

    Attributes:
    - method: name of classifier (str).
    - prevalence: fraction of samples from the positive class (float between 0 and 1).
    - metric: name of performance metric associated with inference (str).
    - N: the number of sample predictions by each base classifier used for fitting (int).
    - cov: covariance decomposition class instance.
    - tensor: tensor decomposition class instance.

    Public Mehtods:
    - fit: Fit the SUMMA model to the empirical covariance and third central moment.
    - get_auc: Compute and return the inferred AUC of each base classifier (ndarray).
    - get_prevalence: Compute and return inferred or return given sample class 
        prevalence (float).
    - get_weights: Get the weights used for SUMMA sample scores (ndarray).
    - get_scores: Compute and return the SUMMA sample scores (ndarray).
    - get_inference:  Compute and return SUMMA inferred sample class labels (ndarray).
    """
    
    def __init__(self, prevalence=None):
        self.method = 'SUMMA'
        self.prevalence = prevalence
        self.metric = "AUC"

    def fit(self, data, tol=1e-3, max_iter=500):
        """Infer SUMMA weights from unlabeled data.

        Args:
            data: matrix of N sample rank predictions by M 
                base classifiers ((M, N) ndarray)
            tol: the tolerance for convergence in matrix 
                decomposition (float, default 1e-3).
            max_iter: the maximum number of iterations for matrix decomposition
                (int, default 500).

        Raises:
            ValueError: when M methods < 5 or N samples < 5.
            ValueError: when data are not ranks.
        """
        check_rank_data(data)

        self.N = data.shape[1]
        
        # Covariance decomposition
        self.cov = Matrix(tol=tol, max_iter=max_iter)
        self.cov.fit(np.cov(data))

        # tensor decomposition
        if self.prevalence is None:
            self.tensor = Tensor(third(data), self.cov.eig_vector)
    
    def get_mean(self, N=None):
        """Compute the mean of a rank list of size N.

        Args:
            N: The number of samples, if None then use the number of samples
                from training set, self.N (int or None, default None)

        Returns:
            The mean ranks, that is the mean of sequence of 
                numbers 1,2, ..., N (float).
        """
        if N is None:
            N = self.N
        return 0.5 * (N + 1)

    def get_auc(self):
        '''Compute AUC from Delta.
        
        Returns:
            The inferred AUC values for each base classifier ((M,) ndarray).
        '''
        return self.get_delta() / self.N + 0.5

    def get_delta(self):
        '''Compute inferred performance vector, Delta.
        
        Returns:
            The inferred Delta value for each of the M 
               base classifiers ((M,) ndarray)
        '''
        return self.cov.eig_vector * self.get_delta_norm()

    def get_delta_norm(self):
        '''Norm of performance vector Delta.

        Case 1: known prevalence
            Return the norm of delta using the a priori 
            specified positive class prevalence.
        Case 2: inferred prevalence
            Return the inferred norm by using
            the tensor and covariance singular values.
        
        Returns: 
            The norm of the performance vector, Delta (float).
        '''
        if self.prevalence is not None:
            return np.sqrt(self.cov.eig_value / (self.prevalence*(1-self.prevalence)))
        else:
            beta = (self.tensor.singular_value / self.cov.eig_value)**2
            return np.sqrt(beta + 4*self.cov.eig_value)

    def get_prevalence(self):
        """Either infer or return the postive class prevalence.

        Case 1: Return the a priori specified prevalence
        Case 2: Return the inferred prevalence
        
        Returns:
           The sample class prevalence (float)
        """
        if self.prevalence is not None:
            return self.prevalence
        else:
            norm = self.get_delta_norm()
            beta = self.tensor.singular_value / self.cov.eig_value
            return 0.5 + 0.5*beta / norm

    def get_weights(self):
        '''Return the SUMMA weights.

        Returns:
            The SUMMA weight for each of the M base classifiers ((M,) ndarray)
        '''
        return self.cov.eig_vector

    def get_scores(self, data):
        """Compute each SUMMA score for each sample.

        Here we assume the convention the postive class samples have low rank,
        while negative class samples have high rank.
        
        Args:
            data : N sample rank predictions by M base classfiers ((M, N) ndarray).

        Returns:
           s: SUMMA score for each of the N samples ((N,) ndarray).

        Raises:
            ValueError: if data are not rank predictions
            TypeError: if data are not in ndarray
        """
        check_rank_data(data)

        M = data.shape[0]
        N = data.shape[1]

        # compute scores
        s = 0
        for j in range(M):
            s += self.cov.eig_vector[j] * (self.get_mean(N=N) - data[j, :])
        return s

    def get_inference(self, data):
        """Estimate class labels from SUMMA scores.

        SUMMA scores greater than or equal to zero are assigned a
        positive class label (1), otherwise a negative class label (0).
        Here we assume the convention the postive class samples have low 
        rank, while negative class samples have high rank.

        Args:
            data: N sample rank predictions by M base classfiers ((M, N) ndarray).

        Returns:
            labels: Inferred sample class labels ((N,) ndarray)

        Raises:
            ValueError: if data are not rank predictions
            TypeError: if data are not in ndarray
        """
        labels = self.get_scores(data)
        labels[labels >= 0] = 1.
        labels[labels < 0] = 0
        return labels
