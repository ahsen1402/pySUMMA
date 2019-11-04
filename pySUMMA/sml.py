""" Implement the Spectral Meta Learner (SML) classifier.

SML infers base classifier performance from the empirical covariance matrix
and third central moment tensor of base classifier predictions.  Predictions of
base classifiers are assumed to be sample class labels, with 1 representing positive
class samples, and -1 negative class samples.  Moreover, SML
assumes that the base classifier predictions are conditionally independent.

THE RESULTS OF SML DEPEND ON THE VALIDITY OF THE CONDITIONAL INDEPENDENCE
ASSUMPTION.  IF YOUR DATA DOES NOT SATISFY THE CONDITIONAL INDEPENDENCE 
ASSUMPTION, DO NOT USE SML.

In this implementation of SML [1], we estimate the rank one Eigenvalue and 
Eigenvector from the empirical covariance matrix by the iterative strategy
proposed by Ahsen et al. [3].  The inferred Eigenvector elements are then
used as the aggregation weights for computing the SML score and for inferring
sample class labels.

We estimate the tensor singular value using a method introduced by Jaffe et al [2].
Using tensor singular value and the Eigenvalue from the covariance matrix
decomposition we estimate the balanced accuracy of each base classifier, and
the prevalence of postive class samples (class 1), without using labeled data.

References:
    1. Fabio Parisi, Francesco Strino, Boaz Nadler, and Yuval Kluger.
    Ranking and combining multiple predictors without labeled data.
    Proceedings of the National Academy of Sciences,
    111(4):1253--1258, 2014.

    2. Ariel Jaffe, Boaz Nadler, and Yuval Kluger. 
    Estimating the accuracies of multiple classifiers without labeled data.
    Artificial Intelligence and Statistics, pages 407--415, 2015.

    3. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
    Unsupervised evaluation and weighted aggregation of ranked predictions.
    arXiv preprint arXiv:1802.04684, 2018.

Available classes:
- sml: Implementation of the SML classifier
"""

import numpy as np

from .decomposition import Matrix
from .decomposition import Tensor
from .utilities.check_data import check_binary_data
from .utilities import third


class Sml:
    """Apply SML ensemble to data.
    
    Args:
        prevalence: the fraction of samples from the  positive class.
            When None given, infer prevalence from data. 
            (float beteen 0 and 1 or None, default None)

    Attributes:
    - method: name of classifier (str).
    - prevalence: fraction of samples from the positive class (float between 0 and 1)
    - metric: name of performance metric associated with inference (str)
    - cov: covariance decomposition class instance
    - tensor: tensor decomposition class instance
    
    Public Methods:
    - fit: Fit the SML model to the empirical covariance and third central moment.
    - get_ba: Compute and return the inferred balanced accuracy of each base 
        classifier (ndarray).
    - get_prevalence: Compute and return inferred or return given sample class 
        prevalence (float).
    - get_weights: Get the weights used for the SML weighted sum (ndarray).
    - get_scores: Compute and return the SML sample scores (ndarray).
    - get_inference: Compute and return the SML inferred sample class labels (ndarray).
    """
    
    def __init__(self, prevalence=None):
        self.method='SML'
        self.prevalence = prevalence
        self.metric = "BA"

    def fit(self, data, tol=1e-3, max_iter=500):
        """Fit the SML model to the empirical covariance and third central moment.

        Args:
            data : N sample predictions by M base classifiers ((M, N) ndarray)
            tol: the tolerance for matrix decomposition (float, default 1e-3).
            max_iter: of the maximum number of iterations for matrix decomposition
                (int, default 500).
        
        Raises:
            ValueError: when M methods < 5 or N samples < 5.
            ValueError: when data are not binary values [-1, 1].
        """
        check_binary_data(data)

        # Covariance decomposition
        self.cov = Matrix(tol=tol, max_iter=max_iter)
        self.cov.fit(np.cov(data))

        # tensor decomposition
        if self.prevalence is None:
            self.tensor = Tensor(third(data), self.cov.eig_vector)

    def get_ba(self):
        """Compute the balanced accuracy of each base classifier.

        Using the fitted Eigenvector and the norm of the
        performance vector compute the inferred balanced accuracy of 
        each base classifier.

        Returns:
            (M,) ndarray of inferred balanced accuracies.
        """
        return 0.5*(1 + self.cov.eig_vector * self.get_ba_norm())

    def get_ba_norm(self):
        '''Compute the norm of the performance vector.

        Case 1: known prevalence
            Return the norm of the performance vector using the a priori
            specified positive class prevalence and Eigenvalue from the
            matrix decomposition.
        Case 2: tensor singular value
            Return the inferred norm of the performance vector using the 
            Eigenvalue and singular value estimated from the matrix and tensor 
            decomposition, respectively.

        Returns:
            float, norm of the performance vector.
        '''
        if self.prevalence is not None:
            return np.sqrt(self.cov.eig_value / (4 * self.prevalence * (1-self.prevalence)))
        else:
            beta = (self.tensor.singular_value / self.cov.eig_value)**2
            return 0.5 * np.sqrt(beta + 4*self.cov.eig_value)

    def get_prevalence(self):
        """Return sample class prevalence.

        Case 1: known prevalence
            Return the a priori known positive class sample prevalence.
        Case 2: tensor singular value
            Return the inferred positive class sample prevalence.

        Returns:
            float, prevalence of the positive class on the interval [0, 1]
        """
        if self.prevalence is not None:
            return self.prevalence
        else:
            beta = self.tensor.singular_value / self.cov.eig_value
            return 0.5 * (1  - 0.5*beta / self.get_ba_norm())

    def get_weights(self):
        '''Return the SML weights.

        Returns:
            (M,) ndarray SML weights
        '''
        return self.cov.eig_vector

    def get_scores(self, data):
        """ Compute SML score for each sample.

        The SML score is the value of approximate likelihood used
        to infer sample class labels in Parisi et al. [1].  Simply, the
        k^th samples score is the weighted sum of all base classifier
        predictions.

        Args:
            data: (M, N) ndarray of N binary value predictions for 
                each M base classifier

        Returns:
            s: (N,) ndarray of SML scores for each sample

        Raises:
            ValueError: if data are not binary predictions
            TypeError: if data are not in ndarray
        """
        check_binary_data(data)
        
        M = data.shape[0]

        s = 0
        for j in range(M):
            s += self.cov.eig_vector[j] * data[j, :]
        return s

    def get_inference(self, data):
        """Compute and return the SML inferred sample class labels.

        The approximate maximum likelihood esimtate of sample class labels
        from Parisi et al. [1].  We assign samples with SML scores greater 
        than or equal to zero with a positive class label (1), otherwise a
        negative class label (-1).

        Args:
            data : (M, N) ndarray of N binary value predictions of M base classifiers

        Returns:
            labels : (N,) ndarray of SML inferred binary values for each sample,
                1 designating positive class and -1 negative class.

        Raises:
            ValueError: if data are not binary predictions
            TypeError: if data are not in ndarray
        """
        labels = self.get_scores(data)
        labels[labels >= 0] = 1.
        labels[labels < 0] = -1.
        return labels
