import numpy as np
from .decomposition import matrix
from .decomposition import tensor

from .utilities import third


class sml:
    """
    Apply SML ensemble to data

    """
    def __init__(self, prevalence=None,
            tensor=True,
            tol=1e-3, max_iter=500):
        """
        Set parameters for our implementation of the SML algorithm

        Input
        -----
        prevalence : float
            (default None) representing the prevalence of the the positive class,
        tensor : True or False
            (default True) Specify whether tensor should be decomposed
        max_iter : int
            (default 500) The maximum number of iterations for matrix & tensor decomposition.
        tol : float
            (default 1e-3), The tolerance for matrix & tensor decomposition
        """
        self.method='SML'
        if prevalence is not None:
            tensor=False
        self.prevalence = prevalence
        self.tol = tol
        self.max_iter = max_iter
        self._set_metric(tensor)

    # ==========================================
    # SET METRIC
    # ==========================================

    def _set_metric(self, tensor):
        """
        If the prevalence is known, by user input or tensor decomposition, return the BA of the base classifiers, otherwise the weights are arbitrary.
        """
        if (self.prevalence is None) & (not tensor):
            self.metric = 'Weights'
        else:
            self.metric = 'BA'

    # ==========================================
    # Verify that input data is binary, {0, 1}
    # ==========================================

    def _check_data(self, data):
        """
        Verify that the input data is binary, {0, 1}.  If data is not binary throw ValueError

        Input
        -----
        data : (M, N) ndarray

        Return
        ------
        None
        """
        if np.setdiff1d(data, [0, 1]).size != 0:
            raise ValueError("Error: Input data must be solely comprised of values {0, 1}")
        return None

    # ==========================================
    # Fit classifier
    # ==========================================

    def fit(self, data):
        """
        Infer SML weights from unlabeled data

        Input
        ------
        data : (M, N) ndarray
            N sample predictions by M base classifiers
        """
        # ensure binary data
        self._check_data(data)

        # save training dimensions
        self.M = data.shape[0]
        self.N = data.shape[1]

        # Covariance decomposition
        self.cov = matrix(tol=self.tol,
                            max_iter=self.max_iter)
        self.cov.fit(np.cov(data))

        # tensor decomposition
        self.tensor = None
        # if we should fit tensor than fit it
        if (self.metric == 'BA') & (self.prevalence is None):
            self.tensor = tensor(tol=self.tol, max_iter=self.max_iter)
            self.tensor.fit(third(data))

    # ==========================================
    # Infer balanced accuracy
    # ==========================================

    def get_ba(self):
        """
        Compute BA from eigenvector

        Return
        ------
        (M,) ndarray
            inferred balanced accuracies
        """
        return 0.5*(1+self.cov.eig_vector * self.get_ba_norm())

    # ==========================================
    # Infer and return BA Norm
    # ==========================================

    def get_ba_norm(self):
        '''
        Norm of BA vector

        Return
        ------
        float
            Case 1 : tensor decomposition
                Return the inferred norm of BA using tensor decomposition
            Case 2 : known prevalence
                Return the norm of BA using the a priori specified positive class prevalence
            Case 3 :
                Throw error because norm of BA cannot be computed without the prevalence or tensor decomposition
        '''
        if self.tensor is not None:
            beta = (self.tensor.singular_value / self.cov.eig_value)**2
            return np.sqrt(beta + 4*self.cov.eig_value)
        elif self.prevalence is not None:
            return np.sqrt(self.cov.eig_value / (self.prevalence * (1-self.prevalence)))
        else:
            raise ValueError("Error : Tensor decomposition is set to False and the positive class prevalence was not provided.  To obtain the balanced accuracy remedy one of these deficienies.")

    # ==========================================
    # Infer and / or return prevalence of positive class
    # ==========================================

    def get_prevalence(self):
        """
        Either infer or return the postive class prevalence

        Return
        ------
        float [0, 1]
            The prevalence of the positive class
        """
        if self.tensor is not None:
            beta = self.tensor.singular_value / self.cov.eig_value
            return 0.5 - 0.5*beta / self.get_ba_norm()
        elif self.prevalence is not None:
            return self.prevalence
        else:
            raise ValueError("Error : the positive class prevalence was not inferred or provided upon initializing SML class.")

    # ==========================================
    # SML weights
    # ==========================================

    def get_weights(self):
        '''
        Return the SML weights

        Return
        ------
        (M,) ndarray
            SML weights
        '''
        return self.cov.eig_vector

    # ==========================================
    # compute sample SML scores
    # ==========================================

    def get_scores(self, data):
        """
        Compute each samples SML score

        Input
        -----
        data : (M, N) ndarray
            array of N binary value predictions for each M base classifier

        Return
        ------
        s : (N,) ndarray
            SUMMA scores for each of the N samples
        """
        # if data is not appropriate throw error
        self._check_data(data)

        # specify the number of methods
        M = data.shape[0]

        # compute scores
        s = 0
        for j in range(M):
            s += self.cov.eig_vector[j] * (data[j, :]-0.5)
        return s

    # ==========================================
    # ==========================================

    def get_inference(self, data):
        """
        Estimate class labels from SML scores.  SML scores greater than zero are assigned a positive class label (1), otherwise a negative class label (0).

        Input
        -----
        data : (M, N) ndarray
            array of N binary value predictions for each M base classifier

        Return
        ------
        labels : (N,) ndarray
            SML inferred binary values for each sample, 1 designating positive class and 0 negative class.
        """
        labels = np.zeros(data.shape[1])
        labels[self.get_scores(data) > 0] = 1.
        return labels


    # ==========================================
    # ==========================================
