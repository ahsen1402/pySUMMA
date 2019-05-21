import numpy as np
from .decomposition import matrix
from .decomposition import tensor

from .utilities import ranks
from .utilities import third


class summa:
    """
    Apply SUMMA ensemble to data
    """
    def __init__(self, prevalence=None,
            tensor=True,
            tol=1e-3, max_iter=500):
        """
        Set parameters for SUMMA algorithm

        Input
        ------
        prevalence : float
            (default None), The prevalence of the the positive class,
        tensor : True or False
            (default True) should the third central moment tensor be decomposed
        max_iter : int
            (default 500) The maximum number of iterations for matrix & tensor decomposition.
        tol : float
            (default 1e-3), The tolerance for matrix & tensor decomposition
        """
        # test data for rank values, if data are not rank values, compute sample ranks.
        self.method = 'SUMMA'
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
        If the prevalence is known, by user input or tensor decomposition, return the AUC of the base classifiers, otherwise the weights are arbitrary.
        """
        if (not tensor) & (self.prevalence is None):
            self.metric = 'Weights'
        else:
            self.metric = 'AUC'

    # ==========================================
    # Fit Classifier
    # ==========================================

    def fit(self, data):
        """
        Infer SUMMA weights from unlabeled data

        Input
        ------
        data : (M, N) ndarray
            array of N sample scores by M base classifiers
        """
        # ensure rank data, if not compute ranks
        R = ranks(data)

        self.M = data.shape[0]
        self.N = data.shape[1]

        # Covariance decomposition
        self.cov = matrix(tol=self.tol, max_iter=self.max_iter)
        self.cov.fit(np.cov(R))

        # tensor decomposition
        self.tensor = None
        if (self.metric == 'AUC') & (self.prevalence is None):
            self.tensor = tensor(tol=self.tol, max_iter=self.max_iter)
            self.tensor.fit(third(R))


    # ==========================================
    # Compute mean of rank list
    # ==========================================

    def get_mean(self, N=None):
        """
        Compute the mean of a rank list of size N.

        Input
        N : int
            (default None) The number of samples, if None then use the number of samples in the training set, self.N.

        Return
        ------
        float
            The mean ranks, that is the mean of sequence of numbers 1,2, ..., N
        """
        if N is None:
            N = self.N
        return (N + 1) / 2

    # ==========================================
    # Infer and return the AUC
    # ==========================================

    def get_auc(self):
        '''
        Compute AUC from Delta

        Return
        ------
        (M,) ndarray
            The inferred AUC values for each base classifier
        '''
        return self.get_delta() / self.N + 0.5

    # ==========================================
    # Infer and return Delta
    # ==========================================

    def get_delta(self):
        '''
        Compute inferred Delta vector

        Return
        ------
        (M,) ndarray
            The inferred delta value for each base classifier
        '''
        return self.cov.eig_vector * self.get_delta_norm()

    # ==========================================
    # Infer and return the norm of Delta
    # ==========================================

    def get_delta_norm(self):
        '''
        Norm of Delta vector

        Return
        ------
        float
            The norm of the inferred Delta vector
            Case 1 : tensor decomposition
                Return the inferred norm of delta using tensor decomposition
            Case 2 : known prevalence
                Return the norm of delta using the a priori specified positive class prevalence
            Case 3 :
                Throw error because ||Delta|| cannot be computed without the prevalence or tensor decomposition
        '''
        if self.tensor is not None:
            beta = (self.tensor.singular_value / self.cov.eig_value)**2
            return np.sqrt(beta + 4*self.cov.eig_value)
        elif self.prevalence is not None:
            return np.sqrt(self.cov.eig_value / (self.prevalence*(1-self.prevalence)))
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
        float
            The prevalence of the positive class

        """
        if self.tensor is not None:
            norm = self.get_delta_norm()
            beta = self.tensor.singular_value / self.cov.eig_value
            return 0.5 - 0.5*beta / norm
        elif self.prevalence is not None:
            return self.prevalence
        else:
            raise ValueError("Error : the positive class prevalence was not inferred or provided upon initializing SUMMA class.")

    # ==========================================
    # SUMMA weights
    # ==========================================

    def get_weights(self):
        '''
        Return the SUMMA weights

        Return
        ------
            (M,) ndarray
                The SUMMA weight for each of the M base classifiers
        '''
        return self.cov.eig_vector

    # ==========================================
    # compute sample SUMMA scores
    # ==========================================

    def get_scores(self, data):
        """
        Compute each samples SUMMA score

        Input
        -----
        data : (M, N) ndarray
            N sample scores by M base classfiers, assumptions:
            Case 1:
                if samples scores are rank, low rank signifies postive class samples
            Case 2:
                if arbitrary score, high values signify postive class samples

        Return
        ------
        s : (N,) ndarray
            SUMMA score for each of the N samples
        """
        M = data.shape[0]
        N = data.shape[1]
        # check and convert to rank
        R = ranks(data)
        # compute scores
        s = 0
        for j in range(M):
            s += self.cov.eig_vector[j] * (self.get_mean(N=N) - R[j, :])
        return s

    # ==========================================
    # infer class labels
    # ==========================================

    def get_inference(self, data):
        """
        Estimate class labels from SUMMA scores.  SUMMA scores greater than zero are assigned a positive class label (1), otherwise a negative class label (0).

        Input
        -----
        data : (M, N) ndarray
            N sample scores by M base classfiers, assumptions:
            Case 1:
                if samples scores are rank, low rank signifies postive class samples
            Case 2:
                if arbitrary score, high values signify postive class samples

        Return
        ------
        labels : (N,) ndarray
            Inferred class labels, 1 designating positive class and 0 negative class.
        """
        labels = np.zeros(data.shape[1])
        labels[self.get_scores(data) > 0] = 1.
        return labels



    # ==========================================
    # ==========================================
