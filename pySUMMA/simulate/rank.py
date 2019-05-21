import numpy as np
from scipy.special import ndtri

from ..utilities import ranks
from ._gaussian_scores import gaussians


class rank:
    """
    Simulate the rank predictions by an ensemble of methods
    """
    def __init__(self, M, N, N1, auc_lims=[0.45, 0.8], auc=None):
        """
        Set simulation parameters

        Input
        -----
        M : int
            The number of base classifiers
        N : int
            The number of samples
        N1 : int
            The number of samples in the positive class
        auc_lims : python list
            (default [0.3, 0.8]])
            [float : min auc,
             float : max auc]
        auc : (M,) ndarray
            (optional) An array of auc values for each M method, auc takes precedence over auc_lims
        """
        self.M = M
        self.N = N
        self.N1 = N1
        # set default auc
        self.set_auc(auc_lims, auc)
        # compute covariance matrix
        self.cov = np.eye(M)

    # ====================================
    # set base classifier auc
    # ====================================

    def set_auc(self, alims, auc):
        """
        Set the user specified AUC values of base classifiers
        """
        if auc is not None:
            self.auc = auc
        else:
            self.auc = (alims[1]-alims[0]) * np.random.rand(self.M) + alims[0]

    # ====================================
    # ====================================

    def sim(self):
        """
        Generate simulation data
        """
        self.labels, gscores = gaussians(self.M, self.N,
                        self.N1,
                        self._delta_mu_from_auc())
        self.data = ranks(gscores)

    # ====================================
    # get the dmu that matches desired AUC
    # ====================================

    def _delta_mu_from_auc(self):
        """
        Compute delta_mu for Gaussian sampling

        Return
        ------
        (M,) ndarray
            delta mu to be using for sampling scores from two Gaussian distributions
        """
        return np.sqrt(2.)*ndtri(self.auc)

    # ====================================
    # compute delta of simulation data
    # ====================================

    def get_delta(self):
        """
        Compute the empirical Delta for each base classifier

        Return
        ------
        (M,) ndarray
            Delta values for each base classifier
        """
        if self.data is None:
            self.sim()
        return np.mean(self.data[:, self.labels == 0], 1) - np.mean(self.data[:, self.labels==1], 1)

    # ====================================
    # compute effective AUC of simulation data
    # ====================================

    def get_auc(self):
        """
        Compute the AUC of base classifiers from their empirical Delta

        Return
        ------
        (M,) ndarray
            AUC of each base classifier
        """
        return self.get_delta() / self.N + 0.5

    # ====================================
    #
    # ====================================
