"""Generate simulation data of the rank pedictions by base classifiers.

Given the number of samples (N), base classifiers (M), and the number
of samples in the positive class, generate an (M, N) ndarray of
sample scores, and (N,) array

Available classes:
- Rank

References:
1. Caren Marzban.  The roc curve and the area under it as performance measures.
Weather and Forecasting, 19(6):1106--1114, 2004.
2. Mehmet Eren Ahsen, Robert Vogel, and Gustavo Stolovitzky.
Unsupervised evaluation and weighted aggregation of ranked predictions.
arXiv preprint arXiv:1802.04684, 2018.
"""

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

def gaussians(M, N, N1, dmu):
    '''Simulate N sample scores by M base classifiers.

    Sample scores are simulated from two Gaussian distributions,
    one representing the positive class samples, the other negative 
    class samples.  The mean of the negative class samples are
    dmu, while the mean of the postive class samples is 0.  For
    both distributions, the standard deviation is 1.

    Args:
        M: the number of base classifiers (integer)
        N: the number of samples (integer)
        N1: the number of samples in the positive class (integer)
        dmu: the difference of the negative class and positive class
            mean scores ((M,) ndarray).

    Returns:
        (labels, r):         
        labels : True sample class labels ((N,) ndarray)
        r: N sample scores by M base classifiers ((M, N) ndarray).
    '''
    # set true sample class labels
    labels = np.zeros(N)
    labels[:N1] = 1.

    # given the true labels, simulate sample scores from 
    # respective classes.
    delta_mu = np.tile(dmu.reshape(M, 1), (1, N - N1))
    s1 = np.random.randn(M, N1)                       # sample from class 1
    s0 = delta_mu + np.random.randn(M, N - N1)        # sample from class 0
    
    return labels, np.hstack([s1, s0])

def ranks(data):
    '''Rank data matrix.

    Assume that high scores are from samples of class 0, and
    low scores are from samples of class 1.

    Args:
        data: N sample score predictions, where a high score is indicative
            of positive class samples and low score negative
            class, by M base classifiers ((M, N) ndarray)

    Returns:
        rdata: N sample rank predictions by M base classifiers ((M, N) ndarray).
    '''
    rdata = np.zeros(shape=data.shape)
    for j in range(data.shape[0]):
        rdata[j, :] = rankdata(data[j, :], method="ordinal")
    return rdata


class Rank:
    """Simulate the rank predictions by an ensemble of methods.
    
    Args:
        M: The number of base classifiers (integer)
        N: The number of samples (integer)
        N1: The number of samples in the positive class, 0 < N1 < N (integer)
        auc_lims: the range of AUC values to sample organized as [auc_min, auc_max].
            By definition 0 <= AUC <= 1, and auc_min < auc_max ([float, float]).
        auc: AUC of M base classifiers or None.  If None, sample AUC values 
            of M base classifiers using auc_lims.  ((M,) ndarray or None, default None).
    
    Attributes:
    - labels: true sample class labels ((M,) ndarray)
    - gscores: generated Gaussian predictions for N samples by 
        M classifiers ((M, N) ndarray).
    - data:

    Public Methods:
    - get_empirical_delta:
    - get_empirical_auc:
    """
    
    def __init__(self, M, N, N1, auc_lims=[0.45, 0.8], auc=None):
        self.M = M
        self.N = N
        self.N1 = N1
        self.set_auc(auc_lims, auc)
    
    def set_auc(self, alims, auc):
        """Set the user specified AUC values of base classifiers."""
        if auc is not None:
            self.auc = auc
        else:
            self.auc = (alims[1]-alims[0]) * np.random.rand(self.M) + alims[0]

    def sim(self):
        """Generate simulation data."""
        self.labels, self.gscores = gaussians(self.M, 
                                              self.N,
                                              self.N1,
                                              self._delta_mu_from_auc())
        self.data = ranks(self.gscores)

    def _delta_mu_from_auc(self):
        """Compute the delta_mu from AUC for Gaussian sampling.

        delta_mu is the difference of means between the Gaussian
        distributions representing positive and negative class sample
        scores.  Here we implement the strategy by Marzban [1] to compute
        the difference of means between two Gaussians with 
        unit variances from a given AUC.  These means that the samples generated
        are those of a base classifier whose performance is the given AUC.
        
        Returns:
            mean difference between positive and negative sample scores
            used for Gaussian sampling ((M,) ndarray)
        """
        return np.sqrt(2.)*ndtri(self.auc)

    def get_empirical_delta(self):
        """Compute the empirical Delta for each base classifier.

        Delta is an M length vector characterizing the performannce
        of each base classifier.  The Delta value of the i^th base classifier
        is:

        Delta_i = mean_{i|0} - mean_{i|1}

        where the mean_{i|0} represents the average rank of samples from class 0,
        while mena_{i|1} is the average rank of samples from class 1.

        Returns:
            Delta values for each base classifier ((M,) ndarray).
        """
        if self.data is None:
            self.sim()
        return (np.mean(self.data[:, self.labels == 0], 1) - 
                np.mean(self.data[:, self.labels == 1], 1))

    def get_empirical_auc(self):
        """Compute the AUC of base classifiers from their empirical Delta.
        
        We use the equation relating AUC and Delta for rank predictions
        of binary classifiers from Ahsen et al. [2].

        Returns:
            AUC of each base classifier ((M,) ndarray).
        """
        return self.get_empirical_delta() / self.N + 0.5
