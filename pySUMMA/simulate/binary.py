import numpy as np
from ..utilities import ba

# ====================================
# ====================================

class binary:
    """
    Simulate the binary predictions by an ensemble of base classifiers
    """
    def __init__(self, M, N, N1, ba_lims=[0.35, 0.8], ba=None):
        """
        """
        self.M = M
        self.N = N
        self.N1 = N1
        self.set_ba(ba_lims, ba)
        self.ba = 0.4*np.random.rand(self.M) + 0.4
        self.labels = np.hstack([np.ones(N1), np.zeros(N-N1)])

    # ====================================
    # set base classifier balanced accuracy
    # ====================================

    def set_ba(self, blims, ba):
        """
        Set the user specified BA values
        """
        if ba is not None:
            self.ba = ba
        else:
            self.ba = (blims[1] - blims[0])*np.random.rand(self.M) + blims[0]

    # ====================================
    # Run simulation
    # ====================================

    def sim(self):
        """
        Generate simulation data
        """
        self.data = np.zeros(shape=(self.ba.size, self.N))
        self.empirical_ba = np.zeros(self.ba.size)
        count = 0
        for wba in self.ba:
            p = np.random.rand(self.N)
            tmp = [np.zeros(self.N1), np.zeros(self.N-self.N1)]
            tmp[0][p[:self.N1] <= self._compute_tpr(wba)] = 1.
            tmp[1][p[self.N1:] > self._compute_tnr(wba)] = 1.
            self.data[count, :] = np.hstack(tmp)
            self.empirical_ba[count] = ba(self.data[count, :], self.labels).ba
            count += 1



    # ====================================
    # compute TPR
    # ====================================

    def _compute_tpr(self, ba):
        """
        Estimate the true positive rate from the class imbalance and balanced accuracy.

        Input
        -----
        ba : (M,) ndarray
            array of base classifier balanced accuracies

        Return
        ------
        (M,) ndarray
            array of tpr
        """
        rho = self.N1/self.N
        return 2*ba*(1 - rho) + 2*rho -1

    # ====================================
    # ====================================

    def _compute_tnr(self, ba):
        """
        Estimate the true negative rate from the class imbalance and balanced accuracy.

        Input
        -----
        ba : (M,) ndarray
            array of base classifier balanced accuracies

        Return
        ------
        (M,) ndarray
            array of tpr
        """
        rho = self.N1 / self.N
        return 2*rho*(ba-1) + 1

    # ====================================
    # Empirical BA
    # ====================================

    def get_ba(self):
        return self.empirical_ba

    # ====================================
    #
    # ====================================
