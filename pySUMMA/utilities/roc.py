import numpy as np

class Roc:
    def __init__(self, s, t):
        """
        Input:
        ------
        s : ndarray
        (N sample) sample scores
        
        t: ndarray
        (N sample) true sample labels [0, 1]
        """
        self.N = len(s)
        self.tpr = np.zeros(self.N)
        self.fpr = np.zeros(self.N)
        self.Npositive = np.sum(t)
        self.Nnegative = self.N - self.Npositive
        self.deltaTPR = 1. / self.Npositive
        self.deltaFPR = 1. / self.Nnegative
        self._sort_data(s, t)
        self._curve()
        
        
    def _sort_data(self, s, t):
        idx = np.argsort(s)
        self.s = s[idx]
        self.t = t[idx]

    def _curve(self):
        i = self.N-1
        
        if self.t[i] == 1:
            self.tpr[0] = self.deltaTPR
        else:
            self.fpr[0] = self.deltaFPR

        self.auc = 0

        j = 1
        i -= 1
        while i >= 0:
            if self.t[i] == 1:
                self.tpr[j] = self.tpr[j-1] + self.deltaTPR
                self.fpr[j] = self.fpr[j-1]
            else:
                self.tpr[j] = self.tpr[j-1]
                self.fpr[j] = self.fpr[j-1] + self.deltaFPR
                self.auc += self.tpr[j] * self.deltaFPR
            j += 1
            i -= 1
        return None

    def to_dict(self):
        return {'fpr':self.fpr.tolist(),
                'tpr':self.tpr.tolist(),
                'auc':self.auc}
