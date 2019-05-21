import numpy as np
from scipy.integrate import trapz

class roc:
    def __init__(self, s, t):
        '''
        Input:
        s = numpy array of scores
        t = numpy array of true class labels
        '''
        self.tpr = None
        self.fpr = None
        self._compute_roc(s, t)
        self._get_auc()

    def _compute_roc(self, s, t):
        if s.size > 1000:
            scores = np.linspace(s.min(), s.max(), 1000)[::-1]
        else:
            scores = np.sort(s)[::-1]
        self.tpr = np.zeros(scores.size)
        self.fpr = np.zeros(scores.size)
        count = 0
        for ws in scores:
            self.tpr[count] = self._get_tpr(s, t, ws)
            self.fpr[count] = self._get_fpr(s, t, ws)
            count += 1

    def _get_tpr(self, s, t, thresh):
        tp = float(np.sum(t[s >= thresh]))
        return tp / np.sum(t)

    def _get_fpr(self, s, t, thresh):
        f = 1- t
        fp = float(np.sum(f[s >= thresh]))
        return fp / np.sum(f)

    def _get_auc(self):
        idx = np.argsort(self.fpr)
        self.auc = trapz(self.tpr[idx], self.fpr[idx])

    def to_dict(self):
        return {'fpr':self.fpr.tolist(),
                'tpr':self.tpr.tolist(),
                'auc':self.auc}
