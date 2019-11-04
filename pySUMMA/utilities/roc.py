"""Implementation of Receiver Operator Characteristic."""
import numpy as np
from warnings import warn

def _check(scores, true_labels):
    """Raise exceptions or warnings for wrong or questionable inputs."""
    if scores.ndim != 1 or true_labels.ndim !=1:
        raise ValueError("Scores and labels must be one dimensional arrays")
    
    if scores.size != true_labels.size:
        raise ValueError("Scores and labels must have same number of entries")
    
    # test that labels are exclusively [0, 1]
    test_value = np.setdiff1d(np.array([0, 1]), true_labels).size
    test_value += np.setdiff1d(true_labels, np.array([0, 1])).size
    if test_value > 0:
        raise ValueError("True sample class labels\n"
                         "must be either 0 or 1, exclusively.")

    if np.unique(scores).size != scores.size:
        warn("Duplicate scores detected, may cause arbitrary sample ranking.")


class Roc:
    """Receiver Operating Characteristic.
    Args:
        s: Sample scores, relatively high score indicative of postive class samples
            ((N sample,) ndarray) 
        t: true sample labels [0, 1] ((N sample,) ndarray)
    Important Attributes:
        self.tpr: true positive rates (ndarray)
        self.fpr: false positive rates (ndarray)
        self.auc: Area Under Curve (float)
    Public Methods:
        to_dict: returns dictionary of Important attributes.
    Raises:
        ValueError: Number of input data entries do not match, or 
            are not 1d array, or true class labels not exclusively [0,1].
        UserWarning: if duplicate sample scores are detected
    """
    def __init__(self, s, t):
        _check(s,t)
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
