import numpy as np

class Ba:
    def __init__(self, predictions, trueLabels):
        self.N1 = np.sum(trueLabels == 1)
        self.N = trueLabels.size
        
        self.set_tpr(predictions, trueLabels)
        self.set_tnr(predictions, trueLabels)
        self.set_ba()

    def set_ba(self):
        self.ba = 0.5*(self.tpr + self.tnr)

    def set_tpr(self, predictions, trueLabels):
        TP = np.sum(predictions[trueLabels == 1] == 1)
        self.tpr = TP / self.N1

    def set_tnr(self, predictions, trueLabels):
        TN = np.sum(predictions[trueLabels == -1] == -1)
        self.tnr = TN / (self.N - self.N1)
