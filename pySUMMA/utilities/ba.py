import numpy as np

class ba:
    def __init__(self, l, t):
        self.set_tpr(l, t)
        self.set_tnr(l, t)
        self.set_ba()

    def set_ba(self):
        self.ba = 0.5*(self.tpr + self.tnr)

    def set_tpr(self,l, t):
        self.tpr = np.sum(l*t) / np.sum(t)

    def set_tnr(self, l, t):
        self.tnr = np.sum((1-l)*(1-t)) / np.sum(1 - t)
