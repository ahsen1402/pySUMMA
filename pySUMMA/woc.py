
import numpy as np
from .utilities import ranks

# ==========================================
# ==========================================

class rankScores:
    def rank_scores(self, data):
        N = data.shape[1]
        return np.mean((N+1)/2 - ranks(data), 0)

# ==========================================
# ==========================================

class binaryScores:
    def binary_scores(self, data):
        return np.mean(data - 0.5, 0)

# ==========================================
# ==========================================

class woc(binaryScores, rankScores):
    def __init__(self, prevalence=None, rv='Rank'):
        self.set_random_variable(rv)
        self.method = 'WOC'
        self.prevalence = prevalence

    # ==========================================
    # ==========================================

    def set_random_variable(self, rv):
        if rv.lower() == 'rank':
            from .utilities import ranks
            self.get_scores = super().rank_scores
        elif rv.lower() == 'binary':
            self.get_scores = super().binary_scores
        else:
            raise ValueError('Error : only rank and binary variables supported')

    # ==========================================
    # ==========================================

    def get_inference(self, data):
        labels = np.zeros(data.shape[1])
        labels[self.get_scores(data) > 0] = 1.
        return labels


# ==========================================
# ==========================================
