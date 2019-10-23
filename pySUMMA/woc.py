"""Implement the Wisdom of the Crowd classifier.

The Wisdom of the Crowd classifier generates an ensemble
score by averaging the predictions of all base classifiers.

Available classes:
- RankWOC : WOC classifier for rank predictions
- BinaryWOC : WOC classifier for binary predictions

References:
1.Daniel Marbach, James C Costello, Robert Kuffner, Nicole~M Vega, Robert J
Prill, Diogo M Camacho, Kyle R Allison, Andrej Aderhold, Richard Bonneau,
Yukun Chen, et al. Wisdom of crowds for robust gene network inference.
Nature methods, 9(8):796, 2012.
"""

import numpy as np

from .utilities.check_data import check_rank_data
from .utilities.check_data import check_binary_data


class RankWoc:
    """WOC for ranked based classifier predictions.

    Args:
        prevalence: the fraction of samples from the  positive class.
            When None given, infer prevalence from data. 
            (float beteen 0 and 1 or None, default None)

    Attributes:
    - method: name of classifier (str).
    - prevalence: fraction of samples from the positive class (float between 0 and 1)
    
    Public Methods:
    - get_scores: Get the WOC score for each sample
    - get_inference: Get inferred class label for each sample
    """
    
    def __init__(self, prevalence=None):
        self.method = 'rankWOC'
        self.prevalence = prevalence
        
    def get_scores(self, data):
        """Compute the score for each sample.
        
        Args:
            data: (M method, N sample) ndarray of rank predictions

        Returns:
            (N,) ndarray of sample scores
        """
        check_rank_data(data)
        
        mean_rank = (data.shape[1] + 1.) / 2     
        return np.mean(mean_rank - data, 0)      # mean sample prediction over base classifiers

    def get_inference(self, data):
        """Get the inferred class label for each sample.

        Args:
            data: (M method, N sample) ndarray of rank predictions
        
        Returns:
            labels: (N sample,) ndarray of inferred class labels.
        """
        labels = self.get_scores(data)
        labels[labels >= 0] = 1.
        labels[labels < 0] = 0
        return labels


class BinaryWoc:
    """WOC for binary (-1, 1) classifier predictions.

    Keyword args:
        prevalence: the fraction of samples from the  positive class.
            When None given, infer prevalence from data. 
            (float beteen 0 and 1 or None, default None)
    
    Data Attributes
    - method: str of the name of classifier.
    - prevalence: fraction of samples from the positive class

    Public Methods:
    - get_scores: Get the WOC score for each sample
    - get_inference: Get inferred class label for each sample

    Args:
        prevalence: 
    """

    def __init__(self, prevalence=None):
        self.method = 'binaryWOC'
        self.prevalence = prevalence
        
    def get_scores(self, data):
        """Compute the score for each sample.
        
        Args:
            data: (M method, N sample) ndarray of binary predictions

        Returns:
            (N sample,) ndarray of scores
        """
        check_binary_data(data)
        return np.mean(data, 0)    # mean sample prediction over base classifiers
                      
    def get_inference(self, data):
        """Get the inferred class label for each sample.

        Args:
            data: (M method, N sample) ndarray of rank predictions
        
        Returns:
            labels: (N sample,) ndarray of inferred class labels.
        """
        labels = self.get_scores(data)
        labels[labels >= 0] = 1.
        labels[labels < 0] = -1
        return labels
