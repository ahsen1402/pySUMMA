"""Generate synthetic binary data [-1, 1].

Generate random samples of synthetic base classifier predictions.  User
can provide either:
 1) both the the true positive (tpr) and true negative rates (tnr), the 
    corresponding balanced accuracy (ba) is then computed by 

    ba = 0.5 * (tpr + tnr)

 2) the boundary for uniformly sampling balanced accuracies of base classifiers.
    In this setting, tpr is uniformly sampled on a calculated interval 
    that guarantees that tpr and tnr are on the interval [0, 1], and lastly
    tnr is computed from the sampled values of ba and tpr by

    tnr = 2 * ba - tpr

 3) no input.  Here classifier parameters are sampled as in (2) with the 
    default limits of [0.35, 0.9] for the balanced accuracy.

Available classes:
- Binary
"""

import numpy as np

def sample_TPR_given_BA(ba):
    """
    Uniformly sample TPR given BA on the interval in which, 
    BA, TPR and TNR are bounded by [0, 1].  How this is 
    accomplished is as follows.

    The Balanced accuracy is defined as,
    
    BA = 0.5 * (TPR + TNR)

    with TPR being the True Positive Rate and TNR being
    the True Negative Rate.  Both TPR and TNR are bounded
    on the interval [0, 1], consequently we take these 
    boundaries into account when sampling.

    From above, TPR can be written

    TPR = 2*BA - TNR

    When TNR = 0, then TPR = 2*BA.  However, if BA > 0.5,
    then TPR > 1, and outside its defined boundary.  Consequently,
    we set,
    
    tpr_max = minimum (1, 2 * ba)

    Given that TPR >= 0, when TNR is its maximum value of 1, 

    tpr_min = maximum (0, 2 * ba - 1).

    Input
    -----
    ba : ndarray
    (M,) length ndarray of balanced accuracies

    Returns
    -------
    tpr : ndarray
    (M, ) length ndarray of randomly sampled true positive rates
    """
    # inititate tpr array
    tpr = np.zeros(ba.size)

    for j in range(ba.size):
        # min TPR is when TNR is its maximum.  
        tpr_min = np.max([0, 2 * ba[j] -1])

        # Max TPR is when TNR = 0
        tpr_max = np.min([1, 2 * ba[j]])
        
        if tpr_min > tpr_max:
            raise ValueError("TPR min > TPR max")
        
        # sample tpr from computed limits
        tpr[j] = np.random.rand() * (tpr_max - tpr_min) + tpr_min

    return tpr
        

class Binary:
    """Simulate the binary predictions by an ensemble of base classifiers.

    Store input parameters, if tpr and tnr are not specified in 
    the 'rates' argument then sample according to ba_lims.
    
    Input
    -----
    M : int
    N : int
    N1 : int
    
    ba_lims : tuple
    tuple of floats where ba_lims[0] < ba_lims[1] and 
    0 <= ba_lims[i] <= 1 for i = 0, 1
    
    rates : dict (optional)
    If specified, the dictionary keys need to be 'tpr' and 'tnr', and the corresponding
    values are (M,) length ndarrays of floats between 0 and 1.  The jth entry
    of the ndarray associated with tpr and tnr keys are the true positive 
    and true negative rates of the jth classifier, respectively.

    Returns
    -------
    noting, but stores the following class properties
    
    self.M : int
    self.N : int
    self.N1 : int
    self.tpr : (M,) length ndarray
    self.tnr : (M,) length ndarray
    self.ba : (M,) length ndarray
    self.labels (M,) length ndarray
    """
    def __init__(self, M, N, N1, 
                 ba_lims=(0.35, 0.9),
                 rates = None):
        self.M = M
        self.N = N
        self.N1 = N1
        # If rates are given, then first check that inputed rate is a
        #    1) a python dictionary with keys
        #    2) keys include tpr and tnr
        #    3) each value is an M length ndarray
        #    4) each entry is between 0 and 1.
        # and then set tpr and tnr 
        if rates is None:
            self.ba, self.tpr, self.tnr = self._sample_rates_from_ba_lims(ba_lims)
        else:
            if type(rates) is not dict:
                raise TypeError("Rates must be a dictionary with keys\n 'tpr' and 'tnr' and the values as M length\nnumpy arrays of rates.")
            else:
                for j in range(self.M):
                    if rates["tpr"][j] < 0 or rates["tpr"][j] > 1:
                        raise ValueError("Each tpr must be between 0 and 1")
                    
                    if rates["tnr"][j] < 0 or rates["tnr"][j] > 1:
                        raise ValueError("Each tnr must be between 0 and 1")

            self.tpr = rates["tpr"]
            self.tnr = rates["tnr"]
            
        # set labels
        self.labels = np.hstack([np.ones(N1), 
                                 -np.ones(N-N1)])
    
    def _sample_rates_from_ba_lims(self, ba_lims):
        """
        Uniformly sample balanced accuracy (ba) values for each of the
        M base classifiers.  Sample true positive rates (tpr), and compute 
        true negative rates from the sampled (BA) values.

        Input
        -----
        ba_lims : python list of floats
        The lower bound (ba_lims[0]) and upper bound (ba_lims[1]) for
        uniformly sampling balanced accuracies.

        Returns
        -------
        ba : ndarray
        (M,) length ndarray of balanced accuracies

        tpr : ndarray
        (M,) length ndarray of true positive rates
        
        tnr : ndarray
        (M,) length ndarray of true negative rates
        """
        # check that ba_lims is:
        #    1) that the first entry is less than the second,
        #    2) that each entry is between [0, 1]
        
        if ba_lims[0] >= ba_lims[1]:
            raise ValueError("ba_lims[0] must be less than ba_lims[1]")
        elif ba_lims[0] < 0 or ba_lims[1] > 1:
            raise ValueError("B.A. limits must be between 0 and 1")

        # uniformly sample balanced accuracy for each method on the 
        # inputed interval
        ba = np.random.rand(self.M) * (ba_lims[1] - ba_lims[0]) + ba_lims[0]

        # sample TPR given the contstraintes of the sampled ba
        tpr = sample_TPR_given_BA(ba)

        # from ba and tpr, compute the tnr
        tnr = 2*ba - tpr
        return [ba, tpr, tnr]

    def sim(self):
        """Generate simulation data, and store as class properties.
        
        Generated properties:
        self.data : ndarray
        (M, N) ndarray of binary [-1, 1] predictions

        
        self.data : ndarray
        (M, N) ndarray of M binary classifier binary predictions of N samples
        """
        # initialize ndarrays
        self.data = np.zeros(shape=(self.M, self.N))

        # generate samples for each classifier
        for j in range(self.M):
            
            # loop over samples
            for i in range(self.N):
                # generate random number u between 0 and 1
                u = np.random.rand()
                
                # if true sample label is positive, then sample
                # from true positive rate
                if self.labels[i] == 1:
                    self.data[j, i] = 1 if u <= self.tpr[j] else -1
                # if samples are not from the positive class, they are from negative class
                else:
                    self.data[j, i] = -1 if u <= self.tnr[j] else 1

    def get_ba(self):
        """Compute the Balanced Accuracy from TPR and TNR.

        Returns:
            The balanced accuracies of M base classifiers ((M,) ndarray)
        """
        return 0.5*(self.tpr + self.tnr)
