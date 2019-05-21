
import numpy as np

# ========================================================
# ========================================================
# Gaussian simulations
# ========================================================
# ========================================================

def gaussians(M, N, N1, dmu):
    '''
    Generate N sample scores by M base classifiers using gaussian random samples.  The performance of each base classifier is chosen randomly between [0, performance)

    Input
    -----
    M : integer
        number of base classifiers
    N : integer
        number of samples
    N1 : integer
        number of samples from class 1
    cov : C

    Returns:
    python dictionary
        r : M x N numpy array of scores
        labels : N length numpy array of class labels
    '''
    # labels
    labels = np.zeros(N)
    labels[:N1] = 1.

    # sample from class 1
    delta_mu = np.tile(dmu.reshape(M, 1), (1, N1))
    s1 = delta_mu + np.random.randn(M, N1)

    # sample from class 0
    s0 = np.random.randn(M, N-N1)
    
    # concatenating samples drawn from each class
    return [labels, np.hstack([s1, s0])]

# ========================================
# ========================================
