import numpy as np

def third(data):
    '''
    Input
    -----
    numpy array M base classifier x N samples numpy array

    Returns
    3rd order tensor of the third central moment
    '''
    M = data.shape[0]
    N = data.shape[1]
    # subtract mean from data
    tmp = data - np.tile(np.mean(data, 1).reshape(M, 1), (1, N))
    # instantiate tensor
    T = np.zeros(shape=(M, M, M))
    # loop over methods along the third mode
    for w in range(data.shape[0]):
        T[:,:, w] = np.dot(np.tile(tmp[w, :], (M, 1)) * tmp, tmp.T) / N
    return T
