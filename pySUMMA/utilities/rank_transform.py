import numpy as np
from scipy.stats import rankdata


# ==========================================
# ==========================================

def ranks(data):
    '''
    Test whether inputed data are contain sample ranks.  If data are determined to be non-rank values, transform to rank

    Input
    -----
    Either
        - (N,) numpy array of sample scores or rank
        - (M, N) numpy array where each row represents the sample predictions by a base classifier, and column a specific sample.

    Return
    ------
    Either
        - (N,) numpy array of sample ranks
        - (M, N) numpy array of sample ranks
    '''
    if len(data.shape) == 1:
        if _is_rank(data) == False:
            return _compute_ranks(data)
        else:
            return data
    elif len(data.shape) == 2:
        rdata = np.zeros(shape=data.shape)
        for j in range(data.shape[0]):
            if _is_rank(data[j, :]) == False:
                rdata[j, :] = _compute_ranks(data[j, :])
            else:
                rdata[j, :] = data[j, :]
        return rdata
    else:
        raise ValueError("Input must either be a Rank 1 or Rank 2 Tensor")

# ==========================================
# Compute the rank values
# ==========================================

def _compute_ranks(data):
    '''
    Given a numpy array, convert data scores to sample rank.  Here assign samples of with a large score low rank.

    Input
    -----
    (N,) numpy array of sample scores

    Return
    ------
    (N, ) numpy array of sample ranks
    '''
    if np.isnan(np.sum(data)):
        tmp = _replace_nans(data)
    else:
        tmp = data.copy()
    idx = np.arange(tmp.size)
    np.random.shuffle(idx)
    rdata = rankdata(tmp[idx], method='ordinal')
    rdata = rdata.max() + 1 - rdata
    return rdata[np.argsort(idx)]

# ==========================================
# Replace NaNs
# ==========================================

def _replace_nans(data):
    """
    Find NaNs, if present give all NaN samples the median score

    Input
    -----
    (N,) numpy array of sample scores

    Return
    ------
    (N,) numpy array in which NaNs are replaced with the median score value
    """
    tmp = data.copy()
    idx = np.where(np.isnan(tmp))[0]
    tmp[idx] = np.median(tmp[~np.isnan(tmp)])
    return tmp

# ==========================================
# Check whether inputed data are already ranks
# ==========================================

def _is_rank(data):
    """
    Test whether input data consists of values 1,2,...,N

    Input
    -----
    (N,) numpy array of data

    Return
    ------
    True if data are in rank and False otherwise
    """
    N = data.size
    return np.setdiff1d(data, np.arange(1, N+1)).size == 0


# ==========================================
#
# ==========================================
