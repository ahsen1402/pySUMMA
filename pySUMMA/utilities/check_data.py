"""Check whether base classifier prediction data is correctly formatted.

Functions:
- _check_rank_data: Throws error if data are not an ndarray or sample ranks
- _check_binary_data: Throws error if data are not an ndarray or binary values [-1, 1].
"""

import numpy as np

def check_rank_data(data):
    """Test whether input data consists of rank values,
    that is specifically the integers 1,2,...,N.

    Args:
       data : (M method, N sample) ndarray of sample ranks
           with rows being base classifier rank predictions.

    Exceptions raised:
       TypeError : If data are not an numpy.ndarray
       ValueError : If data are not rank
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Base classifier predictions must by numpy.ndarray")
    
    test_rank = np.arange(1, data.shape[1] + 1)    # Array to test user input data against
    for j in range(data.shape[0]):                 # Test rank data for each method
        test_val = np.setdiff1d(test_rank, data[j, :]).size
        test_val += np.setdiff1d(data[j, :], test_rank).size
        if test_val > 0:
            raise ValueError("Base classifier predictions must be a ranked list")


def check_binary_data(data):
    """Test whether input data exclusively consists 
    of values -1 or 1.

    Args:
        data : (M method, N sample) ndarray of 
            binary predictions [-1, 1]

    Exceptions raised:
       TypeError : If data are not an numpy.ndarray
       ValueError : If data are not binary
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Base classifier predictions must by numpy.ndarray")
    
    test_binary = np.array([-1, 1])                # Array to test user input data against
    for j in range(data.shape[0]):                 # Test binary data for each method
        test_val = np.setdiff1d(test_binary, data[j, :]).size 
        test_val += np.setdiff1d(data[j, :], test_binary).size
        if test_val > 0:
            raise ValueError("Base classifier predictions must be a binary values -1 or 1")
