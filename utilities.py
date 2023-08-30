import os, sys
sys.dont_write_bytecode = True
import pandas as pd



def arr2pd(x, letters=['A','C','G','T']):
    """Convert Numpy array to Pandas dataframe with proper column headings.

    Parameters
    ----------
    x : ARRAY with shape (L, 4)
        input sequence (one-hot encoding or attribution map).
    letters : 1D ARRAY
        All characters present in the sequence alphabet (e.g., ['A','C','G','T'] for DNA)

    Returns
    -------
    x : DATAFRAME
        Pandas dataframe corresponding to the input Numpy array
    """
    
    labels = {}
    idx = 0
    for i in letters:
        labels[i] = x[:,idx]
        idx += 1
    x = pd.DataFrame.from_dict(labels, orient='index').T
    
    return x