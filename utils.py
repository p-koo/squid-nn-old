import os, sys
sys.dont_write_bytecode = True
import pandas as pd
import numpy as np



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


def oh2seq(OH, alphabet):
    """
    Convert one-hot encoding to sequence
    ----------
    OH : ARRAY with shape (L, 4)
        Input sequence (one-hot encoding)
    alphabet : 1D ARRAY
        All characters present in the sequence alphabet (e.g., ['A','C','G','T'] for DNA)
    
    Returns
    -------
    seq : STRING with length L
        Sequence corresponding to input one-hot encoding (e.g., 'AATGAC...')
    """
    
    seq = []
    for i in range(np.shape(OH)[0]):
        for j in range(len(alphabet)):
            if OH[i][j] == 1:
                seq.append(alphabet[j])
    seq = ''.join(seq)
    return seq


def seq2oh(seq, alphabet):
    """
    Convert sequence to one-hot encoding
    ----------
    seq : STRING with length L
        Input sequence
    alphabet : 1D ARRAY
        All characters present in the sequence alphabet (e.g., ['A','C','G','T'] for DNA)
    
    Returns
    -------
    OH : ARRAY with shape (L, 4)
        One-hot encoding corresponding to input sequence
    """
    
    L = len(seq)
    OH = np.zeros(shape=(L,len(alphabet)), dtype=np.float32)
    for idx, i in enumerate(seq):
        for jdx, j in enumerate(alphabet):
            if i == j:
                OH[idx,jdx] = 1
    return OH


def fix_gauge(x, gauge, wt=None, r=None):
    """    
    Fix the gauge for an attribution matrix
    
    x :         ARRAY with shape (L, 4)
                Matrix of attribution scores for a sequence-of-interest
    gauge :     STRING {'empirical', 'wildtype', 'hierarchical', 'default'}
                Specification of which gauge to use
    OH_wt :     ARRAY with shape (L, 4)
                Wild-type sequence (one-hot encoding); needed if gauge = 'wildtype'
    r :         Probability of mutation used during generation of in silico MAVE dataset

    Returns
    -------
    OH : ARRAY with shape (L, 4)
        Gauge-fixed one-hot encoding corresponding to input sequence
    """

    x1 = x.copy()

    if gauge == 'empirical':
        L = wt.shape[0] #length of sequence
        wt_argmax = np.argmax(wt, axis=1) #index of each wild-type in the one-hot encoding

        p_lc = np.ones(shape=wt.shape) #empirical probability matrix
        p_lc = p_lc*(r/3.)

        for l in range(L):
            p_lc[l,wt_argmax[l]] = (1-r)

        for l in range(L):
            weighted_avg = np.average(x[l,:], weights=p_lc[l,:])
            for c in range(4):
                x1[l,c] -= weighted_avg

    elif gauge == 'wildtype':
        L = wt.shape[0]
        wt_argmax = np.argmax(wt, axis=1)
        for l in range(L):
            wt_val = x[l, wt_argmax[l]]
            x1[l,:] -= wt_val

    elif gauge == 'hierarchical':
        for l in range(x.shape[0]):
            col_mean = np.mean(x[l,:])
            x1[l,:] -= col_mean

    elif gauge == 'default':
        pass

    return x1