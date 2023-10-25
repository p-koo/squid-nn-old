"""
Library of functions for DNN inference on mutagenized sequences
"""

import os, sys
sys.dont_write_bytecode = True
import numpy as np
#import impress


class BasePredictor():

    def __init__(self, pred_fun, reduce_fun, task_idx, batch_size):
        self.pred_fun = pred_fun
        self.reduce_fun = reduce_fun
        self.task_idx = task_idx
        self.batch_size = batch_size

    def __call__(self, x):
        """Return an in silico MAVE based on mutagenesis of 'x'.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (L, A)).

        Returns
        -------
        torch.Tensor
            Batch of one-hot sequences with random augmentation applied.
        """
        raise NotImplementedError()



class ScalarPredictor(BasePredictor):

    def __init__(self, pred_fun, task_idx=0, batch_size=64, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.kwargs = kwargs
        self.batch_size = batch_size

    def __call__(self, x):
        pred = prediction_in_batches(x, self.pred_fun, self.batch_size, **self.kwargs)
        print(pred)
        #return pred[:,self.task_idx][:,np.newaxis]
        return pred[self.task_idx]



class ProfilePredictor(BasePredictor):

    def __init__(self, pred_fun, task_idx=0, batch_size=64, reduce_fun=np.sum, save_dir=None, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        self.reduce_fun = reduce_fun
        #self.axis = axis
        self.save_dir = save_dir
        self.kwargs = kwargs

    def __call__(self, x):
        # get model predictions (all tasks)
        pred = prediction_in_batches(x, self.pred_fun, self.batch_size, **self.kwargs)

        # reduce profile to scalar across axis for a given task_idx
        pred = self.reduce_fun(pred[:,:,self.task_idx], save_dir=self.save_dir)
        return pred[:,np.newaxis]



class BPNetPredictor(BasePredictor):

    def __init__(self, pred_fun, task_idx=0, batch_size=64, reduce_fun=np.sum, axis=1, strand='pos', **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        self.reduce_fun = reduce_fun
        self.axis = axis
        self.kwargs = kwargs
        if strand == 'pos':
            self.strand = 0
        else:
            self.strand = 1

    def __call__(self, x):

        # get model predictions (all tasks)
        pred = prediction_in_batches(x, self.pred_fun, self.batch_size, **self.kwargs)

        # reduce bpnet profile prediction to scalar across axis for a given task_idx
        pred = pred[self.task_idx][0][:,self.strand]
        pred = self.reduce_fun(pred, axis=self.axis)
        print(pred.shape)
        return pred[:,np.newaxis]




################################################################################
# useful functions
################################################################################


def prediction_in_batches(x, model_pred_fun, batch_size=None, **kwargs):

    N, L, A = x.shape
    num_batches = np.floor(N/batch_size).astype(int)
    pred = []
    for i in range(num_batches):
        pred.append(model_pred_fun(x[i*batch_size:(i+1)*batch_size], **kwargs))
    if num_batches*batch_size < N:
        pred.append(model_pred_fun(x[num_batches*batch_size:], **kwargs))

    try:
        preds = np.concatenate(pred, axis=1)
    except ValueError as ve:
        preds = np.vstack(pred)
    return preds



def profile_sum(pred, save_dir=None):

    sum = np.sum(pred, axis=1)
    return sum


def profile_pca(pred, save_dir=None):

    N, B = pred.shape #B : number of bins in profile
    Y = pred.copy()
    sum = np.sum(pred, axis=1) #needed for sense correction

    # normalization: mean of all distributions is subtracted from each distribution
    mean_all = np.mean(Y, axis=0)
    for i in range(N):
        Y[i,:] -= mean_all

    print('    Computing SVD...')
    u,s,v = np.linalg.svd(Y.T, full_matrices=False)
    vals = s**2 #eigenvalues
    vecs = u #eigenvectors
    print('    SVD complete')
    
    U = Y.dot(vecs)
    v1, v2 = 0, 1
    
    corr = np.corrcoef(sum, U[:,v1])
    if corr[0,1] < 0: #correct for eigenvector "sense"
        U[:,v1] = -1.*U[:,v1]
        print('    Corrected eigenvector sense')

    #impress.plot_eig_vals(vals, save_dir=save_dir)
    #impress.plot_eig_vecs(U, v1=v1, v2=v2, save_dir=save_dir)

    return U[:,v1]


"""
def custom_reduce(pred):
    # code to reduce predictions to (N,1)
    return pred_reduce
"""