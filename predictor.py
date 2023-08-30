import os, sys
sys.dont_write_bytecode = True
import numpy as np


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
        return pred[:,self.task_idx][:,np.newaxis]



class ProfilePredictor(BasePredictor):

    def __init__(self, pred_fun, task_idx=0, batch_size=64, reduce_fun=np.sum, axis=1, **kwargs):
        self.pred_fun = pred_fun
        self.task_idx = task_idx
        self.batch_size = batch_size
        self.reduce_fun = reduce_fun
        self.axis = axis
        self.kwargs = kwargs

    def __call__(self, x):
        # get model predictions (all tasks)
        pred = prediction_in_batches(x, self.pred_fun, self.batch_size, **self.kwargs)

        # reduce profile to scalar across axis for a given task_idx
        pred = self.reduce_fun(pred[:,:,self.task_idx], axis=self.axis)
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
    return np.vstack(pred)



def profile_pca(pred):
    N, L = pred.shape
    sum = np.sum(pred, axis=1)

    mean = pred - np.mean(pred, axis=0, keepdims=True)
    u,s,v = np.linalg.svd(mean.T, full_matrices=False)
    vals = s**2 #eigenvalues
    vecs = u #eigenvectors
    U = mean.dot(vecs)

    # correct for eigenvector "sense"
    corr = np.corrcoef(sum, U[:,0][:-1])
    if corr[0,1] < 0:
        U[:,0] *= -1
    return U[:,0][:-1]


# if log2FC is True:
#     pred_min = mave_custom['y'].min()
#     mave_custom['y'] += (abs(pred_min) + 1)
#     pred_scalar_wt += (abs(pred_min) + 1)
#     pred_scalar_wt = np.log2(pred_scalar_wt)
#     mave_custom['y'] = mave_custom['y'].apply(lambda x: np.log2(x))
#     mave_custom['y'] = mave_custom['y'].fillna(0)
#     mave_custom.replace([np.inf, -np.inf], 0, inplace=True)



"""
def custom_reduce(pred):
    # code to reduce predictions to (N,1)
    return pred_reduce
"""