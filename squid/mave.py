"""
Functions for performing in silico MAVE
"""

import os, sys
sys.dont_write_bytecode = True
import numpy as np


class InSilicoMAVE():
    """Module for performing in silico MAVE.

    Parameters
    ----------
    mut_rate : float, optional
        Mutation rate for random mutagenesis (defaults to 0.1).
    start_position : int, optional
        Index of start position along sequence to probe (defaults to 0).
    stop_position : int, optional
        Index of stop position along sequence to probe (defaults to None).
    uniform : bool
        uniform (True), Poisson (false); sets the number of mutations per sequence
    """
    def __init__(self, mut_generator, mut_predictor, seq_length, mut_window=None, log2FC=False):
        self.mut_generator = mut_generator
        self.mut_predictor = mut_predictor
        self.seq_length = seq_length
        self.mut_window = mut_window
        self.log2FC = log2FC
        if mut_window is not None:
            self.start_position = mut_window[0]
            self.stop_position = mut_window[1]
        else:
            self.start_position = 0
            self.stop_position = seq_length


    def generate(self, x, num_sim, seed=None, verbose=1):
        """Randomly mutate segments in a set of one-hot DNA sequences.

        Parameters
        ----------
        x : torch.Tensor
            Batch of one-hot sequences (shape: (L,A)).
        num_sim : int
            Number of sequences to mutagenize for in silico MAVE.
        seed : int, optional
            sets the random number seed

        Returns
        -------
        torch.Tensor
            Sequences with randomly mutated segments (padded to correct shape
            with random DNA)
        """
        np.random.seed(seed)
        if verbose:
            print('')
            print('Building in silico MAVE...')

        # generate in silico MAVE based on mutagenesis strategy
        if verbose:
            print('  Generating mutagenized sequences...')
        if self.mut_window is not None:
            x_window = self.delimit_range(x, self.start_position, self.stop_position)
            x_mut = self.mut_generator(x_window, num_sim)
        else:
            x_mut = self.mut_generator(x, num_sim)

        # predict MAVE data and process
        if verbose:
            print('  Predicting mutagenized sequences...')
        if self.mut_window is not None:
            x_mut_full = self.pad_mave(x_mut, x, self.start_position, self.stop_position)
            y_mut = self.mut_predictor(x_mut_full)
        else:
            y_mut = self.mut_predictor(x_mut)

        if self.log2FC is True:
            y_mut = self.apply_log2FC(y_mut)

        return x_mut, y_mut


    def pad_mave(self, x_mut, x, start_position, stop_position):
        N = x_mut.shape[0]
        x = x[np.newaxis,:]
        x_start = np.tile(x[:,:start_position,:], (N,1,1))
        x_stop = np.tile(x[:,stop_position:,:], (N,1,1))
        return np.concatenate([x_start, x_mut, x_stop], axis=1)


    def delimit_range(self, x, start_position, stop_position):
        return x[start_position:stop_position,:]
    

    def apply_log2FC(self, y):

        if np.amin(y) < 0:
            y += (abs(np.amin(y)) + 1)
        elif 0 <= np.amin(y) < 1:
            y += 1

        y_log2_wt = np.log2(y[0])
        y_log2_all = np.log2(y)
        y_log2_fc = y_log2_wt - y_log2_all

        return y_log2_fc