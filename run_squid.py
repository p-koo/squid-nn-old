import os, sys
sys.dont_write_bytecode = True
import numpy as np
import mutagenizer
import predictor
import mave
import surrogate_zoo
import impress
py_dir = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# Computational settings
# =============================================================================
gpu = False
save = True

# =============================================================================
# Setup for user-defined deep learning model (choose one)
# =============================================================================
if 0:
    # import ResidualBind-32 model and define hyperparameters
    model_name = 'ResidualBind32'
    from models.ResidualBind32 import re32_utils
    model, bin_size = re32_utils.read_model(os.path.join(py_dir,'models/%s/model' % model_name), compile_model=True)
    """
        Inputs shape:   (n, 2048, 4)
        Outputs shape:  (1, 64, 15) : 64-bin-resolution profile for each of the 15 cell lines
    """

    # define sequence-of-interest and output head (task)
    seq_index = 834
    task_idx = 13 #cell line

    # retrieve sequence-of-interest from test set
    import h5py
    with h5py.File(os.path.join(py_dir, 'models/%s/cell_line_%s.h5' % (model_name, task_idx)), 'r') as dataset:
        x_all = np.array(dataset['X']).astype(np.float32)

    # define mutagenesis window for sequence
    start_position = 1042-10 #e.g., the nucleotide at position 1042 is the start of a pronounced AP1 motif (TGANTCA) in cell line 13
    stop_position = 1042+7+10

    # set up predictor class for in silico MAVE
    pred_generator = predictor.ProfilePredictor(pred_fun=model.predict_on_batch, 
                                                task_idx=task_idx, batch_size=512,
                                                reduce_fun=np.sum)
    
    alphabet = ['A','C','G','T']
    log2FC = False
    output_skip = 0
    

elif 1:
    # import DeepSTARR model and define hyperparameters
    model_name = 'DeepSTARR'
    import kipoi
    model = kipoi.get_model('DeepSTARR')
    """
    Inputs shape:   (n, 249, 4)
    Outputs shape:  (2,)
    """

    # define sequence-of-interest and output head (task)
    seq_index = 24869
    task_idx = 0 #developmental (DEV) class

    # retrieve genomic sequences from test set
    import h5py
    with h5py.File(os.path.join(py_dir, 'models/%s/deepstarr_data.h5' % model_name), 'r') as dataset:
        x_all = np.array(dataset['x_test']).astype(np.float32)

   # define mutagenesis window for sequence
    start_position = 123-10 #e.g., the nucleotide at position 123 is the start of a pronounced GATA motif (GATAA) in the DEV program
    stop_position = 123+5+10

    # set up predictor class for in silico MAVE
    pred_generator = predictor.ScalarPredictor(pred_fun=model.predict_on_batch, 
                                               task_idx=task_idx, batch_size=512)
    
    alphabet = ['A','C','G','T']
    log2FC = False
    output_skip = 0

else:
    # empty section for user-defined model
    pass


# =============================================================================
# Setup for workflow independent from choice of deep learning model
# =============================================================================
x = x_all[seq_index]
seq_length = x.shape[0]
mut_window = [start_position, stop_position]

# create save directory
if save is True:
    save_dir = os.path.join(py_dir, 'squid_outputs/%s/%s' % (model_name, seq_index))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# set up mutagenizer class for in silico MAVE
mut_generator = mutagenizer.RandomMutagenesis(mut_rate=0.1, uniform=False)

# generate in silico MAVE
mave = mave.InSilicoMAVE(mut_generator, pred_generator, seq_length, mut_window=mut_window)
x_mut, y_mut = mave.generate(x, num_sim=1000, seed=None)

# set up surrogate model
gpmap = 'additive' #options: {'additive', 'pairwise' if MAVE-NN}
surrogate_model = surrogate_zoo.SurrogateMAVENN(x_mut.shape, num_tasks=y_mut.shape[1],
                                                gpmap=gpmap, regression_type='GE',
                                                linearity='nonlinear', noise='SkewedT',
                                                noise_order=2, reg_strength=0.1,
                                                alphabet=alphabet,
                                                deduplicate=True, gpu=gpu)

# train surrogate model
surrogate, mave_df = surrogate_model.train(x_mut, y_mut, learning_rate=5e-4, epochs=500, batch_size=100,
                                           early_stopping=True, patience=25, restore_best_weights=True,
                                           log2FC=log2FC, save=save, save_dir=save_dir, verbose=True)

# retrieve model parameters
params = surrogate_model.get_params(gauge='empirical', save=True, save_dir=save_dir)

# generate sequence logo
logo = surrogate_model.get_logo(mut_window=mut_window, full_length=seq_length)

# retrieve model performance metrics
info = surrogate_model.get_info(save=save, save_dir=save_dir, verbose=True)

# plot figures
impress.plot_y_hist(mave_df, save=save, save_dir=save_dir)
impress.plot_performance(surrogate, info=info, save=save, save_dir=save_dir) #plot model performance (bits)
impress.plot_additive_logo(logo, center=True, view_window=mut_window, alphabet=alphabet, save=save, save_dir=save_dir)
if gpmap == 'pairwise':
    impress.plot_pairwise_matrix(params[2], view_window=mut_window, alphabet=alphabet, save=save, save_dir=save_dir)
impress.plot_y_vs_yhat(surrogate, mave_df=mave_df, save=save, save_dir=save_dir)
impress.plot_y_vs_phi(surrogate, mave_df=mave_df, save=save, save_dir=save_dir)




# Parameters to include above:
'''
output_skip = 0 #ENFORMER
pred_trans_delimit = None #ENFORMER
max_in_mem = False #ENFORMER
'''