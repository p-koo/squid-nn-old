import os, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import mavenn
import logomaker
import utilities



def plot_performance(model, info, save, save_dir):
    
    # plot mavenn model performance
    fig, ax = plt.subplots(1, 1, figsize=[5, 5])
    # plot I_var_train, the variational information on training data as a function of epoch
    ax.plot(model.history['I_var'], label=r'I_var_train')
    # plot I_var_val, the variational information on validation data as a function of epoch
    ax.plot(model.history['val_I_var'], label=r'val_I_var')
    # plot I_pred_test, the predictive information of the final model on test data
    ax.axhline(info, color='C3', linestyle=':', label=r'test_I_pred')
    ax.set_xlabel('epochs')
    ax.set_ylabel('bits')
    ax.set_title('Training history')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(save_dir, 'mavenn_training.png'), facecolor='w', dpi=200)
    plt.show()


def plot_additive_logo(logo, center=True, view_window=None, alphabet=['A','C','G','T'], save=False, save_dir=None):

    # plot additive logo
    fig, ax = plt.subplots(figsize=[10,3])

    if view_window is None:
        logo_fig = logo
    else:
        logo_fig = logo[view_window[0]:view_window[1]]

    logomaker.Logo(df=utilities.arr2pd(logo_fig, alphabet),
                    ax=ax,
                    fade_below=.5,
                    shade_below=.5,
                    width=.9,
                    center_values=center,
                    font_name='Arial Rounded MT Bold')
    if view_window is not None:


        ax.set_xticks(np.arange(0, view_window[1]-view_window[0], 1))
        ax.set_xticklabels(np.arange(view_window[0], view_window[1], 1))

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Additive effect')
    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(save_dir, 'additive_logo.png'), facecolor='w', dpi=200)
    plt.show()


def plot_pairwise_matrix(theta_lclc, view_window=None, alphabet=['A','C','G','T'], save=False, save_dir=None):

    # plot maveen pairwise matrix
    fig, ax = plt.subplots(figsize=[10,5])
    ax, cb = mavenn.heatmap_pairwise(values=theta_lclc,
                                    alphabet=alphabet,
                                    ax=ax,
                                    gpmap_type='pairwise',
                                    cmap_size='2%',
                                    show_alphabet=False,
                                    cmap='seismic',
                                    cmap_pad=.1,
                                    show_seplines=True,            
                                    sepline_kwargs = {'color': 'k',
                                                      'linestyle': '-',
                                                      'linewidth': .5})
    if view_window is not None:
        ax.xaxis.set_ticks(np.arange(0, view_window[1]-view_window[0], 2))
        ax.set_xticklabels(np.arange(view_window[0], view_window[1], 2))  
    cb.set_label(r'Pairwise Effect',
                  labelpad=8, ha='center', va='center', rotation=-90)
    cb.outline.set_visible(False)
    cb.ax.tick_params(direction='in', size=20, color='white')
    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(save_dir, 'pairwise_matrix.png'), facecolor='w', dpi=200)
    plt.show()


def plot_y_vs_yhat(model, mave_df, save=False, save_dir=None):

    # plot mavenn y versus yhat
    fig, ax = plt.subplots(1,1,figsize=[5,5])
    trainval_df, test_df = mavenn.split_dataset(mave_df)
    y_test = test_df['y'] #get test data y values
    yhat_test = model.x_to_yhat(test_df['x']) #compute yhat on test data
    Rsq = np.corrcoef(yhat_test.ravel(), test_df['y'])[0, 1]**2 #compute R^2 between yhat_test and y_test    
    ax.scatter(yhat_test, y_test, color='C0', s=1, alpha=.1,
               label='test data')
    xlim = [min(yhat_test), max(yhat_test)]
    ax.plot(xlim, xlim, '--', color='k', label='diagonal', zorder=100)
    ax.set_xlabel('model prediction ($\hat{y}$)')
    ax.set_ylabel('measurement ($y$)')
    ax.set_title(f'Standard metric of model performance:\n$R^2$={Rsq:.3}');
    ax.legend(loc='upper left')
    plt.tight_layout()
    if save is True:
        plt.savefig(os.path.join(save_dir,'mavenn_measure_yhat.png'), facecolor='w', dpi=200)
    plt.show()
    
    
def plot_y_vs_phi(model, mave_df, save=False, save_dir=None):

    # plot mavenn y versus phi
    fig, ax = plt.subplots(1,1,figsize=[5,5])
    trainval_df, test_df = mavenn.split_dataset(mave_df)
    phi_test = model.x_to_phi(test_df['x']) #compute φ on test data
    phi_lim = [min(phi_test)-.5, max(phi_test)+.5] #set phi limit and create a grid in phi space
    phi_grid = np.linspace(phi_lim[0], phi_lim[1], 1000)
    yhat_grid = model.phi_to_yhat(phi_grid) #compute yhat each phi grid point
    q = [0.025, 0.975] #compute 95% CI for each yhat
    yqs_grid = model.yhat_to_yq(yhat_grid, q=q)
    ax.fill_between(phi_grid, yqs_grid[:, 0], yqs_grid[:, 1],
                   alpha=0.1, color='C1', lw=0, label='95% CI') #plot 95% confidence interval
    ax.plot(phi_grid, yhat_grid,
            linewidth=2, color='C1', label='nonlinearity') #plot GE nonlinearity
    y_test = test_df['y']
    ax.scatter(phi_test, y_test,
               color='C0', s=1, alpha=.1, label='test data',
               zorder=-100, rasterized=True) #plot scatter of φ and y values
    ax.set_xlim(phi_lim)
    ax.set_xlabel('latent phenotype ($\phi$)')
    ax.set_ylabel('measurement ($y$)')
    ax.set_title('GE measurement process')
    ax.legend(loc='upper left')
    fig.tight_layout()
    if save is True:
        plt.savefig(os.path.join(save_dir, 'mavenn_measure_phi.png'), facecolor='w', dpi=200)
    plt.show()