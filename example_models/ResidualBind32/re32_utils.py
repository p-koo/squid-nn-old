import os, sys
sys.dont_write_bytecode = True
import tensorflow as tf
import yaml
pyDir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(pyDir))
import losses



def get_config(run_path):
    '''
    This function returns config of a wandb run as a dictionary
    :param run_path: dir with run outputs
    :return: dictionary of configs
    '''
    config_file = os.path.join(run_path, 'files', 'config.yaml')
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def read_model(run_path, compile_model=False):
    '''
    This function loads a per-trained model
    :param run_path: run output dir
    :param compile_model: bool compile model using loss from config
    :return: model and resolution
    '''
    if run_path.endswith('.h5'):
        trained_model = tf.keras.models.load_model(run_path, custom_objects={"GELU": GELU})
        bin_size = ''
    else:
        config = get_config(run_path)  # load wandb config
        if 'bin_size' in config.keys():
            bin_size = config['bin_size']['value']  # get bin size
        else:
            bin_size = 'NA'
        model_path = os.path.join(run_path, 'files', 'best_model.h5')  # pretrained model
        # load model
        trained_model = tf.keras.models.load_model(model_path, custom_objects={"GELU": GELU})
    if compile_model:
        loss_fn_str = config['loss_fn']['value']  # get loss
        loss_fn = eval('losses.' + loss_fn_str)()  # turn loss into function
        trained_model.compile(optimizer="Adam", loss=loss_fn)
    return trained_model, bin_size  # model and bin size


class GELU(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x