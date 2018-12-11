import numpy as np
import pandas as pd
import tensorflow as tf

from .train_val_test_split import split

def load_data(hparams, mode='AT305', normalise='global_max', shuffle=False):
    """
    Arguments:
        hparams: tf hyperparameters
        mode:
            -'AT305': Load univariate AT305 dataset
        normalise:
            -'global_max': Scale all observations down according to the largest value in the dataset
            -'local_max': Scale each observation down according to the largest value in the row
    """ 
    # Calculate train, validate and test set sizes
    p_val, p_test = hparams.val_split, hparams.test_split
    p_train = 1 - p_val - p_test
    
    # Load dataset
    prodn = pd.read_csv('../../Data/cr2c_opdata_TMP_PRESSURE_TEMP_WATER_COND_GAS_PH_DPI_LEVEL.csv')

    if mode == 'AT305':
        dataset = prodn['AT305'].values[:10000]
    else:
        dataset = prodn[mode.split()].values[:10000]

    dataset = [dataset[i : i + hparams.in_seq_len * hparams.input_dim + hparams.out_seq_len + 1] \
               for i in range(0, 10000 - hparams.in_seq_len * hparams.input_dim - hparams.out_seq_len)]

    if shuffle:
        np.random.seed(230)
        np.random.shuffle(dataset)
    mask = np.isfinite(dataset)
    dataset = np.nan_to_num(dataset)
    
    # Split into training, validation and test datasets
    train, dev, test = split(hparams, dataset, mask, normalise=normalise)
    
    return train, dev, test

def x_var(hparams, mode='AT305', normalise='global_max'):
    prodn = pd.read_csv('../../Data/cr2c_opdata_TMP_PRESSURE_TEMP_WATER_COND_GAS_PH_DPI_LEVEL.csv')

    if mode == 'AT305':
        dataset = prodn['AT305'].values[:10000]
    else:
        dataset = prodn[mode.split()].values[:10000]

    dataset = [dataset[i : i + hparams.in_seq_len * hparams.input_dim + hparams.out_seq_len + 1] \
               for i in range(0, 10000 - hparams.in_seq_len * hparams.input_dim - hparams.out_seq_len)]

    return np.var(dataset, axis=(0,1))