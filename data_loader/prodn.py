import numpy as np
import pandas as pd
import tensorflow as tf

from .train_val_test_split import split

def load_data(hparams, mode='mini', normalise='global_max'):
    """
    Arguments:
        hparams: tf hyperparameters
        mode:
            -'mini': Get dataset with only one batch
            -'midi': Get dataset with one year of data
            -'maxi': Get dataset with each year's data
        normalise:
            -'global_max': Scale all observations down according to the largest value in the dataset
            -'local_max': Scale each observation down according to the largest value in the row
    """ 
    # Calculate train, validate and test set sizes
    p_val, p_test = hparams.val_split, hparams.test_split
    p_train = 1 - p_val - p_test
    
    # Load dataset
    prodn = pd.read_pickle('../../full_6d.pkl').values
    print(prodn.shape)
    
    # Load one batch of data only
    if mode == 'mini':
        print('Loading Data - Mode: mini')
        end = int(hparams.batch_size / p_train)
        dataset = np.nan_to_num(prodn)[2:2+end]
        mask = np.isfinite(prodn)[2:2+end]
    
    # Load one year of data only
    if mode == 'midi':
        print('Loading Data - Mode: midi')
        dataset = np.nan_to_num(prodn)
        mask = np.isfinite(prodn)
    
    # Load all years of data on an annual cycle
    if mode == 'maxi':
        print('Loading Data - Mode: maxi')
        dataset = [prodn[:, i : i + hparams.in_seq_len * hparams.input_dim + hparams.out_seq_len + 1] \
                    for i in range(0, prodn.shape[1] - hparams.in_seq_len * hparams.input_dim - hparams.out_seq_len,
                                hparams.in_seq_len)]
        dataset = np.vstack(dataset)
        mask = np.isfinite(dataset)
        dataset = np.nan_to_num(dataset)
    
    # Load all years of data on a weekly cycle
    if mode == 'supermaxi':
        print('Loading Data - Mode: supermaxi')
        dataset = [prodn[:, i : i + hparams.in_seq_len * hparams.input_dim + hparams.out_seq_len + 1] \
                    for i in range(0, prodn.shape[1] - hparams.in_seq_len * hparams.input_dim - hparams.out_seq_len)]
        dataset = np.vstack(dataset)
        mask = np.isfinite(dataset)
        dataset = np.nan_to_num(dataset)

    # Split into training, validation and test datasets
    train, val, test = split(hparams, dataset, mask)
    
    return train, val, test

def x_var(hparams, mode='maxi', normalise='global_max'):
    prodn = pd.read_pickle('../../full_6d.pkl').values
    prodn = np.reshape(prodn, -1)
    prodn = prodn[~np.isnan(prodn)]
    x_var = np.cov(prodn)

    return x_var