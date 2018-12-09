import numpy as np
import pandas as pd
import tensorflow as tf

from train_val_test_split import split

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
    n_obs = int(hparams.batch_size / p_train)
    
    dataset = np.zeros((n_obs, 10 * hparams.in_seq_len))
    dataset[1:,0] = np.random.randn(n_obs-1)
    for i in range(dataset.shape[1] - 1):
        dataset[:,i+1] = dataset[:,i] + 2 * np.pi / hparams.in_seq_len
    dataset = np.sin(dataset) + 1
    mask = np.ones_like(dataset)
    
    # Split into training, validation and test datasets
    train, val, test = split(hparams, dataset, mask)
    
    return train, val, test