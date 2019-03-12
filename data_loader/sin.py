import numpy as np
import pandas as pd
import tensorflow as tf

from .train_val_test_split import split

def load_data(hparams, mode='fixed_frequency', normalise='fixed_scale', shuffle=False):
    """
    Arguments:
        hparams: tf hyperparameters
        mode:
            -'fixed_frequency': sin wave frequency is 1 / hparams.in_seq_len
            -'random_frequency': sin wave frequency is in range 3*hparams.in_seq_len to 0.3*hparams.in_seq_len
        normalise:
            -'fixed_scale': Returns sin curves with amplitude 2
            -'random_scale': Randomly scale curve to have amplitude between 0.2 and 2
            -'random_scale_and_offset': Randomly scale curve to have amplitude between 0.2 and 2
                                       Randomly offset curve so min >= 0 and max <= 2
    """ 
    # Calculate train, validate and test set sizes
    p_val, p_test = hparams.val_split, hparams.test_split
    p_train = 1 - p_val - p_test
    n_obs = int(hparams.batch_size / p_train)
    
    dataset = np.zeros((n_obs, 10 * hparams.in_seq_len))
    dataset[1:,0] = np.random.randn(n_obs-1)
    for i in range(dataset.shape[1] - 1):
        dataset[:,i+1] = dataset[:,i] + 2 * np.pi / hparams.in_seq_len

    if mode == 'random_frequency':
        dataset *= np.random.uniform(0.3, 3, (n_obs, 1))

    dataset = (np.sin(dataset) + 1) / 2
    
    if normalise.find('random_scale') >= 0:
        divisor = np.random.uniform(1, 10, (n_obs, 1))
        dataset /= divisor
        
    if normalise == 'random_scale_and_offset':
        offset = np.random.uniform(0, 1 - 1 / divisor, (n_obs, 1))
        dataset += offset
    
    mask = np.ones_like(dataset)
    
    # Split into training, validation and test datasets
    train, val, test = split(hparams, dataset, mask, normalise=None)
    
    return train, val, test