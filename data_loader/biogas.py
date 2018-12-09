import numpy as np
import pandas as pd
import tensorflow as tf

from train_val_test_split import split

def load_data(hparams, mode='AT305', normalise='global_max', shuffle=False):
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
    prodn = pd.read_csv('../../Data/cr2c_opdata_TMP_PRESSURE_TEMP_WATER_COND_GAS_PH_DPI_LEVEL.csv')

    if mode == 'AT305':
        dataset = prodn['AT305'].values[:10000]
        dataset = [dataset[i : i + hparams.in_seq_len * hparams.input_dim + hparams.out_seq_len + 1] \
                   for i in range(0, 10000 - hparams.in_seq_len * hparams.input_dim - hparams.out_seq_len)]
        dataset = np.nan_to_num(dataset)
        if shuffle:
            np.random.seed(230)
            np.random.shuffle(dataset)
        mask = np.isfinite(dataset)
    
    # Split into training, validation and test datasets
    train, val, test = split(hparams, dataset, mask)
    
    return train, val, test