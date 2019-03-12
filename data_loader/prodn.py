import numpy as np
import pandas as pd
import tensorflow as tf

from .train_val_test_split import split

def load_data(hparams, mode='mini', normalise='global_max', is_autoregressive=False, time_encoding=None,
              winter_mask=False, prodn_bins=False):
    """
    Arguments:
        hparams: tf hyperparameters
        mode:
            -'mini':   Get dataset with only one batch
            -'midi':   Get dataset with one year of data
            -'midi_w': Get dataset with one year of data on a weekly cycle
            -'maxi':   Get dataset with each year's data
        normalise:
            -'global_max': Scale all observations down according to the largest value in the dataset
            -'local_max':  Scale each observation down according to the largest value in the row
    """ 
    # Calculate train, validate and test set sizes
    p_val, p_test = hparams.val_split, hparams.test_split
    p_train = 1 - p_val - p_test
    
    # Load dataset
    prodn = pd.read_pickle('../../full_6d.pkl').values
    h, w = prodn.shape
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
    
    # Load one year's data on a weekly cycle
    if mode == 'midi_w':
        print('Loading Data - Mode: midi_w')
        dataset = [prodn[:, i : i + hparams.in_seq_len * hparams.input_dim + hparams.out_seq_len + 1] \
                    for i in range(0, hparams.in_seq_len, 4)]
        dataset = np.vstack(dataset)
        mask = np.isfinite(dataset)
        dataset = np.nan_to_num(dataset)
    
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

    # Add masking for winter stopping
    if winter_mask:
        winter_mask = np.dstack([prodn_[:,0:-2], prodn_[:,1:-1], prodn_[:,2:]])
        winter_mask = winter_mask.sum(axis=2)
        winter_mask[winter_mask > 0] = 1
        for i in range(int(np.floor(winter_mask.shape[1] / 61))):
            i = 61 * i
            winter_mask[:,i:i+61] = winter_mask[:,i:i+61].min(axis=1, keepdims=True)
        mask = np.dstack([mask, winter_mask])
        del(winter_mask)
        
    # Add one-hot production bins
    if prodn_bins:
        # Calculate annual production volumes
        a = np.nan_to_num(prodn)
        a = np.cumsum(a, axis=1)[:,::61]
        a = a[:,1:] - a[:,:-1]
        h_a, w_a = a.shape

        # Create bins
        # TODO: Allow bins to be passed as an argument
        from scipy import stats
        bins = stats.mstats.mquantiles(a[a>0], [i/15 for i in range(15)])
        bins = np.hstack([[0], bins, [10000000]])        
        a = np.digitize(a, bins) - 1

        bins = np.zeros((h_a, w_a, len(bins)-1))
        bins[np.repeat(np.arange(h_a), w_a), np.arange(h_a*w_a) % w_a, np.reshape(a, -1)] = 1

        
    # Add time encoding
    if time_encoding != None:
        time_enc = np.ones((mask.shape[0], mask.shape[1], len(time_encoding)))
        time_enc[:,0] = 0
        time_enc /= np.reshape(time_encoding, (1,1,-1))
        time_enc  = time_enc.cumsum(axis=1)
        time_enc *= 2 * np.pi
        mask = np.dstack([mask, np.sin(time_enc), np.cos(time_enc)])
        del(time_enc)

    # Split into training, validation and test datasets
    train, val, test = split(hparams, dataset, mask, normalise=normalise, is_autoregressive=is_autoregressive)
    
    return train, val, test

def x_var(hparams, mode='maxi', normalise='global_max'):
    prodn = pd.read_pickle('../../full_6d.pkl').values
    prodn = np.reshape(prodn, -1)
    prodn = prodn[~np.isnan(prodn)]
    x_var = np.cov(prodn)

    return x_var