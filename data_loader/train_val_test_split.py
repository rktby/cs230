import numpy as np
import tensorflow as tf


def split(hparams, dataset, mask, normalise='global_max'):
    # Extract parameters defining dataset shape
    in_len, out_len, in_dim = hparams.in_seq_len, hparams.out_seq_len, hparams.input_dim
    end_pos = in_len * (in_dim - 1) + 1

    p_val, p_test = hparams.val_split, hparams.test_split
    p_train = 1 - p_val - p_test

    # Calculate normalisation factor
    if normalise == 'local_max':
        # Normalise the data line by line
        x_max, x_var = np.max(dataset[:,:in_len * in_dim], axis=1), np.var(dataset)
        x_max[x_max==0] = x_max.max()
        x_max = x_max.reshape(-1,1,1)
    elif normalise == 'global_max':
        # Normalise the data based on glabal maximum
        x_max = np.max(dataset)
    else:
        # Do not normalise the data
        x_max = 1.

    # Create x dataset
    # x.shape = (n_obs, in_seq_len, input_dim)
    x = np.array([dataset[:,pos:pos+in_len] for pos in range(0,end_pos,hparams.in_seq_len)])
    x = np.rollaxis(x, 0, 3) / x_max

    # Create y dataset
    y =   dataset[:,end_pos+in_len:end_pos+in_len+out_len, np.newaxis] / x_max
    y_mask = mask[:,end_pos+in_len:end_pos+in_len+out_len, np.newaxis]

    #####################
    # Build into datasets
    #####################
    train_pos = int(x.shape[0] * p_train)
    val_pos   = int(x.shape[0] * (p_train + p_val))

    dataset = tf.data.Dataset.from_tensor_slices(\
                (x[:train_pos].astype(np.float32),
                 y[:train_pos].astype(np.float32),
                 y_mask[:train_pos].astype(np.float32)))
    dataset = dataset.batch(hparams.batch_size, drop_remainder=True)

    dataset_val = tf.data.Dataset.from_tensor_slices(\
                (x[train_pos:val_pos].astype(np.float32),
                 y[train_pos:val_pos].astype(np.float32),
                 y_mask[train_pos:val_pos].astype(np.float32)))
    dataset_val = dataset_val.batch(hparams.batch_size, drop_remainder=True)

    dataset_test = tf.data.Dataset.from_tensor_slices(\
                (x[val_pos:].astype(np.float32),
                 y[val_pos:].astype(np.float32),
                 y_mask[val_pos:].astype(np.float32)))
    dataset_test = dataset_val.batch(hparams.batch_size, drop_remainder=True)

    return dataset, dataset_val, dataset_test