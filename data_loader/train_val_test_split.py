import numpy as np
import tensorflow as tf


def split(hparams, dataset, mask, normalise='global_max', is_autoregressive=False):
    # Extract parameters defining dataset shape
    in_len, out_len, in_dim = hparams.in_seq_len, hparams.out_seq_len, hparams.input_dim
    end_pos = in_len * (in_dim - 1) + 1

    p_val, p_test = hparams.val_split, hparams.test_split
    p_train = 1 - p_val - p_test

    # Calculate normalisation factor
    if normalise in('local_max', 'local_max_min'):
        # Normalise the data line by line
        x_max = np.max(dataset[:,:in_len * in_dim], axis=1, keepdims=True) + hparams.norm_epsilon
        x_min = np.min(dataset[:,:in_len * in_dim], axis=1, keepdims=True)
    elif normalise in('global_max', 'global_max_min'):
        # Normalise the data based on global maximum
        x_max = np.max(dataset, axis=(0,1), keepdims=True)
        x_max = np.repeat(x_max, len(dataset), axis=0)
        x_min = np.min(dataset, axis=(0,1), keepdims=True)
        x_min = np.repeat(x_min, len(dataset), axis=0)
    else:
        # Do not normalise the data
        x_max = np.ones_like(dataset).max(axis=1, keepdims=True)
        x_min = np.zeros_like(dataset).max(axis=1, keepdims=True)

    if normalise in('local_max_min', 'global_max_min'):
        datanorm = dataset - x_min
        datanorm = datanorm / (x_max - x_min)
    else:
        datanorm = dataset / x_max

    # Create x dataset
    # x.shape: (n_obs, in_seq_len, input_dim)
    x = np.array([datanorm[:,pos:pos+in_len] for pos in range(0,end_pos,hparams.in_seq_len)])
    x = np.rollaxis(x, 0, 3)
    x = np.reshape(x,[x.shape[0], x.shape[1], -1])
    if is_autoregressive:
        x_mask = np.array([mask[:,pos:pos+in_len] for pos in range(0,end_pos,hparams.in_seq_len)])
        x_mask = np.rollaxis(x_mask, 0, 3)
        x_mask = np.reshape(x_mask,[x_mask.shape[0], x_mask.shape[1], -1])


    # Create y dataset
    y = datanorm[:,end_pos+in_len:end_pos+in_len+out_len, np.newaxis]
    y = np.reshape(y,[y.shape[0], y.shape[1], -1])
    y_mask = mask[:,end_pos+in_len:end_pos+in_len+out_len, np.newaxis]
    y_mask = np.reshape(y_mask,[y_mask.shape[0], y_mask.shape[1], -1])

    #####################
    # Build into datasets
    #####################
    
    # Determine data to add to dataset
    dataset = [x,y]
    print('Added x, y data')

    if is_autoregressive:
        dataset.append(x_mask)
        print('Added x_mask data')

    dataset.append(y_mask)
    print('Added y_mask data')

    dataset.append(x_max)
    print('Added x_max data')
    
    if normalise in('local_max_min', 'global_max_min'):
        dataset.append(x_min)
        print('Added x_min data')
    
    # Calculate split positions
    train_pos = int(x.shape[0] * p_train)
    val_pos   = int(x.shape[0] * (p_train + p_val))
    end_pos   = int(x.shape[0])
    
    def subsplit(dataset, start, finish):
        
        dataset = tuple(d[start:finish].astype(np.float32) for d in dataset)
        dataset = tf.data.Dataset.from_tensor_slices(dataset)

        return dataset.batch(hparams.batch_size, drop_remainder=True)

    dataset_train = subsplit(dataset, 0, train_pos)
    dataset_val   = subsplit(dataset, train_pos, val_pos)
    dataset_test  = subsplit(dataset, val_pos, end_pos)

    return dataset_train, dataset_val, dataset_test