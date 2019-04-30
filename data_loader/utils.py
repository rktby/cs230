import numpy as np
from scipy.signal import convolve

def norm(dataset, normalise='global_max', norm_epsilon=1e-12):
    """
    Normalise dataset on scale [0..1]
    Arguments:
        dataset: dataset to mormalise
        normalise:
            -'global_max': Scale all observations and down by max across the dataset
            -'local_max':  Scale all observations and down by max row-wise
            -'global_max_min': Subtract min from all observations and scale down by (max-min) across the dataset
            -'local_max_min':  Subtract min from all observations and scale down by (max-min) row-wise
    """
    # Ensure dataset is at least 3D 
    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, 0)
    if len(dataset.shape) == 2:
        dataset = np.expand_dims(dataset, -1)
    
    # Set axis to perform normalise calculation on
    norm_axis = (0,1) if normalise.count('global') == 1 else 1
    
    # Set max normalisation factor
    if normalise.count('max') > 0:
        x_max = np.max(dataset, axis=norm_axis, keepdims=True) + norm_epsilon
        x_max = x_max * np.ones_like(dataset)
    else:
        x_max = np.ones_like(dataset)
        
    # Set min normalisation factor
    if normalise.count('min') > 0:
        x_min = np.min(dataset, axis=norm_axis, keepdims=True)
        x_min = x_min + np.zeros_like(dataset)
    else:
        x_min = np.zeros_like(dataset)
    
    # Normalise dataset
    datanorm = (dataset - x_min) / (x_max - x_min)
    
    return datanorm, x_max, x_min

def conv(dataset, length, start=0, stride=1, dim=1):
    """
    Split dataset into segments
    Arguments:
        dataset: dataset to split
        length:  length of each segment
        start:   index in dataset to start at (default = 0)
        step:    step size to take between segments (default = 1)
        dim:     dimensions to wrap dataset around into
    """
    assert length % dim == 0
    
    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, 0)
    if len(dataset.shape) == 2:
        dataset = np.expand_dims(dataset, -1)
    
    nx, ny, nz = dataset.shape
    ix = (ny - length + stride) % stride
    
    dataset = [dataset[i,j:j+length] for i in range(nx) for j in range(start, ny-ix-length+1, stride)]
    
    #sx, sy, sz = dataset.strides
    
    #ny = ny - length + step
    #nx = nx * (ny // step)
    #ny = length
    #sx = sz * step
    
    #np.lib.stride_tricks.as_strided(c[:,:-ix], shape=(nx, ny, nz), strides=(sx,sy,sz)).squeeze()
    
    # Reshape dataset
    dataset = np.array(dataset)
    dataset = dataset.reshape(-1,length,1)
    if dim > 1:
        dataset = np.reshape(dataset, (len(dataset), -1, dim))
    
    return dataset

def _conv_(dataset, length, start=0, step=1, dim=1):
    """
    Split dataset into segments
    Arguments:
        dataset: dataset to split
        length:  length of each segment
        start:   index in dataset to start at (default = 0)
        step:    step size to take between segments (default = 1)
        dim:     dimensions to wrap dataset around into
    """
    assert length % dim == 0
    
    if len(dataset.shape) == 1:
        dataset = np.expand_dims(dataset, 0)
    if len(dataset.shape) == 2:
        dataset = np.expand_dims(dataset, -1)
    
    eye = np.flip(np.eye(length),1)
    eye = np.expand_dims(eye, 0)
    
    # Convolution calculation
    dataset = convolve(dataset[:,start:], eye, method='direct')
    dataset = dataset[:,length-1:-length+1]
    dataset = dataset[:,::step] if step > 1 else dataset
    
    # Reshape dataset
    dataset = dataset.reshape(-1,length,1)
    if dim > 1:
        dataset = dataset.reshape(len(dataset), -1, dim)
    
    return dataset