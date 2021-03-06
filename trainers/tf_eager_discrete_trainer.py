import tensorflow as tf
import numpy as np
import h5py
import time, datetime, os

def train_model(model, optimizer, dataset, hparams, epochs = 10, verbose = True, dev_dataset=None):

    # Time training run
    training_run_start = time.time()
    if dev_dataset == None:
        stats = {'train_loss': [], 'train_accuracy': [], 'time': []}
    else:
        stats = {'train_loss': [], 'train_accuracy': [], 'time': [], 'dev_loss': [], 'dev_accuracy': []}

    for epoch in range(epochs):
        # Time epoch
        epoch_start = time.time()
        total_loss, total_accuracy = 0, 0

        # Run one epoch of training data
        for (batch, (inp, targ, mask, rescalar)) in enumerate(dataset):
            # Make predictions and calculate loss
            with tf.GradientTape() as tape:
                pred = model(inp, mask)
                loss, accuracy = loss_function(targ, pred, model.variables, lambd=hparams.lambd)

            # Update statistics
            total_loss     += (loss     / int(targ.shape[1]))
            total_accuracy += (accuracy / int(targ.shape[1]))

            # Update gradients
            variables = model.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))

        # Calculate dev loss
        if dev_dataset != None:
            dev_loss, dev_accuracy = 0, 0
            for (dev_batch, (inp, targ, mask, rescalar)) in enumerate(dev_dataset):
                # Make predictions and calculate loss
                pred = model(inp, mask)
                loss, accuracy = loss_function(targ, pred, model.variables, lambd=hparams.lambd)
                # Update statistics
                dev_loss     += (loss     / int(targ.shape[1]))
                dev_accuracy += (accuracy / int(targ.shape[1]))

        stats['train_loss'].append(total_loss / (batch+1))
        stats['train_accuracy'].append(total_accuracy / (batch+1))
        stats['time'].append(time.time() - epoch_start)
        if dev_dataset != None:
            stats['dev_loss'].append(dev_loss / (dev_batch+1))
            stats['dev_accuracy'].append(dev_accuracy / (dev_batch+1))

        if verbose:
            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                    100 * total_loss / (batch+1), 100 * total_accuracy / (batch+1)))
            print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - epoch_start))
        
        # Apply learning rate decay
        optimizer._lr *= hparams.lr_decay
    
    return total_loss / (batch+1), total_accuracy / (batch+1), time.time() - training_run_start, stats

def loss_function(real, preds, variables, lambd = 1e-4):
    # Calculate OLS loss
    mse_loss = tf.losses.mean_squared_error(labels=real, predictions=preds)
    
    # Add L2 Regularisation
    l2_loss = 0
    if lambd > 0:
        for var in variables:
            l2_loss += lambd * tf.reduce_sum(var ** 2)

    accuracy, loss = mse_loss, mse_loss + l2_loss
    return loss, accuracy

def save_model(model, optimizer, dataset, hparams, name, stats=None, path='../outputs/'):
    """
    Method for saving model, parameters, hyperparameters and outputs
    """
    path += name + '_' + datetime.datetime.now().strftime('%y-%m-%d-%H_%M_%S') + '/'
    os.mkdir(path)

    # Save model and parameters
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.save(file_prefix = path)

    # Save hyperparameters
    with open(path + 'hparams.json', 'w') as file:
        file.write(hparams.to_json())

    # Save predictions and actual data
    preds, acts, maxs = [], [], []
    for inp, target, mask, x_max in dataset:
        pred = model(inp, mask)
        preds.extend(np.array(pred))
        acts.extend(np.array(target))
        maxs.extend(np.array(x_max))

    with h5py.File(path + 'data.hdf5', 'w') as file:
        file.create_dataset('actual', data=np.array(acts))
        file.create_dataset('predicted', data=np.array(preds))
        file.create_dataset('x_max', data=np.array(x_max))
        if stats != None:
            for stat_name, stat in stats.items():
                file.create_dataset('stat_' + stat_name, data=np.array(stat))
                
def validate_model(model, optimizer, get_fields, dataset, hparams, search_params, output_csv = False, out_dir = None, dsn_suffix = ''):
    """
    Method for saving model outputs
    """

    hparams = hparams.values()
    sensors = get_fields.split()

    # Save predictions and actual data
    preds, acts, maxs = [], [], []
    for inp, target, mask, x_max in dataset:
        pred = model(inp, mask)
        preds.extend(np.array(pred))
        acts.extend(np.array(target))
        maxs.extend(np.array(x_max))
    
    preds = np.array(preds)*np.array(maxs)
    acts  = np.array(acts)*np.array(maxs)
    errors = np.abs(preds - acts)
    
    mses = np.mean(errors**2, axis = 0)
    maes = np.mean(errors, axis = 0)
    mapes = np.mean(errors/acts, axis = 0)
    rses = np.mean(errors**2, axis = 0)/np.var(acts, axis = 0)
    
    dfs = [pd.DataFrame(array, columns = sensors) for array in [mses, maes, mapes, rses]]
    output_df = pd.concat(dfs, axis = 0)
    output_df['h_ahead'] = output_df.index.values + 1
    output_df['metric'] = np.array(['mse']*24 + ['mae']*24 + ['mape']*24 + ['rse']*24)
    output_df = output_df.melt(id_vars = ['h_ahead','metric'])
    output_df.columns = ['h_ahead','metric','sensor','value']
    
    for search_param in search_params:
        output_df[search_param] = hparams[search_param]
    
    if output_csv:
        search_params_values = sensors + [str(hparams[search_param]) for search_param in search_params]
        op_fname = '_'.join(search_params_values) + dsn_suffix + '.csv'
        op_path = os.path.join(out_dir, op_fname)
        output_df.to_csv(op_path, index = False, encoding = 'utf-8')

    return output_df
    

