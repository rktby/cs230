import tensorflow as tf
import numpy as np
import pandas as pd
import time, datetime, os

def train_model(model, optimizer, dataset, hparams, epochs = 10, verbose = True):

    # Time training run
    training_run_start = time.time()
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

            # Leave loop if loss is nan and return none
            if np.isnan(loss.numpy()):
                return pd.DataFrame([])

            # Update statistics
            total_loss     += loss 
            total_accuracy += accuracy

            # Update gradients
            variables = model.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))


        stats['train_loss'].append(total_loss / (batch+1))
        stats['train_accuracy'].append(total_accuracy / (batch+1))
        stats['time'].append(time.time() - epoch_start)

        if verbose:
            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                  total_loss / (batch+1), total_accuracy / (batch+1)))
            print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - epoch_start))
        
        # Apply learning rate decay
        optimizer._lr *= hparams.lr_decay

    # Output pandas dataframe
    train_loss = [el.numpy() for el in stats['train_loss']]
    train_accuracy = [el.numpy() for el in stats['train_accuracy']]
    train_stats_df = pd.DataFrame({'train_loss': train_loss, 'train_accuracy': train_accuracy})
    
    return train_stats_df

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
    

