import tensorflow as tf
import time

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