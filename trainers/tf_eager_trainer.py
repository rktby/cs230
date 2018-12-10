import tensorflow as tf
import time

def train_model(model, optimizer, dataset, hparams, epochs = 10, verbose = True):

    # Time training run
    training_run_start = time.time()
    stats = []

    for epoch in range(epochs):
        # Time epoch
        epoch_start = time.time()
        total_loss, total_accuracy = 0, 0

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

        if verbose:
            print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
                                                100 * total_loss / (batch+1), 100 * total_accuracy / (batch+1)))
            print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - epoch_start))
        
        # Apply learning rate decay
        optimizer._lr *= hparams.lr_decay
        stats.append([total_loss / (batch+1), total_accuracy / (batch+1), time.time() - epoch_start])
    
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