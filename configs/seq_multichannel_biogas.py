import tensorflow as tf

def hparams():
    return tf.contrib.training.HParams(
        batch_size=100,
        in_seq_len=24,
        input_dim=5,
        out_seq_len=24,
        output_dim=1,
        num_layers=1,
        neurons_unit=32,
        learning_rate = 0.005,
        lr_decay = 0.99,
        lambd = 1e-6,
        train_split = 0.8,
        val_split = 0.1,
        test_split = 0.1,
        datagen = 'prodn',
        logs_path = '/tmp/tensorflow_logs')