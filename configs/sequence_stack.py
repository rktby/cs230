import tensorflow as tf

def hparams():
    return tf.contrib.training.HParams(
        batch_size=100,
        in_seq_len=61,
        input_dim=5,
        input_channels=1,
        out_seq_len=61,
        output_dim=1,
        output_channels=1,
        num_layers=6,
        neurons_unit=32,
        learning_rate = 0.005,
        lr_decay = 0.99,
        lambd = 1e-6,
        train_split = 0.8,
        val_split = 0.1,
        test_split = 0.1,
        norm_epsilon = 1e-12,
        datagen = 'prodn',
        logs_path = '/tmp/tensorflow_logs')