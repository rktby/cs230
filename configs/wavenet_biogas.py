import tensorflow as tf

def hparams():
    return tf.contrib.training.HParams(
        # Batch Data
        batch_size=800,
        in_seq_len=258,
        input_dim=1,
        input_channels=1,
        out_seq_len=24,
        output_dim=1,
        output_channels=1,

        # Learning Parameters
        learning_rate = 10 ** -2,
        lr_decay = 0.995,
        lambd = 1e-10,
        train_split = 0.8,
        val_split = 0.1,
        test_split = 0.1,
        norm_epsilon = 1e-12,
        datagen = 'biogas',
        logs_path = '/tmp/tensorflow_logs',
    
        # Model Parameters
        filter_width = 2,
        sample_rate = 24,
        dilations = [1, 2, 4, 8, 16, 32, 64],
        residual_channels = 16,
        dilation_channels = 16,
        quantization_channels = 100,
        skip_channels = 32,
        use_biases = True,
        scalar_input = False,
        initial_filter_width = 2
        )