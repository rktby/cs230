import numpy as np
import pandas as pd
import tensorflow as tf

tf.enable_eager_execution()
tf.set_random_seed(230)
print(tf.__version__)

import sys
import os
sys.path.append('..')

import trainers.tf_eager_trainer as trainer
from data_loader.biogas import *

# Avoids tf printing to logs (takes up memory on sherlock!)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

### Define Model
def gru(units, name):
    # Use CuDNNGRU is GPU is available (provides a 3x speedup than GRU)
    if tf.test.is_gpu_available():
	    return tf.keras.layers.CuDNNGRU(units, 
	                                    return_sequences=True, 
	                                    return_state=True, 
	                                    recurrent_initializer='glorot_uniform',
	                                    name=name)
    else:
        return tf.keras.layers.GRU(units, 
                                   return_sequences=True, 
                                   return_state=True, 
                                   recurrent_initializer='glorot_uniform',
                                   name=name)


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units, name=None):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units, name=name + '_W1')
        self.W2 = tf.keras.layers.Dense(units, name=name + '_W2')
        self.V = tf.keras.layers.Dense(1, name=name + '_V')

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        context_vector = tf.expand_dims(context_vector, 1)

        return context_vector, attention_weights


class Encoder(tf.keras.Model):
    def __init__(self, output_dim, layers, units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.layer_s = layers
        self.units = units
        self.cells = [gru(self.units, 'encoder_gru_%i' % i) for i in range(layers)]
        
    def call(self, x, hidden):
        outputs, states = [], []
        output = x
        
        for cell in self.cells:
            output, state = cell(output, initial_state = hidden)
            outputs.append(output)
            states.append(state)

        return outputs, states
    
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))



class Decoder(tf.keras.Model):
    def __init__(self, output_dim, num_layers, neurons_unit, batch_size):
        super(Decoder, self).__init__()
        self.output_dim   = output_dim
        self.num_layers   = num_layers
        self.neurons_unit = neurons_unit
        self.batch_size   = batch_size
        
        self.cells = [gru(neurons_unit, 'decoder_gru_%i' % i) for i in range(num_layers)]
        self.attentions = [BahdanauAttention(neurons_unit, 'decoder_attn_%i' % i) for i in range(num_layers)]
        self.fc_out = tf.keras.layers.Dense(output_dim, activation='relu', name='decoder_affine_out')
                
    def call(self, x, dec_states, enc_outputs, mask):
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_output = x
        states = []
        for layer, cell in enumerate(self.cells):
            context_vector, _ = self.attentions[layer](enc_outputs[layer], dec_states[layer])
        
            # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
            dec_output = tf.concat([context_vector, dec_output], axis=-1)

            # passing the concatenated vector to the GRU
            dec_output, dec_state = self.cells[layer](dec_output, initial_state=dec_states[layer])
            dec_states.append(dec_state)
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc_out(dec_output) * mask
        
        return x, dec_states
        
    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.units))


class EncoderDecoder(tf.keras.Model):
    def __init__(self, output_dim, num_layers, neurons_unit, batch_size):
        super(EncoderDecoder, self).__init__()
        self.output_dim   = output_dim
        self.num_layers   = num_layers
        self.neurons_unit = neurons_unit
        self.batch_size   = batch_size

        self.encoder = Encoder(output_dim, num_layers, neurons_unit, batch_size)
        self.decoder = Decoder(output_dim, num_layers, neurons_unit, batch_size)
    
    def call(self, inp, mask):
        hidden = self.encoder.initialize_hidden_state()
        enc_output, enc_hidden = self.encoder(inp, hidden)

        dec_hidden = enc_hidden

        dec_input = inp[:,0,:]
        output_dim = self.output_dim
        output_channels = int(int(inp.shape[-1]) / output_dim)
        for i in range(output_channels-1, inp.shape[-1], output_channels):
            dec_input = tf.concat((dec_input, inp[:,-1,i:i+1]), axis=1)
        
        dec_input = tf.expand_dims(dec_input, 1)

        for t in range(0, inp.shape[1]):
            # passing enc_output to the decoder
            prediction, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output, mask[:,t:t+1])

            # Concatenate with prediction from previous time step
            dec_input = tf.concat((tf.expand_dims(inp[:,t,:],1), prediction), axis=2)
            
            if t == 0:
                predictions = prediction
            else:
                predictions = tf.concat([predictions, prediction], axis=1)
        
        return predictions


def run_training(hparams, search_params, dataset, dataset_val):

	model = EncoderDecoder(hparams.output_channels, hparams.num_layers, hparams.neurons_unit, hparams.batch_size)
	optimizer = tf.train.AdamOptimizer(learning_rate = hparams.learning_rate)

	# Start training run
	train_stats = trainer.train_model(model, optimizer, dataset, hparams,
	                                  epochs=10, verbose=False)

	if not train_stats.empty:

		output_dev = trainer.validate_model(model, optimizer, get_fields, dataset_val, hparams, search_params)

	return output_dev, train_stats

# Hyperparameters being run
get_fields = sys.argv[1]
num_layers = int(sys.argv[2])

print('Running hparam search for sensors: {} with {} layers'.format(get_fields, num_layers))
# output path
out_dir = '~/CS230/output'
# out_dir = '/Users/josebolorinos/Google Drive/Coursework Stuff/CS230/CS230 Final Project/Results'
# Load dataset
data_path = '~/CS230/data/sensor_data.csv'
# data_path = '/Users/josebolorinos/Google Drive/Coursework Stuff/CS230/CS230 Final Project/Data/sensor_data.csv'

# List of names of hparams being searched for
search_params = ['input_dim','num_layers','neurons_unit','learning_rate','lambd']
top_search_params = ['num_layers']
hparams = \
	tf.contrib.training.HParams(
        batch_size=200,
        in_seq_len=24,
        input_dim= 1,
        input_channels = len(get_fields.split()),
        out_seq_len=24,
        output_dim=1,
        output_channels=len(get_fields.split()),
        num_layers=num_layers,
        neurons_unit=16,
        lr_decay = 0.99,
        train_split = 0.8,
        val_split = 0.1,
        test_split = 0.1,
        norm_epsilon = 1e-12,
        learning_rate = 10 ** -5,
        lambd = 10 ** -5,
        datagen = 'biogas',
        logs_path = '/tmp/tensorflow_logs'
    )

dataset, dataset_val, dataset_test = load_data(data_path, hparams, mode = get_fields, normalise = 'local_max', shuffle = True)

input_dims_s = range(2,6)
neurons_unit_s = [32,64,128]
learning_rate_power_s = [el*0.5 for el in range(-9,0,1)]
lambda_power_s = list(range(-4,4))

output_dev_all, train_stats_all = [],[]
# Run hparam search for learning rate power and regularization power
for input_dims in input_dims_s:
	for neurons_unit in neurons_unit_s:
		for learning_rate_power in learning_rate_power_s:
			for lambda_power in lambda_power_s:

				hparams.set_hparam('input_dim', input_dims)
				hparams.set_hparam('neurons_unit', neurons_unit)
				hparams.set_hparam('learning_rate', 10 ** learning_rate_power)
				hparams.set_hparam('lambd', 10 ** lambda_power)
				output_dev, train_stats = run_training(hparams, search_params, dataset, dataset_val)
				output_dev_all.append(output_dev)
				train_stats_all.append(train_stats)

# Output all results (keeping track of hparams)
hparams = hparams.values()
sensors = get_fields.split()
search_params_values = sensors + [str(hparams[search_param]) for search_param in top_search_params]
train_stats_all = pd.concat(train_stats_all, axis = 0)
output_dev_all = pd.concat(output_dev_all, axis = 0)
dev_op_fname = 'dev' + '_' + '_'.join(search_params_values) + '.csv'
dev_op_path = os.path.join(out_dir, dev_op_fname)
train_stats_op_fname = 'train' + '_' + '_'.join(search_params_values) + '.csv'
train_stats_op_path = os.path.join(out_dir, train_stats_op_fname)

train_stats_all.to_csv(train_stats_op_path, index = False, encoding = 'utf-8')
output_dev_all.to_csv(dev_op_path, index = False, encoding = 'utf-8')




