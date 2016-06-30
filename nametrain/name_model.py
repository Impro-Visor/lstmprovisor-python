import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy as np

import constants
import input_parts
from relshift_lstm import RelativeShiftLSTMStack
from queue_managers import QueueManager
from adam import Adam
from note_encodings import Encoding
import leadsheet

import itertools
import functools
from theano_lstm import LSTM, StackedCells, Layer
from util import *
import random

import pickle

CHARKEY = " !\"'(),-.01245679:?ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwxyz"

def name_model():

    LSTM_SIZE = 300
    layer1 = LSTM(len(CHARKEY), LSTM_SIZE, activation=T.tanh)
    layer2 = Layer(LSTM_SIZE, len(CHARKEY), activation=lambda x:x)
    params = layer1.params + [layer1.initial_hidden_state] + layer2.params

    ################# Train #################
    train_data = T.ftensor3()
    n_batch = train_data.shape[0]
    train_input = T.concatenate([T.zeros([n_batch,1,len(CHARKEY)]),train_data[:,:-1,:]],1)
    train_output = train_data

    def _scan_train(last_out, last_state):
        new_state = layer1.activate(last_out, last_state)
        layer_out = layer1.postprocess_activation(new_state)
        layer2_out = layer2.activate(layer_out)
        new_out = T.nnet.softmax(layer2_out)
        return new_out, new_state

    outputs_info = [None, initial_state(layer1, n_batch)]
    (scan_outputs, scan_states), _ = theano.scan(_scan_train, sequences=[train_input.dimshuffle([1,0,2])], outputs_info=outputs_info)

    flat_scan_outputs = scan_outputs.dimshuffle([1,0,2]).reshape([-1,len(CHARKEY)])
    flat_train_output = train_output.reshape([-1,len(CHARKEY)])
    crossentropy = T.nnet.categorical_crossentropy(flat_scan_outputs, flat_train_output)
    loss = T.sum(crossentropy)/T.cast(n_batch,'float32')

    adam_updates = Adam(loss, params)

    train_fn = theano.function([train_data],loss,updates=adam_updates)

    ################# Eval #################

    length = T.iscalar()
    srng = MRG_RandomStreams(np.random.randint(1, 1024))

    def _scan_gen(last_out, last_state):
        new_state = layer1.activate(last_out, last_state)
        layer_out = layer1.postprocess_activation(new_state)
        layer2_out = layer2.activate(layer_out)
        new_out = T.nnet.softmax(T.shape_padleft(layer2_out))
        sample = srng.multinomial(n=1,pvals=new_out)[0,:]
        sample = T.cast(sample,'float32')
        return sample, new_state

    initial_input = np.zeros([len(CHARKEY)], np.float32)
    outputs_info = [initial_input, layer1.initial_hidden_state]
    (scan_outputs, scan_states), updates = theano.scan(_scan_gen, n_steps=length, outputs_info=outputs_info)

    gen_fn = theano.function([length],scan_outputs,updates=updates)

    return layer1, layer2, train_fn, gen_fn

def train_name(dataset_file):
    with open(dataset_file,'r') as f:
        dataset = [x.strip() for x in f]
    maxlen = max(len(x) for x in dataset)
    dataset = [x+" "*(maxlen-len(x)) for x in dataset]

    layer1, layer2, train_fn, gen_fn = name_model()
    params = layer1.params + [layer1.initial_hidden_state] + layer2.params

    print("Starting train...")

    BATCH_SIZE = 20
    for iteration in range(10000):
        sample = [random.choice(dataset) for _ in range(BATCH_SIZE)]
        sample_encoded = np.zeros([BATCH_SIZE, maxlen, len(CHARKEY)], np.float32)
        for i,train_string in enumerate(sample):
            for j,c in enumerate(train_string):
                try:
                    idx = CHARKEY.index(c)
                except ValueError:
                    print("Couldn't find character <{}>, replacing with space".format(c))
                    idx = len(CHARKEY)-1
                sample_encoded[i,j,idx] = 1.0

        loss = train_fn(sample_encoded)
        if iteration % 100 == 0:
            print("Iter",iteration,"has loss",loss)
            for _ in range(10):
                generate_name(maxlen, layer1, layer2, train_fn, gen_fn)


    pickle.dump([p.get_value() for p in params], open("name_params.p", 'wb'))

def generate_name(length, layer1=None, layer2=None, train_fn=None, gen_fn=None):
    if layer1 is None:
        layer1, layer2, train_fn, gen_fn = name_model()
        params = layer1.params + [layer1.initial_hidden_state] + layer2.params
        loaded_params = pickle.load(open("name_params.p", 'rb'))
        for p,v in zip(params, loaded_params):
            p.set_value(v)

    scan_outputs = gen_fn(length)
    outval = []
    for output in scan_outputs:
        idx = np.nonzero(output)[0][0]
        outval.append(CHARKEY[idx])
    print(''.join(outval))