import numpy as np
import theano
import theano.tensor as T
import random

from .base_encoding import Encoding

import constants
import leadsheet
import math

class AbsoluteSequentialEncoding( Encoding ):

    STARTING_POSITION = 0
    WINDOW_SIZE = 1

    def __init__(self, low_bound, high_bound):
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.ENCODING_WIDTH = high_bound-low_bound+2
        self.RAW_ENCODING_WIDTH = self.ENCODING_WIDTH

    def encode_melody_and_position(self, melody, chords):
        abs_encoded_idxs = Encoding.encode_absolute_melody(melody, self.low_bound, self.high_bound)
        encoded_form = np.eye(self.ENCODING_WIDTH)[abs_encoded_idxs]
        position = np.zeros([abs_encoded_idxs.shape[0]])
        return encoded_form, position

    def decode_to_probs(self, activations, relative_position, low_bound, high_bound):
        squashed = T.reshape(activations, (-1,self.RAW_ENCODING_WIDTH))
        probs = T.nnet.softmax(squashed)
        fixed = T.reshape(probs, activations.shape)
        return fixed

    def note_to_encoding(self, chosen_note, relative_position, low_bound, high_bound):
        encoded_form = T.extra_ops.to_one_hot(chosen_note, self.ENCODING_WIDTH)
        return encoded_form

    def get_new_relative_position(self, last_chosen_note, last_rel_pos, last_out, low_bound, high_bound, **cur_kwargs):
        return T.zeros_like(last_chosen_note)

    def initial_encoded_form(self):
        return np.array([1]+[0]*(self.ENCODING_WIDTH-1), np.float32)
