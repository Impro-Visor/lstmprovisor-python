import numpy as np
import theano
import theano.tensor as T
import random

from .base_encoding import Encoding

import constants
import leadsheet
import math

class RhythmOnlyEncoding( Encoding ):
    """
    An encoding that only encodes rhythm, of the form

    [

        rest,                   \
        continue,                } (a softmax set of excl probs) 
        articulate,             /

    ]
    """

    ENCODING_WIDTH = 3
    WINDOW_SIZE = 12
    RAW_ENCODING_WIDTH = 3

    def encode_melody_and_position(self, melody, chords):

        time = 0
        positions = []
        encoded_form = []

        for note, dur in melody:
            root, ctype = chords[time]
            if note is None:
                encoded_form.append([1,0,0])
            else:
                encoded_form.append([0,0,1])

            for _ in range(dur-1):
                encoded_form.append([1,0,0] if note is None else [0,1,0])
            time += dur

        positions = [root for root,ctype in chords]

        return np.array(encoded_form, np.float32), np.array(positions, np.int32)

    def decode_to_probs(self, activations, relative_position, low_bound, high_bound):
        squashed = T.reshape(activations, (-1,self.RAW_ENCODING_WIDTH))
        n_parallel = squashed.shape[0]
        probs = T.nnet.softmax(squashed)

        abs_probs = probs[:,:2]
        artic_prob = probs[:,2:]
        repeated_artic_probs = T.tile(artic_prob, (1,high_bound-low_bound))

        full_probs = T.concatenate([abs_probs,repeated_artic_probs],1)

        newshape = T.concatenate([activations.shape[:-1],[2+high_bound-low_bound]],0)
        fixed = T.reshape(full_probs, newshape, ndim=activations.ndim)
        return fixed

    def note_to_encoding(self, chosen_note, relative_position, low_bound, high_bound):
        """
        Convert a chosen note back into an encoded form

        Parameters:
            relative_position: A theano tensor of shape (...) giving the current relative position
            chosen_note: A theano tensor of shape (...) giving an index into encoded_probs

        Returns:
            sampled_output: A theano tensor (float32) of shape (..., ENCODING_WIDTH) that is
                sampled from encoded_probs. Should have the same representation as encoded_form from
                encode_melody
        """
        new_idx = T.switch(chosen_note<2, chosen_note, 2)
        sampled_output = T.extra_ops.to_one_hot(new_idx, self.ENCODING_WIDTH)
        return sampled_output

    def get_new_relative_position(self, last_chosen_note, last_rel_pos, last_out, low_bound, high_bound, cur_chord_root, **cur_kwargs):
        return cur_chord_root

    def initial_encoded_form(self):
        return np.array([1,0,0], np.float32)
