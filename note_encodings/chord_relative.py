import numpy as np
import theano
import theano.tensor as T
import random

from .base_encoding import Encoding

import constants
import leadsheet
import math

class ChordRelativeEncoding( Encoding ):
    """
    An encoding based on the chord. Encoding format is a one-hot

    [

        rest,                   \
        continue,                } (a softmax set of excl probs) 
        play x 12,     /

    ]

    where play is relative to the chord root
    """

    ENCODING_WIDTH = 1 + 1 + 12
    RAW_ENCODING_WIDTH = ENCODING_WIDTH
    WINDOW_SIZE = 12

    def encode_melody_and_position(self, melody, chords):

        time = 0
        positions = []
        encoded_form = []

        for note, dur in melody:
            root, ctype = chords[time]
            if note is None:
                encoded_form.append([1]+[0]+[0]*self.WINDOW_SIZE)
            else:
                index = (note - root)%self.WINDOW_SIZE
                encoded_form.append([0]+[0]+[1 if i==index else 0 for i in range(self.WINDOW_SIZE)])

            for _ in range(dur-1):
                rcp = [1 if note is None else 0] + [0 if note is None else 1] + [0]*self.WINDOW_SIZE
                encoded_form.append(rcp)
            time += dur

        positions = [root for root,ctype in chords]

        return np.array(encoded_form, np.float32), np.array(positions, np.int32)

    def decode_to_probs(self, activations, relative_position, low_bound, high_bound):
        squashed = T.reshape(activations, (-1,self.RAW_ENCODING_WIDTH))
        n_parallel = squashed.shape[0]
        probs = T.nnet.softmax(squashed)


        def _scan_fn(cprobs, cpos):

            abs_probs = cprobs[:2]
            rel_probs = cprobs[2:]

            aligned = T.roll(rel_probs, (cpos-low_bound)%12)

            num_tile = int(math.ceil((high_bound-low_bound)/self.WINDOW_SIZE))

            tiled = T.tile(aligned, (num_tile,))[:(high_bound-low_bound)]

            full = T.concatenate([abs_probs, tiled], 0)
            return full

        # probs = theano.printing.Print("probs",['shape'])(probs)
        # relative_position = theano.printing.Print("relative_position",['shape'])(relative_position)
        from_scan, _ = theano.map(fn=_scan_fn, sequences=[probs, T.flatten(relative_position)])
        # from_scan = theano.printing.Print("from_scan",['shape'])(from_scan)
        newshape = T.concatenate([activations.shape[:-1],[2+high_bound-low_bound]],0)
        fixed = T.reshape(from_scan, newshape, ndim=activations.ndim)
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
        new_idx = T.switch(chosen_note<2, chosen_note, (chosen_note-2+low_bound-relative_position)%self.WINDOW_SIZE + 2)
        sampled_output = T.extra_ops.to_one_hot(new_idx, self.ENCODING_WIDTH)
        return sampled_output

    def get_new_relative_position(self, last_chosen_note, last_rel_pos, last_out, low_bound, high_bound, cur_chord_root, **cur_kwargs):
        return cur_chord_root

    def initial_encoded_form(self):
        return np.array([1]+[0]+[0]*self.WINDOW_SIZE, np.float32)
