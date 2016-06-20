import numpy as np
import theano
import theano.tensor as T
import random

from .base_encoding import Encoding

import constants
import leadsheet

def rotate(li, x):
    """
    Rotate list li by x spaces to the right, i.e.
        rotate([1,2,3,4],1) -> [4,1,2,3]
    """
    if len(li) == 0: return []
    return li[-x % len(li):] + li[:-x % len(li)]

class RelativeJumpEncoding( Encoding ):
    """
    An encoding based on relative jumps. Encoding format is a one-hot

    [

        rest,                   \
        continue,                } (a softmax set of excl probs) 
        play x WINDOW_SIZE,     /

    ]

    where WINDOW_SIZE gives the number of places to which we can jump.
    """

    WINDOW_RADIUS = 12
    WINDOW_SIZE = WINDOW_RADIUS*2+1

    STARTING_POSITION = 72

    ENCODING_WIDTH = 1 + 1 + WINDOW_SIZE
    RAW_ENCODING_WIDTH = ENCODING_WIDTH

    def encode_melody_and_position(self, melody, chords):

        positions = []
        encoded_form = []

        cur_pos = next((n for n,d in melody if n is not None), self.STARTING_POSITION) + random.randrange(-self.WINDOW_RADIUS, self.WINDOW_RADIUS+1)

        positions.append(cur_pos)

        for note, dur in melody:
            if note is None:
                delta = 0
            else:
                delta = note - cur_pos
                cur_pos = note
            if not (-self.WINDOW_RADIUS <= delta <= self.WINDOW_RADIUS):
                olddelta = delta
                if delta>0:
                    delta = delta % self.WINDOW_RADIUS
                else:
                    delta = -(-delta % self.WINDOW_RADIUS)
                # print("WARNING: Jump of size {} from {} to {} not allowed. Substituting jump of size {}".format(olddelta, note-olddelta, note, delta))

            rcp = ([1 if note is None else 0] +
                [0] +
                [(1 if i==delta and note is not None else 0) for i in range(-self.WINDOW_RADIUS, self.WINDOW_RADIUS+1)])

            encoded_form.append(rcp) # for this timestep
            positions.append(cur_pos) # for next timestep

            for _ in range(dur-1):
                rcp = [1 if note is None else 0] + [0 if note is None else 1] + [0]*self.WINDOW_SIZE
                encoded_form.append(rcp)
                positions.append(cur_pos)

        # Remove last position, since nothing is relative to it
        positions = positions[:-1]

        return np.array(encoded_form, np.float32), np.array(positions, np.int32)

    def decode_to_probs(self, activations, relative_position, low_bound, high_bound):
        squashed = T.reshape(activations, (-1,self.RAW_ENCODING_WIDTH))
        n_parallel = squashed.shape[0]
        probs = T.nnet.softmax(squashed)

        def _scan_fn(cprobs, cpos):
            # cprobs = theano.printing.Print("cprobs",['shape'])(cprobs)
            # cpos = theano.printing.Print("cpos",['shape'])(cpos)

            abs_probs = cprobs[:2]
            rel_probs = cprobs[2:]

            # abs_probs = theano.printing.Print("abs_probs",['shape'])(abs_probs)
            # rel_probs = theano.printing.Print("rel_probs",['shape'])(rel_probs)

            # Start index:
            #      *****[-----------------------------]
            #      [****{-|------]
            #
            #           [-----------------------------]
            #           ~~~~~{------|------]
            start_diff = low_bound - (cpos-self.WINDOW_RADIUS)
            startidx = T.maximum(0, start_diff)
            startpadding = T.maximum(0, -start_diff)
            # End index:
            #          [-----------------------------]
            #                              [******|**}---]
            #
            #           [-----------------------------]
            #                [******|******}~~~~~~~~~~~
            endidx = T.minimum(self.WINDOW_SIZE, high_bound - (cpos-self.WINDOW_RADIUS))
            endpadding = T.maximum(0, high_bound-(cpos+self.WINDOW_RADIUS+1))

            # start_diff = theano.printing.Print("start_diff",['shape','__str__'])(start_diff)
            # startidx = theano.printing.Print("startidx",['shape','__str__'])(startidx)
            # startpadding = theano.printing.Print("startpadding",['shape','__str__'])(startpadding)
            # endidx = theano.printing.Print("endidx",['shape','__str__'])(endidx)
            # endpadding = theano.printing.Print("endpadding",['shape','__str__'])(endpadding)

            cropped = rel_probs[startidx:endidx]
            normalize_sum = T.sum(cropped) + T.sum(abs_probs)
            normalize_sum = T.maximum(normalize_sum, constants.EPSILON)
            padded = T.concatenate([abs_probs/normalize_sum, T.zeros((startpadding,)), cropped/normalize_sum, T.zeros((endpadding,))], 0)
            # padded = theano.printing.Print("padded",['shape'])(padded)
            return padded

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
        new_idx = T.switch(chosen_note<2, chosen_note, chosen_note+low_bound-relative_position+self.WINDOW_RADIUS)
        new_idx = T.opt.Assert("new_idx should be less than {}".format(self.ENCODING_WIDTH))(new_idx, T.all(new_idx < self.ENCODING_WIDTH))
        sampled_output = T.extra_ops.to_one_hot(new_idx, self.ENCODING_WIDTH)
        return sampled_output

    def get_new_relative_position(self, last_chosen_note, last_rel_pos, last_out, low_bound, high_bound, **cur_kwargs):
        return T.switch(last_chosen_note<2, last_rel_pos, last_chosen_note+low_bound-2)

    def initial_encoded_form(self):
        return np.array([1]+[0]+[0]*self.WINDOW_SIZE, np.float32)
