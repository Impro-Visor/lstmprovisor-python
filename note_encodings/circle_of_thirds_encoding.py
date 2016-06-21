import numpy as np
import theano
import theano.tensor as T
import random

from .base_encoding import Encoding

import constants
import leadsheet
import math

class CircleOfThirdsEncoding( Encoding ):
    """
    [ rest sustain play ]
    [ () () () () ]
    [ () () () ]
    [ octave0 ... ]
    """

    STARTING_POSITION = 0
    WINDOW_SIZE = 12

    def __init__(self, octave_start, num_octaves):
        self.octave_start = octave_start
        self.num_octaves = num_octaves
        self.ENCODING_WIDTH = 3 + 4 + 3 + num_octaves
        self.RAW_ENCODING_WIDTH = self.ENCODING_WIDTH

    def encode_melody_and_position(self, melody, chords):
        encoded_form = []

        for note, dur in melody:
            if note is None:
                for _ in range(dur):
                    encoded_form.append([1] + [0]*(self.ENCODING_WIDTH-1))
            else:
                pitchclass = note % 12
                octave = (note - self.octave_start)//12

                first_circle = [(1 if ((pitchclass-i)%4 == 0) else 0) for i in range(4)]
                second_circle = [(1 if ((pitchclass-i)%3 == 0) else 0) for i in range(3)]
                octave_enc = [1 if (i==octave) else 0 for i in range(self.num_octaves)]

                enc_timestep = [0,0,1] + first_circle + second_circle + octave_enc

                encoded_form.append(enc_timestep)

                for _ in range(dur-1):
                    encoded_form.append([0, 1] + [0]*(self.ENCODING_WIDTH-2))

        encoded_form = np.array(encoded_form, np.float32)
        position = np.zeros([encoded_form.shape[0]])
        return encoded_form, position

    def decode_to_probs(self, activations, relative_position, low_bound, high_bound):
        assert (low_bound%12==0) and (high_bound-low_bound == self.num_octaves*12), "Circle of thirds must evenly divide into octaves"
        squashed = T.reshape(activations, (-1,self.RAW_ENCODING_WIDTH))

        rsp = T.nnet.softmax(squashed[:,:3])
        c1 = T.nnet.softmax(squashed[:,3:7])
        c2 = T.nnet.softmax(squashed[:,7:10])
        octave_choice = T.nnet.softmax(squashed[:,10:])
        octave_notes = T.tile(c1,(1,3)) * T.tile(c2,(1,4))
        full_notes = T.reshape(T.shape_padright(octave_choice) * T.shape_padaxis(octave_notes, 1), (-1,12*self.num_octaves))
        full_probs = T.concatenate([rsp[:,:2], T.shape_padright(rsp[:,2])*full_notes], 1)

        newshape = T.concatenate([activations.shape[:-1],[2+high_bound-low_bound]],0)
        fixed = T.reshape(full_probs, newshape, ndim=activations.ndim)
        return fixed

    def note_to_encoding(self, chosen_note, relative_position, low_bound, high_bound):
        assert chosen_note.ndim == 1
        n_batch = chosen_note.shape[0]

        dont_play_version = T.switch( T.shape_padright(T.eq(chosen_note, 0)),
                                        T.tile(np.array([[1,0] + [0]*(self.ENCODING_WIDTH-2)], dtype=np.float32), (n_batch, 1)),
                                        T.tile(np.array([[0,1] + [0]*(self.ENCODING_WIDTH-2)], dtype=np.float32), (n_batch, 1)))

        rcp = T.tile(np.array([0,0,1],dtype=np.float32), (n_batch, 1))
        circle_1 = T.eye(4)[(chosen_note-2)%4]
        circle_2 = T.eye(3)[(chosen_note-2)%3]
        octave = T.eye(self.num_octaves)[(chosen_note-2+low_bound-self.octave_start)//12]

        play_version = T.concatenate([rcp, circle_1, circle_2, octave], 1)

        encoded_form = T.switch( T.shape_padright(T.lt(chosen_note, 2)), dont_play_version, play_version )
        return encoded_form

    def get_new_relative_position(self, last_chosen_note, last_rel_pos, last_out, low_bound, high_bound, **cur_kwargs):
        return T.zeros_like(last_chosen_note)

    def initial_encoded_form(self):
        return np.array([1]+[0]*(self.ENCODING_WIDTH-1), np.float32)
