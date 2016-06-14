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
    LOW_BOUND = 48
    HIGH_BOUND = 84

    WINDOW_RADIUS = 12
    WINDOW_SIZE = WINDOW_RADIUS*2+1

    STARTING_POSITION = 72

    ENCODING_WIDTH = 1 + 1 + WINDOW_SIZE
    RAW_ENCODING_WIDTH = ENCODING_WIDTH

    def encode_melody(self, melody):

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

    def convert_activations(self, activations):
        squashed = T.reshape(activations, (-1,self.ENCODING_WIDTH))
        probs = T.nnet.softmax(squashed)
        return T.reshape(probs, activations.shape)

    def sample_output(self, srng, relative_position, encoded_probs):
        n_batch = relative_position.shape[0]
        # We want to create a mask that selects only those entries which are OK to transition to
        jump_posns = (T.shape_padright(relative_position) + np.expand_dims(np.arange(-self.WINDOW_RADIUS, self.WINDOW_RADIUS+1), 0))
        jump_mask = (jump_posns >= self.LOW_BOUND) * (jump_posns <= self.HIGH_BOUND)
        mask = T.concatenate([
                T.ones((n_batch, 2)),
                jump_mask
            ], 1)

        # Now we normalize it back to a regular probability
        masked_output_probs = mask * encoded_probs
        fixed_output_probs = masked_output_probs / T.sum(masked_output_probs, 1, keepdims=True)

        cum_probs = T.extra_ops.cumsum(fixed_output_probs, 1)
        # cum_probs = theano.printing.Print("Cumulative probs")(cum_probs)

        sampler = srng.uniform((n_batch,1))

        indicator = T.switch(cum_probs > sampler, cum_probs, 2)
        argmin = T.cast(T.argmin(indicator, 1), 'int32')
        sampled_output = T.extra_ops.to_one_hot(argmin, self.ENCODING_WIDTH)
        # Note: As long as the probabilities add up to 1, this works perfectly.
        # Due to numerical precision, it is possible that the cumulative sum is slightly <1
        # If this is the case, and sampler is too high, all values will become 2 in the
        # indicator and it should pick index [0], which is rest.

        # If we chose 0 or 1, we aren't shifting, so don't adjust position. If we chose
        # something else, shift as needed
        new_positions = relative_position + T.switch(argmin<2, 0, (argmin-2)-self.WINDOW_RADIUS)

        return sampled_output, new_positions

    def compute_loss(self, correct_encoded_form, encoded_probs, extra_info=False):
        loglikelihoods = T.log( encoded_probs + constants.EPSILON )*correct_encoded_form
        full_loss = T.neg(T.sum(loglikelihoods))

        if extra_info:
            n_batch, n_time, _ = encoded_probs.shape
            loss_per_timestep = full_loss/T.cast(n_batch*n_time, theano.config.floatX)
            accuracy_per_timestep = T.exp(-loss_per_timestep)

            loss_per_batch = full_loss/T.cast(n_batch, theano.config.floatX)
            accuracy_per_batch = T.exp(-loss_per_batch)

            num_jumps = T.sum(correct_encoded_form[:,:,2:])
            loss_per_jump = full_loss/T.cast(num_jumps, theano.config.floatX)
            accuracy_per_jump = T.exp(-loss_per_jump)

            return full_loss, {
                "loss_per_timestep":loss_per_timestep,
                "accuracy_per_timestep":accuracy_per_timestep,
                "loss_per_batch":loss_per_batch,
                "accuracy_per_batch":accuracy_per_batch,
                "loss_per_jump":loss_per_jump,
                "accuracy_per_jump":accuracy_per_jump
            }
        else:
            return full_loss

    def initial_encoded_form(self):
        return np.array([1]+[0]+[0]*self.WINDOW_SIZE, np.float32)

    def decode_melody(self, encoded_form, relative_positions):
        melody = []

        for out, pos in zip(encoded_form.tolist(), relative_positions.tolist()):
            should_rest = out[0]
            should_cont = out[1]
            should_jump = out[2:]

            if should_cont:
                # Continue a note
                if len(melody) == 0:
                    print("ERROR: Can't continue from nothing! Inserting rest")
                    melody.append([None, 0])
                melody[-1][1] += 1
            elif should_rest:
                # Rest
                if len(melody)>0 and melody[-1][0] is None:
                    # More rest
                    melody[-1][1] += 1
                else:
                    melody.append([None, 1])
            else:
                # We have a jump!
                jump_amt = should_jump.index(1) - self.WINDOW_RADIUS
                melody.append([pos+jump_amt, 1])

        return [tuple(x) for x in melody]