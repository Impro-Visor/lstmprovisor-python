import theano
import theano.tensor as T
import numpy as np
import constants

class Encoding( object ):
    """
    Base class for note encodings
    """
    ENCODING_WIDTH = 0
    RAW_ENCODING_WIDTH = 0
    STARTING_POSITION = 0

    def encode_melody_and_position(self, melody, chords):
        """
        Encode a melody in the correct format

        Parameters:
            melody: A melody object, of the form [(note_or_none, dur), ... ],
                where note_or_none is either a MIDI note value or None if this is a rest,
                and dur is the duration, relative to constants.RESOLUTION_SCALAR.
            chords: A chord object, of the form [(root, typevec), ...],
                where root is a MIDI note value in 0-12, typevec is a boolean list of length 12

        Returns:
            encoded_form: A numpy ndarray (float32) of shape (timestep, ENCODING_WIDTH) representing
                the encoded form of the melody
            relative_positions: A numpy ndarray (int32) of shape (timestep), where relative_positions[t]
                gives the note position that the given timestep encoding is relative to.
        """
        raise NotImplementedError("encode_melody_and_position not implemented")

    def decode_to_probs(self, activations, relative_position, low_bound, high_bound):
        """
        Convert a set of activations to a probability form across notes.

        Parameters:
            activations: A theano tensor (float32) of shape (..., RAW_ENCODING_WIDTH) giving
                raw activations from a standard neural network layer
            relative_position: A theano tensor of shape (...) giving the current relative position
            low_bound: The MIDI index of the lowest note to return
            high_bound: The MIDI index of one past the highest note to return

        Returns:
            encoded_probs: A theano tensor (float32) of shape (..., 2+high_bound-low_bound) giving a
                probability distribution for chosen notes, where
                    [0]: rest
                    [1]: continue
                    [2+x]: play note (low_bound + x)
        """
        raise NotImplementedError("decode_to_probs not implemented")

    def note_to_encoding(self, chosen_note, relative_position, low_bound, high_bound):
        """
        Convert a chosen note back into an encoded form

        Parameters:
            relative_position: A theano tensor of shape (...) giving the current relative position
            chosen_note: A theano tensor of shape (...) giving an index into encoded_probs
            low_bound: The MIDI index of the lowest note to return
            high_bound: The MIDI index of one past the highest note to return

        Returns:
            sampled_output: A theano tensor (float32) of shape (..., ENCODING_WIDTH) that is
                sampled from encoded_probs. Should have the same representation as encoded_form from
                encode_melody
        """
        raise NotImplementedError("note_to_output not implemented")

    def get_new_relative_position(self, last_chosen_note, last_rel_pos, last_out, low_bound, high_bound, **cur_kwargs):
        """
        Get the new relative position for this timestep

        Parameters:
            last_chosen_note is a theano tensor of shape (n_batch) indexing into 2+high_bound-low_bound
            last_rel_pos is a theano tensor of shape (n_batch)
            last_out will be a theano tensor of shape (n_batch, output_size)
            cur_kwargs[k] is a theano tensor of shape (n_batch, ...), from kwargs
            low_bound: The MIDI index of the lowest note to return
            high_bound: The MIDI index of one past the highest note to return

        Returns:
            new_pos, a theano tensor of shape (n_batch), giving the new relative position
        """
        raise NotImplementedError("get_new_relative_position not implemented")


    def initial_encoded_form(self):
        """
        Returns: A numpy ndarray (float32) of shape (ENCODING_WIDTH) for an initial encoding of
            the "previous note" when there is no previous data. Generally should be a representation
            of nothing, i.e. of a rest.
        """
        raise NotImplementedError("initial_encoded_form not implemented")

    @staticmethod
    def encode_absolute_melody(melody, low_bound, high_bound):
        """
        Encode an absolute melody

        Parameters:
            melody: A melody object, of the form [(note_or_none, dur), ... ],
                where note_or_none is either a MIDI note value or None if this is a rest,
                and dur is the duration, relative to constants.RESOLUTION_SCALAR.
            low_bound: The MIDI index of the lowest note to return
            high_bound: The MIDI index of one past the highest note to return

        Returns
            A numpy matrix of shape (timestep) giving the int index (in 2+high_bound-low_bound) of the correct note
        """
        positions = []

        for note, dur in melody:
            positions.append(0 if note is None else (note-low_bound+2))

            for _ in range(dur-1):
                positions.append(0 if note is None else 1)

        return np.array(positions, np.int32)


    @staticmethod
    def decode_absolute_melody(positions, low_bound, high_bound):
        """
        Decode an absolute melody

        Parameters:
            A numpy matrix of shape (timestep) giving the int index (in 2+high_bound-low_bound) of the correct note
            low_bound: The MIDI index of the lowest note to return
            high_bound: The MIDI index of one past the highest note to return

        Returns
            melody: A melody object, of the form [(note_or_none, dur), ... ],
                where note_or_none is either a MIDI note value or None if this is a rest,
                and dur is the duration, relative to constants.RESOLUTION_SCALAR.
        """
        melody = []

        for out in positions.tolist():
            if out==1:
                # Continue a note
                if len(melody) == 0:
                    print("ERROR: Can't continue from nothing! Inserting rest")
                    melody.append([None, 0])
                melody[-1][1] += 1
            elif out==0:
                # Rest
                if len(melody)>0 and melody[-1][0] is None:
                    # More rest
                    melody[-1][1] += 1
                else:
                    melody.append([None, 1])
            else:
                note = out-2 + low_bound
                melody.append([note, 1])

        return [tuple(x) for x in melody]

    @staticmethod
    def sample_absolute_probs(srng, probs):
        """
        Sample from a probability distribution

        Parameters:
            srng: A RandomStreams instance
            probs: A matrix of probabilities of shape (n_batch, sample_from)

        Returns:
            Sampled output, an index in [0,sample_from) of shape (n_batch)
        """
        n_batch,sample_from = probs.shape

        cum_probs = T.extra_ops.cumsum(probs, 1)
        # cum_probs = theano.printing.Print("Cumulative probs")(cum_probs)

        sampler = T.shape_padright(srng.uniform((n_batch,)))

        indicator = T.switch(cum_probs > sampler, cum_probs, 2)
        argmin = T.cast(T.argmin(indicator, 1), 'int32')
        # Note: As long as the probabilities add up to 1, this works perfectly.
        # Due to numerical precision, it is possible that the cumulative sum is slightly <1
        # If this is the case, and sampler is too high, all values will become 2 in the
        # indicator and it should pick index [0], which is rest.

        return argmin

    @staticmethod
    def compute_loss(probs, absolute_melody, extra_info=False):
        """
        Compute loss between probs and an absolute melody

        Parameters:
            probs: A theano tensor of shape (batch, time, 2+high_bound-low_bound)
            absolute_melody: A tensor of shape (batch, time) with correct indices
            extra_info: If True, return extra info

        Returns
            A theano tensor loss value.
            Also, if extra_info is true, an additional info dict.
        """
        n_batch, n_time, prob_width = probs.shape
        correct_encoded_form = T.reshape(T.extra_ops.to_one_hot(T.flatten(absolute_melody), prob_width), probs.shape)
        loglikelihoods = T.log( probs + constants.EPSILON )*correct_encoded_form
        full_loss = T.neg(T.sum(loglikelihoods))

        if extra_info:
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
