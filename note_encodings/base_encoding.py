
class Encoding( object ):
    """
    Base class for note encodings
    """
    ENCODING_WIDTH = 0
    RAW_ENCODING_WIDTH = 0
    STARTING_POSITION = 0

    def encode_melody(self, melody):
        """
        Encode a melody in the correct format

        Parameters:
            melody: A melody object, of the form [(note_or_none, dur), ... ],
                where note_or_none is either a MIDI note value or None if this is a rest,
                and dur is the duration, relative to constants.RESOLUTION_SCALAR.

        Returns:
            encoded_form: A numpy ndarray (float32) of shape (timestep, ENCODING_WIDTH) representing
                the encoded form of the melody
            relative_positions: A numpy ndarray (int32) of shape (timestep), where relative_positions[t]
                gives the note position that the given timestep encoding is relative to.
        """
        raise NotImplementedError("encode_melody not implemented")

    def convert_activations(self, activations):
        """
        Convert a set of activations to a probability form

        Parameters:
            activations: A theano tensor (float32) of shape (..., RAW_ENCODING_WIDTH) giving
                raw activations from a standard neural network layer

        Returns:
            encoded_probs: A theano tensor (float32) of shape (..., ENCODING_WIDTH) giving an
                appropriate probability distribution for the encoded output
        """
        raise NotImplementedError("convert_activations not implemented")

    def sample_output(self, srng, relative_position, encoded_probs):
        """
        Convert an encoded probability form into a specific encoded form and new position

        Parameters:
            srng: A theano RandomStreams random number generator
            relative_position: A theano tensor (int32) of shape (parallel_n) giving a position to
                interpret the probabilties relative to
            encoded_probs: A theano tensor (float32) of shape (parallel_n, ENCODING_WIDTH) giving an
                appropriate probability distribution for the encoded output

        Returns:
            sampled_output: A theano tensor (float32) of shape (parallel_n, ENCODING_WIDTH) that is
                sampled from encoded_probs. Should have the same representation as encoded_form from
                encode_melody
            next_position: A theano tensor (int32) of shape (parallel_n) giving a position for the next
                timestep, corresponding to whatever output was sampled above
        """
        raise NotImplementedError("sample_output not implemented")

    def compute_loss(self, correct_encoded_form, encoded_probs, extra_info=False):
        """
        Compute the loss for a distribution relative to the correct encodings

        Parameters:
            correct_encoded_form: A theano tensor (float32) of the encoded form (from encode_melody)
                of the correct output, of shape (batch, time, enc_data)
            encoded_probs: A theano tensor (float32) of the probability distribution (from
                convert_activations) to evaluate, of shape (batch, time, enc_data)
            extra_info: A boolean. If true, return a list of supplementary info values.

        Returns
            loss: A theano scalar (float32) of the loss value. Smaller should be better, representing a
                close match between the two.
                Ideally, should be a negative log likelihood (i.e. a cross-entropy) between the correct
                answers and the encoded probabilities
            extra_values: If extra_info==True, this is also returned. This is a list of theano tensors
                that give more information about the network performance
        """
        raise NotImplementedError("compute_loss not implemented")

    def initial_encoded_form(self):
        """
        Returns: A numpy ndarray (float32) of shape (ENCODING_WIDTH) for an initial encoding of
            the "previous note" when there is no previous data. Generally should be a representation
            of nothing, i.e. of a rest.
        """
        raise NotImplementedError("initial_encoded_form not implemented")


    def decode_melody(self, encoded_form, relative_positions):
        """
        Decode a melody in the correct format

        Parameters:
            encoded_form: A numpy ndarray (float32) of shape (timestep, ENCODING_WIDTH) representing
                the encoded form of the melody
            relative_positions: A numpy ndarray (int32) of shape (timestep), where relative_positions[t]
                gives the note position that the given timestep encoding is relative t

        Returns:o.
            melody: A melody object, of the form [(note_or_none, dur), ... ],
                where note_or_none is either a MIDI note value or None if this is a rest,
                and dur is the duration, relative to constants.RESOLUTION_SCALAR.
        """
        raise NotImplementedError("decode_melody not implemented")