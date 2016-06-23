import numpy as np
import theano
import theano.tensor as T

import constants

from .base_input_part import InputPart

class ChordShiftInputPart( InputPart ):
    """
    An input part that builds a shifted chord representation
    """

    CHORD_WIDTH = 12
    PART_WIDTH = 12

    def generate(self, relative_position, cur_chord_root, cur_chord_type, **kwargs):
        """
        Generate a chord input for a given timestep.

        Parameters: 
            relative_position: A theano tensor (int32) of shape (n_parallel), giving the
                current relative position for this timestep
            cur_chord_root: A theano tensor (int32) of shape (n_parallel) giving the unshifted chord root
            cur_chord_type: A theano tensor (int32) of shape (n_parallel, CHORD_WIDTH), giving the unshifted chord
                type representation, parsed from the leadsheet

        Returns:
            piece: A theano tensor (float32) of shape (n_parallel, PART_WIDTH)
        """
        def _map_fn(pos, chord):
            # Now pos is scalar and chord is of shape (CHORD_WIDTH), so we can roll
            return T.roll(chord, (-pos)%12, 0)

        shifted_chords, _ = theano.map(_map_fn, sequences=[relative_position-cur_chord_root, cur_chord_type])

        # shifted_chords = theano.printing.Print("ChordShiftInputPart")(shifted_chords)
        # shifted_chords = T.opt.Assert()(shifted_chords, T.eq(shifted_chords.shape[1], self.PART_WIDTH))
        return shifted_chords