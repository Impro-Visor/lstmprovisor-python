import numpy as np
import theano
import theano.tensor as T

import constants

from .base_input_part import InputPart

class BeatInputPart( InputPart ):
    """
    An input part that builds a beat
    """

    BEAT_PERIODS = np.array([x//constants.RESOLUTION_SCALAR for x in [
                        constants.WHOLE,
                        constants.HALF,
                        constants.QUARTER,
                        constants.EIGHTH,
                        constants.SIXTEENTH,
                        constants.HALF_TRIPLET,
                        constants.QUARTER_TRIPLET,
                        constants.EIGHTH_TRIPLET,
                        constants.SIXTEENTH_TRIPLET,
                    ]], np.int32)
    PART_WIDTH = len(BEAT_PERIODS)

    def generate(self, timestep, **kwargs):
        """
        Generate a beat input for a given timestep.

        Parameters: 
            timestep: A theano int of shape (n_parallel). The current timestep to generate the beat input for.

        Returns:
            piece: A theano tensor (float32) of shape (n_parallel, PART_WIDTH)
        """

        result = T.eq(T.shape_padright(timestep) % np.expand_dims(self.BEAT_PERIODS,0), 0)

        # result = theano.printing.Print("BeatInputPart")(result)
        # result = T.opt.Assert()(result, T.eq(result.shape[1], self.PART_WIDTH))
        return result
