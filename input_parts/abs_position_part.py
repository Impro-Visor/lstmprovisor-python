import numpy as np
import theano
import theano.tensor as T

import constants

from .base_input_part import InputPart

class PositionInputPart( InputPart ):
    """
    An input part that constructs a position
    """

    def __init__(self, low_bound, up_bound, num_divisions):
        """
        Build an input part with num_divisions ranging betweeen low_bound and up_bound.
        This part will activate each division depending on how close the relative_position
        is to it.
        """
        assert num_divisions >= 2, "Must have at least 2 divisions!"
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.PART_WIDTH = num_divisions

    def generate(self, relative_position, **kwargs):
        """
        Generate a position input for a given timestep.

        Parameters: 
            relative_position: A theano tensor (int32) of shape (n_parallel), giving the
                current relative position for this timestep

        Returns:
            piece: A theano tensor (float32) of shape (n_parallel, PART_WIDTH)
        """
        delta = (self.up_bound-self.low_bound) / (self.PART_WIDTH-1)
        indicator_pos = np.array([[i*delta + self.low_bound for i in range(self.PART_WIDTH)]], np.float32)

        # differences[i][j] is the difference between relative_position[i] and indicator_pos[j]
        differences = T.cast(T.shape_padright(relative_position),'float32') - indicator_pos

        # We want each indicator to activate at its division point, and fall off linearly around it,
        # capped from 0 to 1.
        activities = T.maximum(0, 1-abs(differences/delta))

        # activities = theano.printing.Print("PositionInputPart")(activities)
        # activities = T.opt.Assert()(activities, T.eq(activities.shape[1], self.PART_WIDTH))

        return activities
