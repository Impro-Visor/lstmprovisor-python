import numpy as np
import theano
import theano.tensor as T

import constants

from .base_input_part import InputPart

class PassthroughInputPart( InputPart ):
    """
    An input part that passes through one of its parameters unchanged
    """

    def __init__(self, keyword, width):
        """
        Initialize the input part to passthrough the input given by keyword
        """
        self.keyword = keyword
        self.PART_WIDTH = width

    def generate(self, **kwargs):
        """
        Generate a beat input for a given timestep.

        Parameters: 
            kwargs[keyword]: A theano tensor (float32) of shape (n_parallel, PART_WIDTH).
                This method will extract this keyword argument and return it unchanged

        Returns:
            piece: A theano tensor (float32) of shape (n_parallel, PART_WIDTH)
        """
        result = kwargs[self.keyword]
        # result = theano.printing.Print("PassthroughInputPart")(result)
        # result = T.opt.Assert()(result, T.eq(result.shape[1], self.PART_WIDTH))
        return result
