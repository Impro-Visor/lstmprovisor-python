
class InputPart( object ):
    """
    Base class for input parts
    """

    PART_WIDTH = 0

    def generate(self, **kwargs):
        """
        Generate the appropriate input.

        Parameters:
            **kwargs: Depending on the particular class, may take in different values.
                To allow flexibility, all subclasses should ignore any kwargs that they do not need,
                so this method can be called with all relevant parameters.

        Returns:
            part: A theano tensor (float32) of shape (n_parallel, PART_WIDTH), where n_parallel
                is either an explicit parameter or determined by shapes of input
        """
        raise NotImplementedError("generate not implemented")
