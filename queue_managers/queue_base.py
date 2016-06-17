import theano
import theano.tensor as T
import numpy as np

class QueueManager( object ):
    """
    Manages the queue transformation
    """

    @property
    def activation_width(self):
        """
        The activation width of the queue manager, determining the dimensions of the
        input_activations
        """
        raise NotImplementedError("activation_width not implemented")

    @property
    def feature_size(self):
        """
        The feature width of the queue manager, determining the dimensions of the transformed output
        """
        raise NotImplementedError("feature_size not implemented")
    
    def get_strengths_and_vects(self, input_activations):
        """
        Prepare a set of input activations, returning the feature strengths and vectors.

        Parameters:
            input_activations: a theano tensor (float32) of shape (batch, timestep, activation_width)

        Returns:
            raw_feature_strengths: A theano tensor (float32) of shape (batch, timestep) giving the
                raw push strength for each timestep
            raw_feature_vects: A theano tensor (float32) of shape (batch, timestep, feature_size)
                giving the raw vector of the input at each timestep
        """
        raise NotImplementedError("get_strengths_and_vects not implemented")

    def get_loss(self, input_activations, raw_feature_strengths, raw_feature_vects):
        """
        Calculate the loss for the given vects and strengths

        Parameters:
            raw_feature_strengths: A theano tensor (float32) of shape (batch, timestep) giving the
                raw push strength for each timestep
            raw_feature_vects: A theano tensor (float32) of shape (batch, timestep, feature_size)
                giving the raw vector of the input at each timestep

        Returns:
            sparsity_loss: a theano scalar (float32) giving the sparsity loss
        """
        raise NotImplementedError("get_loss not implemented")

    def process(self, input_activations):
        """
        Process a set of input activations, returning the transformed output and sparsity loss.

        Parameters:
            input_activations: a theano tensor (float32) of shape (batch, timestep, activation_width)

        Returns:
            transformed_output: a theano tensor (float32) of shape (batch, timestep, feature_width)
            sparsity_loss: a theano scalar (float32) giving the sparsity loss
        """
        raw_feature_strengths, raw_feature_vects = self.get_strengths_and_vects(input_activations)
        sparsity_loss = self.get_loss(raw_feature_strengths, raw_feature_vects)
        transformed_output = fragmented_queue_process(feature_strengths, feature_vects)
        
        return transformed_output, sparsity_loss

def fragmented_queue_process(feature_strengths, feature_vects, return_strengths=False):
    """
    Process features according to a "fragmented queue", where each timestep
    gets a size-1 window onto a feature queue. Effectively,
        feature_strengths gives how much to push onto queue
        feature_vects gives what to push on
        pop weights are tied to feature_strengths
        output is a size-1 peek (without popping)

    Parameters:
        - feature_strengths: float32 tensor of shape (batch, push_timestep) in [0,1]
        - feature_vects: float32 tensor of shape (batch, push_timestep, feature_dim)

    Returns:
        - peek_vects: float32 tensor of shape (batch, timestep, feature_dim)
    """
    n_batch, n_time, n_feature = feature_vects.shape

    cum_sum_str = T.extra_ops.cumsum(feature_strengths, 1)

    # We will be working in (batch, timestep, push_timestep)
    # For each timestep, if we subtract out the sum of pushes before that timestep
    # and then cap to 0-1 we get the cumsums for just the features active in that
    # timestep
    timestep_adjustments = T.shape_padright(cum_sum_str - feature_strengths)
    push_time_cumsum = T.shape_padaxis(cum_sum_str, 1)
    relative_cumsum = push_time_cumsum - timestep_adjustments
    capped_cumsum = T.minimum(T.maximum(relative_cumsum, 0), 1)

    # Now we can recover the peek strengths by taking a diff
    shifted = T.concatenate([T.zeros((n_batch, n_time, 1)), capped_cumsum[:,:,:-1]],2)
    peek_strengths = capped_cumsum-shifted
    # Peek strengths is now (batch, timestep, push_timestep)

    result = T.batched_dot(peek_strengths, feature_vects)

    if return_strengths:
        return peek_strengths, result
    else:
        return result

def test_frag_queue():
    feature_strengths = T.fmatrix()
    feature_vects = T.ftensor3()
    peek_strengths, res = fragmented_queue_process(feature_strengths, feature_vects, True)
    grad_s, grad_v = theano.gradient.grad(T.sum(res[:,:,1]), [feature_strengths,feature_vects])

    fun = theano.function([feature_strengths, feature_vects], [peek_strengths, res, grad_s, grad_v], allow_input_downcast=True)

    mystrengths = np.array([[0.3,0.3,0.2,0.6,0.3,0.7,0.2,1], [0.3,0.3,0.2,0.6,0.3,0.7,0.2,1]], np.float32)
    myvects = np.tile(np.eye(8, dtype=np.float32), (2,1,1))
    mypeek, myres, mygs, mygv = fun(mystrengths, myvects)

    print(mypeek)
    print(myres)
    print(mygs)
    print(mygv)
    return mypeek, myres, mygs, mygv