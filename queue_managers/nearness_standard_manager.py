from .queue_base import QueueManager
import theano
import theano.tensor as T
import numpy as np
from .standard_manager import StandardQueueManager

class NearnessStandardQueueManager( StandardQueueManager ):
    """
    A standard queue manager, using a configurable set of functions, with an exponential different loss
    """

    def __init__(self, feature_size, penalty_shock, penalty_base, falloff_rate, vector_activation_fun=T.nnet.sigmoid, loss_fun=(lambda x:x)):
        super().__init__(feature_size, vector_activation_fun, loss_fun)
        self._penalty_shock = penalty_shock
        self._penalty_base = penalty_base
        self._falloff_rate = falloff_rate

    def get_loss(self, raw_feature_strengths, raw_feature_vects, extra_info=False):
        raw_losses = self._loss_fun(raw_feature_strengths)
        raw_sum = T.sum(raw_losses)

        n_parallel, n_timestep = raw_feature_strengths.shape

        falloff_arr = np.array(self._falloff_rate, np.float32) ** T.cast(T.arange(n_timestep), 'float32')
        falloff_mat = T.shape_padright(falloff_arr) / T.shape_padleft(falloff_arr)
        falloff_scaling = T.switch(T.ge(falloff_mat,1), 0, falloff_mat)/self._falloff_rate
        # falloff_scaling is of shape (n_timestep, n_timestep) with 0 along diagonal, and jump to 1 falling off along dimension 1
        # now we want to multiply through on both dimensions
        first_multiply = T.dot(raw_feature_strengths, falloff_scaling) # shape (n_parallel, n_timestep)
        second_multiply = raw_feature_strengths * first_multiply
        unscaled_falloff_penalty = T.sum(second_multiply)

        full_loss = self._penalty_base * raw_sum + self._penalty_shock * unscaled_falloff_penalty

        if extra_info:
            return full_loss, {"raw_loss_sum":raw_sum}
        else:
            return full_loss
