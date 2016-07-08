from .queue_base import QueueManager
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np

class QueuelessStandardQueueManager( QueueManager ):
    """
    A variational-autoencoder-based manager which does not use the queue, with a configurable loss
    """

    def __init__(self, feature_size, period=None, vector_activation_fun=T.nnet.sigmoid):
        """
        Initialize the manager.

        Parameters:
            feature_size: The width of a feature
            period: Period for queue activations
            vector_activation_fun: The activation function to apply to the vectors. Will be applied
                to a tensor of shape (n_parallel, feature_size) and should return a tensor of the
                same shape
        """
        self._feature_size = feature_size
        self._vector_activation_fun = vector_activation_fun
        self._period = period

    @property
    def activation_width(self):
        return self.feature_size

    @property
    def feature_size(self):
        return self._feature_size

    def get_strengths_and_vects(self, input_activations):
        n_batch, n_time, _ = input_activations.shape
        pre_vects = input_activations

        strengths = T.zeros((n_batch, n_time))
        if self._period is None:
            strengths = T.set_subtensor(strengths[:,-1],1)
        else:
            strengths = T.set_subtensor(strengths[:,self._period-1::self._period],1)

        flat_pre_vects = T.reshape(pre_vects,(-1,self.feature_size))
        flat_vects = self._vector_activation_fun( flat_pre_vects )
        vects = T.reshape(flat_vects, pre_vects.shape)

        return strengths, vects

    def get_loss(self, raw_feature_strengths, raw_feature_vects, extra_info=False):
        loss = T.as_tensor_variable(np.float32(0.0))
        if extra_info:
            return loss, {}
        else:
            return loss
