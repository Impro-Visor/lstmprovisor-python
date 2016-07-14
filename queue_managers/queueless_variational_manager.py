from .queue_base import QueueManager
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
import constants

class QueuelessVariationalQueueManager( QueueManager ):
    """
    A variational-autoencoder-based manager which does not use the queue, with a configurable loss
    """

    def __init__(self, feature_size, period=None, variational_loss_scale=1):
        """
        Initialize the manager.

        Parameters:
            feature_size: The width of a feature
            period: Period for queue activations
            variational_loss_scale: Factor by which to scale variational loss
        """
        self._feature_size = feature_size
        self._period = period
        self._srng = MRG_RandomStreams(np.random.randint(0, 1024))
        self._variational_loss_scale = np.array(variational_loss_scale, np.float32)

    @property
    def activation_width(self):
        return self.feature_size*2

    @property
    def feature_size(self):
        return self._feature_size

    def helper_sample(self, input_activations):
        n_batch, n_time, _ = input_activations.shape
        means = input_activations[:,:,:self.feature_size]
        stdevs = abs(input_activations[:,:,self.feature_size:]) + constants.EPSILON
        wiggle = self._srng.normal(means.shape)

        vects = means + (stdevs * wiggle)

        strengths = T.zeros((n_batch, n_time))
        if self._period is None:
            strengths = T.set_subtensor(strengths[:,-1],1)
        else:
            strengths = T.set_subtensor(strengths[:,self._period-1::self._period],1)

        return strengths, vects, means, stdevs, {}

    def get_strengths_and_vects(self, input_activations):
        strengths, vects, means, stdevs, _ = self.helper_sample(input_activations)
        return strengths, vects

    def process(self, input_activations, extra_info=False):

        strengths, vects, means, stdevs, sample_info = self.helper_sample(input_activations)

        means_sq = means**2
        variance = stdevs**2
        loss_parts = 1 + T.log(variance) - means_sq - variance
        if self._period is None:
            loss_parts = loss_parts[:,-1]
        else:
            loss_parts = loss_parts[:,self._period-1::self._period]
        variational_loss = -0.5 * T.sum(loss_parts) * self._variational_loss_scale

        info = {"variational_loss":variational_loss}
        info.update(sample_info)
        if extra_info:
            return variational_loss, strengths, vects, info
        else:
            return variational_loss, strengths, vects
