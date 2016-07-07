from .queue_base import QueueManager
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np

class QueuelessVariationalQueueManager( QueueManager ):
    """
    A variational-autoencoder-based manager which does not use the queue, with a configurable loss
    """

    def __init__(self, feature_size, variational_loss_scale=1):
        """
        Initialize the manager.

        Parameters:
            feature_size: The width of a feature
            variational_loss_scale: Factor by which to scale variational loss
        """
        self._feature_size = feature_size
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
        means = input_activations[:,-1,:self.feature_size]
        stdevs = abs(input_activations[:,-1,self.feature_size:]) + constants.EPSILON
        wiggle = self._srng.normal(means.shape)

        vect = means + (stdevs * wiggle)

        strengths = T.zeros((n_batch, n_time))
        strengths = T.set_subtensor(strengths[:,-1],1)
        vects = T.zeros((n_batch, n_time, self.feature_size))
        vects = T.set_subtensor(vects[:,-1,:],vect)

        return strengths, vects, means, stdevs, {}

    def get_strengths_and_vects(self, input_activations):
        strengths, vects, means, stdevs, _ = self.helper_sample(input_activations)
        return strengths, vects

    def process(self, input_activations, extra_info=False):

        strengths, vects, means, stdevs, sample_info = self.helper_sample(input_activations)

        means_sq = means**2
        variance = stdevs**2
        variational_loss = -0.5 * T.sum(1 + T.log(variance) - means_sq - variance) * self._variational_loss_scale

        info = {"variational_loss":variational_loss}
        info.update(sample_info)
        if extra_info:
            return variational_loss, strengths, vects, info
        else:
            return variational_loss, strengths, vects
