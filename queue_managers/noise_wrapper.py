import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams
from .queue_base import QueueManager
from util import sliceMaker

class NoiseWrapper( QueueManager ):
    """
    Queue manager that wraps another queue manager and adds noise to it
    """
    def __init__(self, inner_manager, pre_std=None, post_std=None, pre_mask=sliceMaker[:]):
        """
        Initialize the manager.

        Parameters:
            pre_std: Standard deviation of noise to apply to input activations
            post_std: Standard deviation of noise to apply to output vector
            pre_mask: A slice that determines which part of input activations
                to apply noise to (i.e. activations[:,:,pre_mask])
        """
        self._srng = MRG_RandomStreams(np.random.randint(0, 1024))
        self._inner_manager = inner_manager
        self._pre_std = pre_std
        self._pre_mask = pre_mask
        self._post_std = post_std

    @property
    def activation_width(self):
        return self._inner_manager.activation_width

    @property
    def feature_size(self):
        return self._inner_manager.feature_size

    def get_strengths_and_vects(self, input_activations):
        if self._pre_std is not None:
            input_activations = T.inc_subtensor(input_activations[:,:,self._pre_mask], self._srng.normal(input_activations.shape, self._pre_std))
        strengths, vects = self._inner_manager.get_strengths_and_vects(input_activations)
        if self._post_std is not None:
            vects = vects + self._srng.normal(vects.shape, self._post_std)
        return strengths, vects

    def get_loss(self, input_activations, raw_feature_strengths, raw_feature_vects, extra_info=False):
        return self._inner_manager.get_loss(input_activations, raw_feature_strengths, raw_feature_vects, extra_info)

    def process(self, input_activations, extra_info=False):
        if self._pre_std is not None:
            input_activations = T.inc_subtensor(input_activations[:,:,self._pre_mask], self._srng.normal(input_activations.shape, self._pre_std))
        stuff = self._inner_manager.process(input_activations, extra_info)
        vects = stuff[2]
        if self._post_std is not None:
            vects = vects + self._srng.normal(vects.shape, self._post_std)
        return stuff[:2] + (vects,) + stuff[3:]
