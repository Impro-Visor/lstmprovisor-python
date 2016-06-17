from .queue_base import QueueManager
import theano
import theano.tensor as T

class StandardQueueManager( QueueManager ):
    """
    A standard queue manager, using a configurable set of functions
    """

    def __init__(self, feature_size, vector_activation_fun=T.nnet.sigmoid, loss_fun=(lambda x:x)):
        """
        Initialize the manager.

        Parameters:
            feature_size: The width of a feature
            vector_activation_fun: The activation function to apply to the vectors. Will be applied
                to a tensor of shape (n_parallel, feature_size) and should return a tensor of the
                same shape
            loss_fun: A function which computes the loss for each timestep. Should be an elementwise
                operation.
        """
        self._feature_size = feature_size
        self._vector_activation_fun = vector_activation_fun
        self._loss_fun = loss_fun

    @property
    def activation_width(self):
        return 1 + self.feature_size

    @property
    def feature_size(self):
        return self._feature_size

    def get_strengths_and_vects(self, input_activations):
        pre_strengths = input_activations[:,:,0]
        pre_vects = input_activations[:,:,1:]

        strengths = T.nnet.sigmoid(pre_strengths)

        flat_pre_vects = T.reshape(pre_vects,(-1,self.feature_size))
        flat_vects = self._vector_activation_fun( flat_pre_vects )
        vects = T.reshape(flat_vects, pre_vects.shape)

        return strengths, vects

    def get_loss(self, raw_feature_strengths, raw_feature_vects, extra_info=False):
        losses = self._loss_fun(raw_feature_strengths)
        full_loss = T.sum(losses)
        if extra_info:
            return full_loss, {}
        else:
            return full_loss