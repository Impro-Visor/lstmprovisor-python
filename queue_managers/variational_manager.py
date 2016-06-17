from .queue_base import QueueManager, fragmented_queue_process
import theano
import theano.tensor as T

class VariationalQueueManager( QueueManager ):
    """
    A variational-autoencoder-based queue manager, using a configurable loss
    """

    def __init__(self, feature_size, srng, loss_fun=(lambda x:x)):
        """
        Initialize the manager.

        Parameters:
            feature_size: The width of a feature
            srng: A theano random streams object
            loss_fun: A function which computes the loss for each timestep. Should be an elementwise
                operation.
        """
        self._feature_size = feature_size
        self._srng = srng
        self._vector_activation_fun = vector_activation_fun
        self._loss_fun = loss_fun

    @property
    def activation_width(self):
        return 1 + self.feature_size*2

    @property
    def feature_size(self):
        return self._feature_size

    def helper_sample(self, input_activations):
        pre_strengths = input_activations[:,:,0]
        strengths = T.sigmoid(pre_strengths)

        means = input_activations[:,:,1:1+self.feature_size]
        stdevs = input_activations[:,:,1+self.feature_size:]
        wiggle = self._srng.normal(means.shape)

        vects = means + (stdevs * wiggle)

        return strengths, vects, means, stdevs

    def get_strengths_and_vects(self, input_activations):
        strengths, vects, means, stdevs = self.helper_sample(input_activations)
        return strengths, vects

    def process(self, input_activations):

        strengths, vects, means, stdevs = self.helper_sample(input_activations)
        transformed_output = fragmented_queue_process(strengths, vects)

        sparsity_losses = self._loss_fun(strengths)
        full_sparsity_loss = T.sum(losses)

        means_sq = means**2
        variance = stdevs**2
        variational_loss = 0.5 * T.sum(1 + T.log(variance) - means_sq - variance)

        full_loss = full_sparsity_loss + variational_loss
        
        return transformed_output, full_loss