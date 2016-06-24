from .queue_base import QueueManager
from .variational_manager import VariationalQueueManager
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import numpy as np
import constants

class SamplingVariationalQueueManager( VariationalQueueManager ):
    """
    A variational-autoencoder-based queue manager, using a configurable loss, with sampled pushes and pops
    """

    def __init__(self, feature_size, loss_fun=(lambda x:x), baseline_scale=0.9):
        """
        Initialize the manager.

        Parameters:
            feature_size: The width of a feature
            loss_fun: A function which computes the loss for each timestep. Should be an elementwise
                operation.
            baseline_scale: How much to adjust the baseline
        """
        super().__init__(feature_size, loss_fun)
        self._baseline_scale = baseline_scale

    def helper_sample(self, input_activations):
        strength_probs, vects, means, stdevs, old_info = super().helper_sample(input_activations)
        samples = self._srng.uniform(strength_probs.shape)
        strengths = T.cast(strength_probs>samples, 'float32')
        return strengths, vects, means, stdevs, {"sample_strength_probs":strength_probs, "sample_strength_choices":strengths}

    def surrogate_loss(self, reconstruction_cost, extra_info):
        """
        Based on "Gradient Estimation Using Stochastic Computation Graphs", we can compute the gradient estimate
        as

            grad(E[cost]) ~= E[ sum(grad(log p(w|...)) * (Q - b)) + grad(cost(w))]

        where
         - w is the current thing we sampled (so p(w|...) is the probability we would do what we sampled doing)
         - Q is the cost "downstream" of w
         - b is an arbitrary baseline, which must not be downstream of w

        In this case, each w is a particular choice we made in sampling the strengths, and  Q is just the
        reconstruction cost (since the final output can depend on strengths both in the past and the future).
        We let b be an exponential average of previous values of Q.

        We can construct our surrogate loss function as

            L = sum(log p(w|...)*(Q - b)) + actual costs
              = (Q - b)*sum(log p(w|...)) + actual costs

        as long as we consider Q and b constant wrt any derivative operation. This function thus returns

            S = (Q - b)*sum(log p(w|...))
        """
        s_probs = extra_info["sample_strength_probs"]
        s_choices = extra_info["sample_strength_choices"]
        prob_do_sampled =  s_probs * s_choices + (1-s_probs)*(1-s_choices)
        logprobsum = T.sum(T.log(prob_do_sampled))

        accum_prev_Q = theano.shared(np.array(0, np.float32))
        accum_divisor = theano.shared(np.array(constants.EPSILON, np.float32))
        baseline = accum_prev_Q / accum_divisor

        Q = theano.gradient.disconnected_grad(reconstruction_cost)

        surrogate_loss_component = logprobsum * (Q - baseline)

        new_prev_Q = (self._baseline_scale)*accum_prev_Q + (1-self._baseline_scale)*Q
        new_divisor = (self._baseline_scale)*accum_divisor + (1-self._baseline_scale)

        updates = [(accum_prev_Q, new_prev_Q), (accum_divisor, new_divisor)]

        return surrogate_loss_component, updates




