from .queue_base import QueueManager
import theano
import theano.tensor as T

class StandardForgettableQueueManager( QueueManager ):
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
        return 3 + self.feature_size

    @property
    def feature_size(self):
        return self._feature_size

    def get_strengths_and_vects(self, input_activations):
        #
        pre_strengths = input_activations[:,:,0]
        pre_strengths = T.nnet.sigmoid(pre_strengths)
        #
        pre_commitments = input_activations[:,:,1]
        pre_commitments = T.nnet.sigmoid(pre_commitments)
        #
        pre_forgets = input_activations[:,:,2]
        pre_forgets = T.nnet.sigmoid(pre_forgets)

        init_strengths = T.zeros(pre_strengths.shape)
        init_uncertainties = T.zeros(pre_forgets.shape)

        pre_strengths = pre_strengths.swapaxes(0,1)
        pre_commitments = pre_commitments.swapaxes(0,1)
        pre_forgets = pre_forgets.swapaxes(0,1)

        def forgetFun(strengthSignal, forgetSignal, commitSignal, timeStep, prev_strengths, prev_uncertainties):
            commitSignal = T.shape_padright(commitSignal)
            forgetSignal = T.shape_padright(forgetSignal)
            #relationshipgoals
            prev_uncertainties = prev_uncertainties - (prev_uncertainties * commitSignal) #decrement our uncertainty according to commitSignal

            #unstoppable
            forgetStrengths = (prev_uncertainties * forgetSignal) #get our forget amount per timeStep from forgetSignal and uncertainty strength
            prev_strengths = prev_strengths - forgetStrengths #decrement our strength and uncertainty by our forget amount
            prev_uncertainties = prev_uncertainties - forgetStrengths

            #kony2012
            prev_uncertainties = T.set_subtensor(prev_uncertainties[:,timeStep], strengthSignal) #set our current timeStep values
            prev_strengths = T.set_subtensor(prev_strengths[:,timeStep], strengthSignal)

            #yolo
            timeStep = timeStep + 1
            return timeStep, prev_strengths, prev_uncertainties

        #scan strength, forget, commit signals to apply forgetting through timeSteps, starting at timeStep 0
        (timeStep, strengths, uncertainties), updates = theano.scan(fn=forgetFun,
                                               outputs_info=[0, init_strengths, init_uncertainties],
                                               sequences=[pre_strengths,pre_forgets,pre_commitments])
        strengths = strengths[-1]
        #Set final strength to 1
        strengths = T.set_subtensor(strengths[:,-1],1)

        pre_vects = input_activations[:,:,3:]

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
