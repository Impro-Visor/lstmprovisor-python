import theano
import theano.tensor as T
import numpy as np

WINDOW_SIZE = 5
import relative_data
from model import OutputConversionOp
import constants

def test_rotation(test_input, indep, per_note, shifts):
    n_batch = test_input.shape[0]
    indep_hiddens = test_input[:,:indep]
    per_note_hiddens = test_input[:,indep:]
    # per_note_hiddens is (batch, per_note_hiddens)
    separated_hiddens = per_note_hiddens.reshape((n_batch, WINDOW_SIZE, per_note))
    # separated_hiddens is (batch, note, hiddens)
    if True:
        # [a b c ... x y z] shifted up 1 goes to   [b c ... x y z 0]
        # [a b c ... x y z] shifted down 1 goes to [0 a b c ... x y]
        def _drop_shift_step(c_hiddens, c_shift):
            # c_hiddens is (note, hiddens)
            # c_shift is an int
            ins_at_front = T.zeros((T.maximum(0,-c_shift),per_note))
            ins_at_back = T.zeros((T.maximum(0,c_shift),per_note))
            take_part = c_hiddens[T.maximum(0,c_shift):WINDOW_SIZE-T.maximum(0,-c_shift),:]
            return T.concatenate([ins_at_front, take_part, ins_at_back], 0)

        shifted_hiddens,_ = theano.map(_drop_shift_step, [separated_hiddens, shifts])
    else:
        raise NotImplementedError("Only drop mode is implemented")

    new_per_note_hiddens = shifted_hiddens.reshape((n_batch, WINDOW_SIZE * per_note))
    new_layer_hiddens = T.concatenate([indep_hiddens, new_per_note_hiddens], 1)
    return new_layer_hiddens

def run_test_rotation():
    test_input = T.fmatrix()
    shifts = T.ivector()
    indep = 4
    per_note = 3
    output = test_rotation(test_input, indep, per_note, shifts)
    runfn = theano.function(inputs=[test_input, shifts], outputs=output, allow_input_downcast=True)

    batchsize = 2*WINDOW_SIZE+1
    otherdim = (indep + WINDOW_SIZE*per_note)
    # my_input = np.arange(batchsize*otherdim).reshape((batchsize, otherdim))
    my_input = np.tile(np.expand_dims(np.arange(otherdim),0), (batchsize,1))
    print(my_input)
    myshifts = np.arange(-WINDOW_SIZE,WINDOW_SIZE+1)
    print(myshifts)
    my_output = runfn(my_input, myshifts)
    print(my_output)

def test_sampling(probs,srng):
    """
    probs of shape (batch, choices)
    """
    cum_probs = T.extra_ops.cumsum(probs, 1)

    sampler = srng.uniform([probs.shape[0],1])

    indicator = T.switch(cum_probs > sampler, cum_probs, 2)
    argmin = T.argmin(indicator, 1)
    sampled_output = T.extra_ops.to_one_hot(argmin, probs.shape[1])

    return sampled_output

def run_test_sampling():
    test_probs = T.fmatrix()
    srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))
    output = test_sampling(test_probs, srng)
    runfn = theano.function(inputs=[test_probs], outputs=output, allow_input_downcast=True)

    my_input = [[0,0,1,0,0], [0.25,0.5,0.25,0,0], [0,0.5,0,0.5,0]]
    print(my_input)

    for i in range(10):
        my_output = runfn(my_input)
        print(my_output)

def run_test_output_conversion():

    input_out = T.fmatrix()
    input_pos = T.ivector()
    input_chord = T.fmatrix()
    out_ipt, out_shift, out_pos = OutputConversionOp()(input_out, input_pos, input_chord)
    runfn = theano.function(inputs=[input_out, input_pos, input_chord],
                            outputs=[out_ipt, out_shift, out_pos],
                            allow_input_downcast=True)

    my_out = np.array([
        [1,0] + [0 for i in range(relative_data.WINDOW_SIZE)],
        [0,1] + [0 for i in range(relative_data.WINDOW_SIZE)],
        [0,0] + [(i==relative_data.WINDOW_RADIUS) for i in range(relative_data.WINDOW_SIZE)],
        [0,0] + [(i==relative_data.WINDOW_RADIUS+3) for i in range(relative_data.WINDOW_SIZE)],
    ], np.float32)
    my_pos = np.array([60,62,60,60], np.int32)
    my_chord = np.array([constants.CHORD_TYPES['']]*4, np.int32)

    print(my_out)
    print(my_pos)
    print(my_chord)
    print("---------")

    res = runfn(my_out,my_pos,my_chord)
    for r in res:
        print(r)









