import theano
import theano.tensor as T
import numpy as np

import relative_data

from theano_lstm import Embedding, LSTM, RNN, StackedCells, Layer, create_optimization_updates, masked_loss, MultiDropout
from adam import Adam

EPSILON = np.finfo(np.float32).tiny

def has_hidden(layer):
    """
    Whether a layer has a trainable
    initial hidden state.
    """
    return hasattr(layer, 'initial_hidden_state')

def matrixify(vector, n):
    return T.repeat(T.shape_padleft(vector), n, axis=0)

def initial_state(layer, dimensions = None):
    """
    Initalizes the recurrence relation with an initial hidden state
    if needed, else replaces with a "None" to tell Theano that
    the network **will** return something, but it does not need
    to send it to the next step of the recurrence
    """
    if dimensions is None:
        return layer.initial_hidden_state if has_hidden(layer) else None
    else:
        return matrixify(layer.initial_hidden_state, dimensions) if has_hidden(layer) else None

def initial_state_with_taps(layer, dimensions = None):
    """Optionally wrap tensor variable into a dict with taps=[-1]"""
    state = initial_state(layer, dimensions)
    if state is not None:
        return dict(initial=state, taps=[-1])
    else:
        return None

        
def get_last_layer(result):
    if isinstance(result, list):
        return result[-1]
    else:
        return result

def ensure_list(result):
    if isinstance(result, list):
        return result
    else:
        return [result]

class OutputConversionOp(theano.Op):
    # Properties attribute
    __props__ = ()

    def make_node(self, out, pos, chord):
        out = T.as_tensor_variable(out)
        pos = T.as_tensor_variable(pos)
        chord = T.as_tensor_variable(chord)
        return theano.Apply(self, [out, pos, chord], [T.fmatrix(), T.ivector(), T.ivector()])
    
    # Python implementation:
    def perform(self, node, inputs_storage, output_storage):
        out, pos, chord = inputs_storage

        new_input = []
        new_shift = []
        new_pos = []
        for c_out, c_pos, c_chord in zip(out,pos,chord):
            if c_out[0] or c_out[1]:
                n_shift = 0
            else:
                n_shift = np.nonzero(c_out)[0][0]-2-relative_data.WINDOW_RADIUS
            n_pos = c_pos + n_shift
            n_chord = np.roll(c_chord, -n_pos, 0)

            n_input = np.concatenate([
                n_chord,
                [relative_data.vague_position(n_pos)],
                c_out
            ],0)
            new_input.append(n_input)
            new_shift.append(n_shift)
            new_pos.append(n_pos)

        output_storage[0][0] = np.array(new_input, np.float32)
        output_storage[1][0] = np.array(new_shift, np.int32)
        output_storage[2][0] = np.array(new_pos, np.int32)


class Model(object):

    def __init__(self, layer_sizes, shift_mode="drop", dropout=0, setup=False):
        """
        Initialize the model.

        layer_sizes: An array of the form
            [ (indep, per_note), ... ]
        where
            indep is the number of non-shifted cells to have, and
            per_note is the number of cells to have per window note, which shift as the
                network moves

        shift_mode: Must be exactly
            "drop": Discard information that leaves the window. Replace with zeros.
            (other options may be added later)
        """
        self.layer_sizes = layer_sizes
        self.tot_layer_sizes = [(indep + per_note*relative_data.WINDOW_SIZE) for indep, per_note in layer_sizes]
        self.dropout = dropout

        self.shift_mode = shift_mode

        self.input_size = relative_data.INPUT_SIZE
        self.output_size = relative_data.OUTPUT_SIZE

        self.cells = StackedCells( self.input_size, celltype=LSTM, layers = self.tot_layer_sizes )
        self.cells.layers.append(Layer(self.tot_layer_sizes[-1], self.output_size, activation = lambda x:x))

        self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

        if setup:
            print("Setting up train")
            self.setup_train()
            print("Setting up gen")
            self.setup_generate()
            print("Done setting up")

    @property
    def learned_config(self):
        return [self.cells.params, [l.initial_hidden_state for l in self.cells.layers if has_hidden(l)]]

    @learned_config.setter
    def learned_config(self, learned_list):
        self.cells.params = learned_list[0]
        for l, val in zip((l for l in self.cells.layers if has_hidden(l)), learned_list[1]):
            l.initial_hidden_state.set_value(val.get_value())

    def get_params(self):
        return self.cells.params + list(l.initial_hidden_state for l in self.cells.layers if has_hidden(l))

    def get_step_fn(self, n_batch, deterministic_dropout=False):
        def _helper(in_data, shifts, *other):
            other = list(other)
            if self.dropout and not deterministic_dropout:
                split = -len(self.tot_layer_sizes)
                hiddens = other[:split]
            else:
                hiddens = other

            # hiddens is of shape [layer](batch, hidden_idx)
            # We want to permute the hidden_idx values according to shifts,
            # which are ints of shape (batch)
            new_hiddens = []
            for layer_i, (indep, per_note) in enumerate(self.layer_sizes):
                if per_note == 0:
                    # Don't bother with this layer
                    new_hiddens.append(hiddens[layer_i])
                    continue
                # The theano_lstm code puts [memory_cells... , old_activations...]
                # We want to slide the memory cells only.
                lstm_hsplit = self.cells.layers[layer_i].hidden_size
                indep_mem = hiddens[layer_i][:,:indep]
                per_note_mem = hiddens[layer_i][:,indep:lstm_hsplit]
                remaining_values = hiddens[layer_i][:,lstm_hsplit:]
                # per_note_mem is (batch, per_note_mem)
                separated_mem = per_note_mem.reshape((n_batch, relative_data.WINDOW_SIZE, per_note))
                # separated_mem is (batch, note, mem)
                if self.shift_mode == "drop":
                    # [a b c ... x y z] shifted up 1 goes to   [b c ... x y z 0]
                    # [a b c ... x y z] shifted down 1 goes to [0 a b c ... x y]
                    def _drop_shift_step(c_mem, c_shift):
                        # c_mem is (note, mem)
                        # c_shift is an int
                        ins_at_front = T.zeros((T.maximum(0,-c_shift),per_note))
                        ins_at_back = T.zeros((T.maximum(0,c_shift),per_note))
                        take_part = c_mem[T.maximum(0,c_shift):relative_data.WINDOW_SIZE-T.maximum(0,-c_shift),:]
                        return T.concatenate([ins_at_front, take_part, ins_at_back], 0)

                    shifted_mem, _ = theano.map(_drop_shift_step, [separated_mem, shifts])
                else:
                    raise NotImplementedError("Only drop mode is implemented")

                new_per_note_mem = shifted_mem.reshape((n_batch, relative_data.WINDOW_SIZE * per_note))
                new_layer_hiddens = T.concatenate([indep_mem, new_per_note_mem, remaining_values], 1)
                new_hiddens.append(new_layer_hiddens)

            if not self.dropout:
                masks = []
            elif deterministic_dropout:
                masks = [1 - self.dropout for layer in self.cells.layers]
                masks[0] = None
            else:
                masks = [None] + other[split:]
            new_states = self.cells.forward(in_data, prev_hiddens=new_hiddens, dropout=masks)
            return new_states + [T.nnet.softmax(new_states[-1])]
        return _helper

    def setup_train(self):

        # dimensions: (batch, time, input_data)
        self.input_mat = T.btensor3()

        # dimensions: (batch, time)
        self.mem_shifts = T.imatrix()

        # dimesions: (batch, time, output_data)
        self.output_mat = T.btensor3()

        n_batch, n_time, _ = self.input_mat.shape

        # time_inputs is (time, batch, input_data)
        time_inputs = self.input_mat.transpose((1,0,2))
        shift_inputs = self.mem_shifts.transpose((1,0))

        # apply dropout
        if self.dropout > 0:
            dropout_masks = MultiDropout( [(n_batch, shape) for shape in self.tot_layer_sizes], self.dropout)
        else:
            dropout_masks = []

        outputs_info = [initial_state_with_taps(layer, n_batch) for layer in self.cells.layers] + [None]
        result, _ = theano.scan(fn=self.get_step_fn(n_batch,False), sequences=[time_inputs, shift_inputs], non_sequences=dropout_masks, outputs_info=outputs_info)
        
        # result is a list [layer] of matrix (time, batch, hiddens/output_data)
        # final_out is last layer of result transposed back to (batch, time, output_data)
        final_out = get_last_layer(result).transpose((1,0,2))

        loglikelihoods = T.log( final_out + EPSILON )*self.output_mat
        self.loss = T.neg(T.sum(loglikelihoods)/T.cast(n_batch*n_time, theano.config.floatX))

        updates = Adam(self.loss, self.get_params())

        self.update_fun = theano.function(
            inputs=[self.input_mat, self.mem_shifts, self.output_mat],
            outputs=self.loss,
            updates=updates,
            allow_input_downcast=True)

        self.eval_fun = theano.function(
            inputs=[self.input_mat, self.mem_shifts, self.output_mat],
            outputs=self.loss,
            allow_input_downcast=True)

    def setup_generate(self):

        # dimensions: (batch, time, chord_bit)
        self.chords = T.btensor3()
        n_batch, n_time, _ = self.chords.shape

        chord_input = self.chords.transpose((1,0,2))

        initial_output_single = np.expand_dims(np.array([1, 0] + [0]*relative_data.WINDOW_SIZE, np.float32),0)
        initial_position_single = np.expand_dims(np.array(relative_data.STARTING_POSITION, np.int32),0)

        initial_output = T.tile(initial_output_single, (n_batch, 1))
        initial_position = T.tile(initial_position_single, (n_batch))

        def step_gen(cur_chord, last_output, last_pos, *other):
            new_input, new_shifts, new_pos = OutputConversionOp()(last_output, last_pos, cur_chord)
            next_stuff = self.get_step_fn(n_batch,True)(new_input, new_shifts, *other)

            # next_output_probs is of shape (batch, output_softmax)
            next_output_probs = next_stuff[-1]

            cum_probs = T.extra_ops.cumsum(next_output_probs, 1)
            # cum_probs = theano.printing.Print("Cumulative probs")(cum_probs)

            sampler = self.srng.uniform([n_batch,1])

            indicator = T.switch(cum_probs > sampler, cum_probs, 2)
            argmin = T.argmin(indicator, 1)
            sampled_output = T.extra_ops.to_one_hot(argmin, relative_data.OUTPUT_SIZE)
            # Note: As long as the probabilities add up to 1, this works.
            # Somewhat sneakily, if probabilities add to <1, if we try to sample too
            # high it should pick index [0], which is rest.
            # sampled_output = theano.printing.Print("Sampled")(sampled_output)

            return [sampled_output, new_pos,] + next_stuff[:-1]

        outputs_info = ([ dict(initial=initial_output, taps=[-1]) ] + 
                        [ dict(initial=initial_position, taps=[-1]) ] + 
                        [initial_state_with_taps(layer, n_batch) for layer in self.cells.layers])
        result, updates = theano.scan(fn=step_gen, sequences=[chord_input], outputs_info=outputs_info)

        all_outputs = result[0].transpose((1,0,2))
        # all_outputs is (batch, time, chord_bits)

        self.generate_fun = theano.function(
            inputs=[self.chords],
            updates=updates,
            outputs=all_outputs,
            allow_input_downcast=True)







