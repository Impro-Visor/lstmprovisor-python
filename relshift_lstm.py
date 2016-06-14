import theano
import theano.tensor as T
import numpy as np


from theano_lstm import LSTM, StackedCells, Layer, MultiDropout
from util import *

class RelativeShiftLSTMStack( object ):
    """
    Manages a stack of LSTM cells with potentially a relative shift applied
    """

    def __init__(self, input_parts, layer_sizes, output_size, window_size=0, dropout=0):
        """
        Parameters:
            input_parts: A list of InputParts
            layer_sizes: A list of the form [ (indep, per_note), ... ] where
                    indep is the number of non-shifted cells to have, and
                    per_note is the number of cells to have per window note, which shift as the
                        network moves
                    Alternately can just be [ indep, ... ]
            output_size: An integer, the width of the desired output
            dropout: How much dropout to apply.
        """

        self.input_parts = input_parts
        self.window_size = window_size

        layer_sizes = [x if isinstance(x,tuple) else (x,0) for x in layer_sizes]
        self.layer_sizes = layer_sizes
        self.tot_layer_sizes = [(indep + per_note*self.window_size) for indep, per_note in layer_sizes]
        
        self.output_size = output_size
        self.dropout = dropout

        self.input_size = sum(part.PART_WIDTH for part in input_parts)

        self.cells = StackedCells( self.input_size, celltype=LSTM, activation=T.tanh, layers = self.tot_layer_sizes )
        self.cells.layers.append(Layer(self.tot_layer_sizes[-1], self.output_size, activation = lambda x:x))

    @property
    def params(self):
        return self.cells.params + list(l.initial_hidden_state for l in self.cells.layers if has_hidden(l))

    @params.setter
    def params(self, paramlist):
        self.cells.params = paramlist[:len(self.cells.params)]
        for l, val in zip((l for l in self.cells.layers if has_hidden(l)), paramlist[len(self.cells.params):]):
            l.initial_hidden_state.set_value(val.get_value())

    def perform_step(self, in_data, shifts, hiddens, dropout_masks=None):
        """
        Perform a step through the LSTM network.

        in_data: A theano tensor (float32) of shape (batch, input_size)
        shifts: A theano tensor (int32) of shape (batch), giving the relative
            shifts to apply to the last hiddens
        hiddens: A list of hiddens [layer](batch, hidden_idx)
        dropout_masks: If None, apply dropout deterministically. Otherwise, should
            be a set of masks returned by get_dropout_masks, generally passed through
            a scan as a non-sequence.
        """

        # hiddens is of shape [layer](batch, hidden_idx)
        # We want to permute the hidden_idx values according to shifts,
        # which are ints of shape (batch)

        n_batch = in_data.shape[0]
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
            separated_mem = per_note_mem.reshape((n_batch, self.window_size, per_note))
            # separated_mem is (batch, note, mem)
            # [a b c ... x y z] shifted up 1 goes to   [b c ... x y z 0]
            # [a b c ... x y z] shifted down 1 goes to [0 a b c ... x y]
            def _drop_shift_step(c_mem, c_shift):
                # c_mem is (note, mem)
                # c_shift is an int
                ins_at_front = T.zeros((T.maximum(0,-c_shift),per_note))
                ins_at_back = T.zeros((T.maximum(0,c_shift),per_note))
                take_part = c_mem[T.maximum(0,c_shift):self.window_size-T.maximum(0,-c_shift),:]
                return T.concatenate([ins_at_front, take_part, ins_at_back], 0)

            shifted_mem, _ = theano.map(_drop_shift_step, [separated_mem, shifts])

            new_per_note_mem = shifted_mem.reshape((n_batch, self.window_size * per_note))
            new_layer_hiddens = T.concatenate([indep_mem, new_per_note_mem, remaining_values], 1)
            new_hiddens.append(new_layer_hiddens)

        if not self.dropout:
            masks = []
        elif dropout_masks is None:
            masks = [1 - self.dropout for layer in self.cells.layers]
            masks[0] = None
        else:
            masks = [None] + dropout_masks
        new_states = self.cells.forward(in_data, prev_hiddens=new_hiddens, dropout=masks)
        return new_states

    def do_preprocess_scan(self, deterministic_dropout=False, **kwargs):
        """
        Run a scan using this LSTM, preprocessing all inputs before the scan.

        Parameters:
            kwargs[k]: should be a theano tensor of shape (n_batch, n_time, ... )
                Note that "relative_position" should be a keyword argument given here if there are relative
                shifts.
            deterministic_dropout: If True, apply dropout deterministically, scaling everything. If false,
                sample dropout

        Returns:
            A theano tensor of shape (n_batch, n_time, output_size) of activations
        """

        assert len(kwargs)>0, "Need at least one input argument!"
        n_batch, n_time = list(kwargs.values())[0].shape[:2]

        squashed_kwargs = {
            k: v.reshape([n_batch*n_time] + [x for x in v.shape[2:]]) for k,v in kwargs.items()
        }

        full_input = T.concatenate([ part.generate(**squashed_kwargs) for part in self.input_parts ], 1)
        adjusted_input = full_input.reshape([n_batch, n_time, self.input_size]).dimshuffle((1,0,2))

        # adjusted_input = theano.printing.Print("adjusted_input")(adjusted_input)

        if "relative_position" in kwargs:
            relative_position = kwargs["relative_position"]
            diff_shifts = T.extra_ops.diff(relative_position, axis=1)
            cat_shifts = T.concatenate([T.zeros((n_batch, 1), 'int32'), diff_shifts], 1)
            shifts = cat_shifts.dimshuffle((1,0))
        else:
            shifts = T.zeros(n_time, n_batch, 'int32')

        def _scan_fn(in_data, shifts, *other):
            other = list(other)
            if not self.dropout:
                masks = []
                hiddens = other
            elif deterministic_dropout:
                masks = [1 - self.dropout for layer in self.cells.layers]
                masks[0] = None
                hiddens = other
            else:
                split = -len(self.tot_layer_sizes)
                hiddens = other[:split]
                masks = [None] + other[split:]

            return self.perform_step(in_data, shifts, hiddens, dropout_masks=masks)

        if self.dropout and not deterministic_dropout:
            dropout_masks = MultiDropout( [(n_batch, shape) for shape in self.tot_layer_sizes], self.dropout)
        else:
            dropout_masks = []

        outputs_info = [initial_state_with_taps(layer, n_batch) for layer in self.cells.layers]
        result, _ = theano.scan(fn=_scan_fn, sequences=[adjusted_input, shifts], non_sequences=dropout_masks, outputs_info=outputs_info)

        final_out = get_last_layer(result).transpose((1,0,2))

        return final_out

    def do_sample_scan(self, start_pos, start_out, sample_fn, out_to_in_fn, deterministic_dropout=True, **kwargs):
        """
        Run a scan using this LSTM, sampling and processing as we go.

        Parameters:
            kwargs[k]: should be a theano tensor of shape (n_batch, n_time, ... )
                Note that "relative_position" should be a keyword argument given here if there are relative
                shifts.
            start_pos: a theano tensor of shape (n_batch) giving the initial position passed to the
                out_to_in function
            start_out: a theano tensor of shape (n_batch, X) giving the initial "output" passed
                to the out_to_in_fn
            sample_fn: a function with signature
                    sample_fn(out_activations, rel_pos) -> new_out, new_rel_pos
                where
                    - rel_pos is a theano tensor of shape (n_batch)
                    - out_activations is a tensor of shape (n_batch, output_size)
                and
                    - new_out is a tensor of shape (n_batch, X) to be output
                    - new_rel_pos should be a theano tensor of shape (n_batch)
            out_to_in_fn: a function with signature
                    out_to_in_fn(rel_pos, last_out, **cur_kwargs) -> addtl_kwargs
                where 
                    - rel_pos is a theano tensor of shape (n_batch)
                    - last_out will be a theano tensor of shape (n_batch, output_size)
                    - cur_kwargs[k] is a theano tensor of shape (n_batch, ...), from kwargs
                and
                    - addtl_kwargs[k] is a theano tensor of shape (n_batch, ...) to be added to cur kwargs
                        Note that "relative_position" will be added automatically.
            deterministic_dropout: If True, apply dropout deterministically, scaling everything. If false,
                sample dropout

        Returns: positions, raw_output, sampled_output, updates
        """

        assert len(kwargs)>0, "Need at least one input argument!"
        n_batch, n_time = list(kwargs.values())[0].shape[:2]

        transp_kwargs = {
            k: v.dimshuffle((1,0) + tuple(range(2,v.ndim))) for k,v in kwargs.items()
        }

        def _scan_fn(*stuff):
            """
            stuff will be [ kwarg_sequences..., cur_pos, last_shift, last_out, hiddens..., masks?... ]
            """
            stuff = list(stuff)
            I = len(transp_kwargs)
            kwarg_seq_vals = stuff[:I]
            cur_kwargs = {k:v for k,v in zip(transp_kwargs.keys(), kwarg_seq_vals)}
            cur_pos, last_shift, last_out = stuff[I:I+3]
            other = stuff[I+3:]

            if not self.dropout:
                masks = []
                hiddens = other
            elif deterministic_dropout:
                masks = [1 - self.dropout for layer in self.cells.layers]
                masks[0] = None
                hiddens = other
            else:
                split = -len(self.tot_layer_sizes)
                hiddens = other[:split]
                masks = [None] + other[split:]

            addtl_kwargs = out_to_in_fn(cur_pos, last_out, **cur_kwargs)

            all_kwargs = {
                "relative_position": cur_pos
            }
            all_kwargs.update(cur_kwargs)
            all_kwargs.update(addtl_kwargs)

            full_input = T.concatenate([ part.generate(**all_kwargs) for part in self.input_parts ], 1)

            step_stuff = self.perform_step(full_input, last_shift, hiddens, dropout_masks=masks)
            new_hiddens = step_stuff[:-1]
            raw_output = step_stuff[-1]
            sampled_output, new_pos = sample_fn(raw_output, cur_pos)

            new_shift = new_pos - cur_pos

            return [new_pos, new_shift, sampled_output] + step_stuff

        if self.dropout and not deterministic_dropout:
            dropout_masks = MultiDropout( [(n_batch, shape) for shape in self.tot_layer_sizes], self.dropout)
        else:
            dropout_masks = []

        outputs_info = [{"initial":start_pos, "taps":[-1]}, {"initial":T.zeros_like(start_pos), "taps":[-1]}, {"initial":start_out, "taps":[-1]}] + [initial_state_with_taps(layer, n_batch) for layer in self.cells.layers]
        result, updates = theano.scan(fn=_scan_fn, sequences=list(transp_kwargs.values()), non_sequences=dropout_masks, outputs_info=outputs_info)

        positions = T.concatenate([T.shape_padright(start_pos), result[0].transpose((1,0))[:,:-1]], 1)
        sampled_output = result[2].transpose((1,0,2))
        raw_output = result[-1].transpose((1,0,2))

        return positions, raw_output, sampled_output, updates
