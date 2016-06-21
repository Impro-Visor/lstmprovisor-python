import theano
import theano.tensor as T
import numpy as np


from theano_lstm import LSTM, StackedCells, Layer
from util import *

from collections import namedtuple

SampleScanSpec = namedtuple('SampleScanSpec', ['sequences', 'non_sequences', 'outputs_info', 'num_taps', 'kwargs_keys', 'deterministic_dropout', 'start_pos'])

class RelativeShiftLSTMStack( object ):
    """
    Manages a stack of LSTM cells with potentially a relative shift applied
    """

    def __init__(self, input_parts, layer_sizes, output_size, window_size=0, dropout=0, mode="drop", unroll_batch_num=None):
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
            mode: Either "drop" or "roll". If drop, discard memory that goes out of range. If roll, roll it instead
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

        assert mode in ("drop", "roll"), "Must specify either drop or roll mode"
        self.mode = mode

        self.unroll_batch_num = unroll_batch_num

    @property
    def params(self):
        return self.cells.params + list(l.initial_hidden_state for l in self.cells.layers if has_hidden(l))

    @params.setter
    def params(self, paramlist):
        self.cells.params = paramlist[:len(self.cells.params)]
        for l, val in zip((l for l in self.cells.layers if has_hidden(l)), paramlist[len(self.cells.params):]):
            l.initial_hidden_state.set_value(val.get_value())

    def perform_step(self, in_data, shifts, hiddens, dropout_masks=[]):
        """
        Perform a step through the LSTM network.

        in_data: A theano tensor (float32) of shape (batch, input_size)
        shifts: A theano tensor (int32) of shape (batch), giving the relative
            shifts to apply to the last hiddens
        hiddens: A list of hiddens [layer](batch, hidden_idx)
        dropout_masks: If [], apply dropout deterministically. Otherwise, should
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
            # [a b c ... x y z] shifted up 1   (+1) goes to  [b c ... x y z 0]
            # [a b c ... x y z] shifted down 1 (-1) goes to [0 a b c ... x y]
            def _shift_step(c_mem, c_shift):
                # c_mem is (note, mem)
                # c_shift is an int
                if self.mode=="drop":
                    def _clamp_w(x):
                        return T.maximum(0,T.minimum(x,self.window_size))
                    ins_at_front = T.zeros((_clamp_w(-c_shift),per_note))
                    ins_at_back = T.zeros((_clamp_w(c_shift),per_note))
                    take_part = c_mem[_clamp_w(c_shift):self.window_size-_clamp_w(-c_shift),:]
                    return T.concatenate([ins_at_front, take_part, ins_at_back], 0)
                elif self.mode=="roll":
                    return T.roll(c_mem, -c_shift, axis=0)

            if self.unroll_batch_num is None:
                shifted_mem, _ = theano.map(_shift_step, [separated_mem, shifts])
            else:
                shifted_mem_parts = []
                for i in range(self.unroll_batch_num):
                    shifted_mem_parts.append(_shift_step(separated_mem[i], shifts[i]))
                shifted_mem = T.stack(shifted_mem_parts)

            new_per_note_mem = shifted_mem.reshape((n_batch, self.window_size * per_note))
            new_layer_hiddens = T.concatenate([indep_mem, new_per_note_mem, remaining_values], 1)
            new_hiddens.append(new_layer_hiddens)

        if dropout_masks == [] or not self.dropout:
            masks = []
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


        if "relative_position" in kwargs:
            relative_position = kwargs["relative_position"]
            diff_shifts = T.extra_ops.diff(relative_position, axis=1)
            cat_shifts = T.concatenate([T.zeros((n_batch, 1), 'int32'), diff_shifts], 1)
            shifts = cat_shifts.dimshuffle((1,0))
        else:
            shifts = T.zeros(n_time, n_batch, 'int32')

        def _scan_fn(in_data, shifts, *other):
            other = list(other)
            if self.dropout and not deterministic_dropout:
                split = -len(self.tot_layer_sizes)
                hiddens = other[:split]
                masks = [None] + other[split:]
            else:
                masks = []
                hiddens = other

            return self.perform_step(in_data, shifts, hiddens, dropout_masks=masks)

        if self.dropout and not deterministic_dropout:
            dropout_masks = UpscaleMultiDropout( [(n_batch, shape) for shape in self.tot_layer_sizes], self.dropout)
        else:
            dropout_masks = []

        outputs_info = [initial_state_with_taps(layer, n_batch) for layer in self.cells.layers]
        result, _ = theano.scan(fn=_scan_fn, sequences=[adjusted_input, shifts], non_sequences=dropout_masks, outputs_info=outputs_info)

        final_out = get_last_layer(result).transpose((1,0,2))

        return final_out

    def prepare_sample_scan(self, start_pos, start_out, deterministic_dropout=False, **kwargs):
        """
        Prepare a sample scan

        Parameters:
            kwargs[k]: should be a theano tensor of shape (n_batch, n_time, ... )
                Note that "relative_position" should be a keyword argument given here if there are relative
                shifts.
            start_pos: a theano tensor of shape (n_batch) giving the initial position passed to the
                out_to_in function
            start_out: a theano tensor of shape (n_batch, X) giving the initial "output" passed
                to the out_to_in_fn
            deterministic_dropout: If True, apply dropout deterministically, scaling everything. If false,
                sample dropout

        Returns:
            A namedtuple, where
                sequences: a list of sequences to input into scan
                non_sequences: a list of non_sequences into scan
                outputs_info: a list of outputs_info for scan
                num_taps: the number of outputs with taps for this 
                (other values): for internal use
        """
        assert len(kwargs)>0, "Need at least one input argument!"
        n_batch, n_time = list(kwargs.values())[0].shape[:2]

        transp_kwargs = {
            k: v.dimshuffle((1,0) + tuple(range(2,v.ndim))) for k,v in kwargs.items()
        }

        if self.dropout and not deterministic_dropout:
            dropout_masks = UpscaleMultiDropout( [(n_batch, shape) for shape in self.tot_layer_sizes], self.dropout)
        else:
            dropout_masks = []

        outputs_info = [{"initial":start_pos, "taps":[-1]}, {"initial":start_out, "taps":[-1]}] + [initial_state_with_taps(layer, n_batch) for layer in self.cells.layers]
        sequences = list(transp_kwargs.values())
        non_sequences = dropout_masks
        num_taps = len([True for x in outputs_info if x is not None])
        return SampleScanSpec(sequences=sequences, non_sequences=non_sequences, outputs_info=outputs_info, num_taps=num_taps, kwargs_keys=list(transp_kwargs.keys()), deterministic_dropout=deterministic_dropout, start_pos=start_pos)


    def sample_scan_routine(self, spec, *inputs):
        """
        Start a scan routine. This is implemented as a generator, since we may need to interrupt the state in the
        middle of iteration. How to use:

        scan_rout = x.sample_scan_routine(spec, *inputs)
                - spec: The SampleScanSpec returned by prepare_sample_scan
                - *inputs: The scan inputs, in [ sequences..., taps..., non_sequences... ] order

        last_rel_pos, last_out, cur_kwargs = scan_rout.send(None)
                - last_rel_pos is a theano tensor of shape (n_batch)
                - last_out will be a theano tensor of shape (n_batch, output_size)
                - cur_kwargs[k] is a theano tensor of shape (n_batch, ...), from kwargs

        out_activations = scan_rout.send((new_pos, addtl_kwargs))
                - new_pos is a theano tensor of shape (n_batch), giving the new relative position
                - addtl_kwargs[k] is a theano tensor of shape (n_batch, ...) to be added to cur kwargs
                    Note that "relative_position" will be added automatically.

        scan_outputs = scan_rout.send(new_out)
                - new_out is a tensor of shape (n_batch, X) to be output

        scan_rout.close()

        -> scan_outputs should be returned back to scan
        """
        stuff = list(inputs)
        I = len(spec.kwargs_keys)
        kwarg_seq_vals = stuff[:I]
        cur_kwargs = {k:v for k,v in zip(spec.kwargs_keys, kwarg_seq_vals)}
        last_pos, last_out = stuff[I:I+2]
        other = stuff[I+2:]

        if self.dropout and not deterministic_dropout:
            split = -len(self.tot_layer_sizes)
            hiddens = other[:split]
            masks = [None] + other[split:]
        else:
            masks = []
            hiddens = other

        cur_pos, addtl_kwargs = yield(last_pos, last_out, cur_kwargs)
        shift = cur_pos - last_pos

        all_kwargs = {
            "relative_position": cur_pos
        }
        all_kwargs.update(cur_kwargs)
        all_kwargs.update(addtl_kwargs)

        full_input = T.concatenate([ part.generate(**all_kwargs) for part in self.input_parts ], 1)

        step_stuff = self.perform_step(full_input, shift, hiddens, dropout_masks=masks)
        new_hiddens = step_stuff[:-1]
        raw_output = step_stuff[-1]
        sampled_output = yield(raw_output)

        yield [cur_pos, sampled_output] + step_stuff

    def extract_sample_scan_results(self, spec, outputs):
        """
        Extract outputs from the scan results. 

        Parameters:
            outputs: The outputs from the scan associated with this stack

        Returns:
            positions, raw_output, sampled_output
        """
        positions = T.concatenate([T.shape_padright(spec.start_pos), outputs[0].transpose((1,0))[:,:-1]], 1)
        sampled_output = outputs[2].transpose((1,0,2))
        raw_output = outputs[-1].transpose((1,0,2))

        return positions, raw_output, sampled_output


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
        raise NotImplementedError()
        spec = self.prepare_sample_scan(start_pos, start_out, sample_fn, deterministic_dropout, **kwargs)

        def _scan_fn(*stuff):
            scan_rout = self.sample_scan_routine(spec, *stuff)
            rel_pos, last_out, cur_kwargs = scan_rout.send(None)
            addtl_kwargs = out_to_in_fn(rel_pos, last_out, **cur_kwargs)
            out_activations = scan_rout.send(addtl_kwargs)
            sampled_output, new_pos = sample_fn(out_activations, rel_pos)
            scan_outputs = scan_rout.send((sampled_output, new_pos))
            scan_rout.close()
            return scan_outputs

        result, updates = theano.scan(fn=_scan_fn, sequences=spec.sequences, non_sequences=spec.non_sequences, outputs_info=spec.outputs_info)
        positions, raw_output, sampled_output = self.extract_sample_scan_results(spec, result)
        return positions, raw_output, sampled_output, updates
