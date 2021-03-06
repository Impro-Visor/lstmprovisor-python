
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy as np

import constants
import input_parts
from relshift_lstm import RelativeShiftLSTMStack
from adam import Adam
from note_encodings import Encoding
import leadsheet

import itertools
import functools

from theano.compile.nanguardmode import NanGuardMode


class ProductOfExpertsModel(object):
    def __init__(self, encodings, all_layer_sizes, inputs=None, shift_modes=None, dropout=0, setup=False, nanguard=False, unroll_batch_num=None, bounds=constants.BOUNDS, normalize_artic_only=False, skip_training_experts=[]):
        self.encodings = encodings

        self.bounds = bounds
        self.normalize_artic_only = normalize_artic_only
        self.skip_training_experts = skip_training_experts
        
        if shift_modes is None:
            shift_modes = ["drop"]*len(encodings)

        if inputs is None:
            inputs = [[
                input_parts.BeatInputPart(),
                input_parts.PositionInputPart(self.bounds.lowbound, self.bounds.highbound, 2),
                input_parts.ChordShiftInputPart()]]*len(self.encodings)

        self.all_layer_sizes = all_layer_sizes
        self.lstmstacks = []
        for layer_sizes, encoding, shift_mode, ipt in zip(all_layer_sizes,encodings,shift_modes, inputs):
            parts = ipt + [
                input_parts.PassthroughInputPart("last_output", encoding.ENCODING_WIDTH)
            ]
            lstmstack = RelativeShiftLSTMStack(parts, layer_sizes, encoding.RAW_ENCODING_WIDTH, encoding.WINDOW_SIZE, dropout, mode=shift_mode, unroll_batch_num=unroll_batch_num)
            self.lstmstacks.append(lstmstack)

        self.srng = MRG_RandomStreams(np.random.randint(1, 1024))

        self.learning_rate_var = theano.shared(np.array(0.0002, theano.config.floatX))

        self.update_fun = None
        self.eval_fun = None
        self.gen_fun = None

        self.nanguard = nanguard

        if setup:
            print("Setting up train")
            self.setup_train()
            print("Setting up gen")
            self.setup_generate()
            print("Done setting up")

    @property
    def params(self):
        return list(itertools.chain(*(lstmstack.params for lstmstack in self.lstmstacks)))

    @params.setter
    def params(self, paramlist):
        mycopy = list(paramlist)
        for lstmstack in self.lstmstacks:
            lstmstack.params = mycopy[:len(lstmstack.params)]
            del mycopy[:len(lstmstack.params)]
        assert len(mycopy) == 0

    def get_optimize_params(self):
        return list(itertools.chain(*(lstmstack.params for i,lstmstack in enumerate(self.lstmstacks) if i not in self.skip_training_experts)))

    def set_learning_rate(self, lr):
        self.learning_rate_var.set_value(np.array(lr, theano.config.floatX))

    def setup_train(self):

        # dimensions: (batch, time, 12)
        chord_types = T.btensor3()

        # dimensions: (batch, time)
        chord_roots = T.imatrix()

        # dimensions: (batch, time)
        relative_posns = [T.imatrix() for _ in self.encodings]

        # dimesions: (batch, time, output_data)
        encoded_melodies = [T.btensor3() for _ in self.encodings]

        # dimesions: (batch, time)
        correct_notes = T.imatrix()

        n_batch, n_time = chord_roots.shape

        def _build(det_dropout):
            all_out_probs = []
            for encoding, lstmstack, encoded_melody, relative_pos in zip(self.encodings, self.lstmstacks, encoded_melodies, relative_posns):
                activations = lstmstack.do_preprocess_scan( timestep=T.tile(T.arange(n_time), (n_batch,1)) ,
                                                            relative_position=relative_pos,
                                                            cur_chord_type=chord_types,
                                                            cur_chord_root=chord_roots,
                                                            last_output=T.concatenate([T.tile(encoding.initial_encoded_form(), (n_batch,1,1)),
                                                                                encoded_melody[:,:-1,:] ], 1),
                                                            deterministic_dropout=det_dropout)

                out_probs = encoding.decode_to_probs(activations, relative_pos, self.bounds.lowbound, self.bounds.highbound)
                all_out_probs.append(out_probs)
            reduced_out_probs = functools.reduce((lambda x,y: x*y), all_out_probs)
            if self.normalize_artic_only:
                non_artic_probs = reduced_out_probs[:,:,:2]
                artic_probs = reduced_out_probs[:,:,2:]
                non_artic_sum = T.sum(non_artic_probs, 2, keepdims=True)
                artic_sum = T.sum(artic_probs, 2, keepdims=True)
                norm_artic_probs = artic_probs*(1-non_artic_sum)/artic_sum
                norm_out_probs = T.concatenate([non_artic_probs, norm_artic_probs], 2)
            else:
                normsum = T.sum(reduced_out_probs, 2, keepdims=True)
                normsum = T.maximum(normsum, constants.EPSILON)
                norm_out_probs = reduced_out_probs/normsum
            return Encoding.compute_loss(norm_out_probs, correct_notes, True)

        train_loss, train_info = _build(False)
        updates = Adam(train_loss, self.get_optimize_params(), lr=self.learning_rate_var)

        eval_loss, eval_info = _build(True)

        self.loss_info_keys = list(train_info.keys())

        self.update_fun = theano.function(
            inputs=[chord_types, chord_roots, correct_notes] + relative_posns + encoded_melodies,
            outputs=[train_loss]+list(train_info.values()),
            updates=updates,
            allow_input_downcast=True,
            on_unused_input='ignore',
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

        self.eval_fun = theano.function(
            inputs=[chord_types, chord_roots, correct_notes] + relative_posns + encoded_melodies,
            outputs=[eval_loss]+list(eval_info.values()),
            allow_input_downcast=True,
            on_unused_input='ignore',
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

    def _assemble_batch(self, melody, chords):
        encoded_melodies = [[] for _ in self.encodings]
        relative_posns = [[] for _ in self.encodings]
        correct_notes = []
        chord_roots = []
        chord_types = []
        for m,c in zip(melody,chords):
            m = leadsheet.constrain_melody(m, self.bounds)
            for i,encoding in enumerate(self.encodings):
                e_m, r_p = encoding.encode_melody_and_position(m,c)
                encoded_melodies[i].append(e_m)
                relative_posns[i].append(r_p)
            correct_notes.append(Encoding.encode_absolute_melody(m, self.bounds.lowbound, self.bounds.highbound))
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        return ([np.array(chord_types, np.float32),
                 np.array(chord_roots, np.int32),
                 np.array(correct_notes, np.int32)]
                + [np.array(x, np.int32) for x in relative_posns]
                + [np.array(x, np.int32) for x in encoded_melodies])

    def train(self, chords, melody):
        assert self.update_fun is not None, "Need to call setup_train before train"
        res = self.update_fun(*self._assemble_batch(melody,chords))
        loss = res[0]
        info = dict(zip(self.loss_info_keys, res[1:]))
        return loss, info

    def eval(self, chords, melody):
        assert self.update_fun is not None, "Need to call setup_train before eval"
        res = self.eval_fun(*self._assemble_batch(melody,chords))
        loss = res[0]
        info = dict(zip(self.loss_info_keys, res[1:]))
        return loss, info

    def setup_generate(self):

        # dimensions: (batch, time, 12)
        chord_types = T.btensor3()

        # dimensions: (batch, time)
        chord_roots = T.imatrix()

        n_batch, n_time = chord_roots.shape

        specs = [lstmstack.prepare_sample_scan(  start_pos=T.alloc(np.array(encoding.STARTING_POSITION, np.int32), (n_batch)),
                                                    start_out=T.tile(encoding.initial_encoded_form(), (n_batch,1)),
                                                    timestep=T.tile(T.arange(n_time), (n_batch,1)),
                                                    cur_chord_type=chord_types,
                                                    cur_chord_root=chord_roots,
                                                    deterministic_dropout=True )
                    for lstmstack, encoding in zip(self.lstmstacks, self.encodings)]

        updates, all_chosen, all_probs, indiv_probs = helper_generate_from_spec(specs, self.lstmstacks, self.encodings, self.srng, n_batch, n_time, self.bounds, self.normalize_artic_only)

        self.generate_fun = theano.function(
            inputs=[chord_roots, chord_types],
            updates=updates,
            outputs=all_chosen,
            allow_input_downcast=True,
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

        self.generate_visualize_fun = theano.function(
            inputs=[chord_roots, chord_types],
            updates=updates,
            outputs=[all_chosen, all_probs] + indiv_probs,
            allow_input_downcast=True,
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

    def generate(self, chords):
        assert self.generate_fun is not None, "Need to call setup_generate before generate"

        chord_roots = []
        chord_types = []
        for c in chords:
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        chosen = self.generate_fun(np.array(chord_roots, np.int32),np.array(chord_types, np.float32))
        return [Encoding.decode_absolute_melody(c, self.bounds.lowbound, self.bounds.highbound) for c in chosen]

    def generate_visualize(self, chords):
        assert self.generate_fun is not None, "Need to call setup_generate before generate"
        chord_roots = []
        chord_types = []
        for c in chords:
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        stuff = self.generate_visualize_fun(chord_roots, chord_types)
        chosen, all_probs = stuff[:2]

        melody = [Encoding.decode_absolute_melody(c, self.bounds.lowbound, self.bounds.highbound) for c in chosen]
        return melody, chosen, all_probs, stuff[2:]

    def setup_produce(self):
        self.setup_generate()

    def produce(self, chords, melody):
        return self.generate_visualize(chords)

def helper_generate_from_spec(specs, lstmstacks, encodings, srng, n_batch, n_time, bounds, normalize_artic_only=False):
    """Helper function to generate through a product LSTM model"""
    def _scan_fn(*inputs):
        # inputs is [ spec_sequences..., last_absolute_position, spec_taps..., spec_non_sequences... ]
        inputs = list(inputs)

        partitioned_inputs = [[] for _ in specs]
        for cur_part, spec in zip(partitioned_inputs, specs):
            cur_part.extend(inputs[:len(spec.sequences)])
            del inputs[:len(spec.sequences)]
        last_absolute_chosen = inputs.pop(0)
        for cur_part, spec in zip(partitioned_inputs, specs):
            cur_part.extend(inputs[:spec.num_taps])
            del inputs[:spec.num_taps]
        for cur_part, spec in zip(partitioned_inputs, specs):
            cur_part.extend(inputs[:len(spec.non_sequences)])
            del inputs[:len(spec.non_sequences)]

        scan_routs = [ lstmstack.sample_scan_routine(spec, *p_input) for lstmstack,spec,p_input in zip(lstmstacks, specs, partitioned_inputs) ]
        new_posns = []
        all_out_probs = []
        for scan_rout, encoding in zip(scan_routs, encodings):
            last_rel_pos, last_out, cur_kwargs = scan_rout.send(None)

            new_pos = encoding.get_new_relative_position(last_absolute_chosen, last_rel_pos, last_out, bounds.lowbound, bounds.highbound, **cur_kwargs)
            new_posns.append(new_pos)
            addtl_kwargs = {
                "last_output": last_out
            }

            out_activations = scan_rout.send((new_pos, addtl_kwargs))
            out_probs = encoding.decode_to_probs(out_activations,new_pos,bounds.lowbound, bounds.highbound)
            all_out_probs.append(out_probs)

        reduced_out_probs = functools.reduce((lambda x,y: x*y), all_out_probs)
        if normalize_artic_only:
            non_artic_probs = reduced_out_probs[:,:2]
            artic_probs = reduced_out_probs[:,2:]
            non_artic_sum = T.sum(non_artic_probs, 1, keepdims=True)
            artic_sum = T.sum(artic_probs, 1, keepdims=True)
            norm_artic_probs = artic_probs*(1-non_artic_sum)/artic_sum
            norm_out_probs = T.concatenate([non_artic_probs, norm_artic_probs], 1)
        else:
            normsum = T.sum(reduced_out_probs, 1, keepdims=True)
            normsum = T.maximum(normsum, constants.EPSILON)
            norm_out_probs = reduced_out_probs/normsum

        sampled_note = Encoding.sample_absolute_probs(srng, norm_out_probs)

        outputs = []
        for scan_rout, encoding, new_pos in zip(scan_routs, encodings, new_posns):
            encoded_output = encoding.note_to_encoding(sampled_note, new_pos, bounds.lowbound, bounds.highbound)
            scan_outputs = scan_rout.send(encoded_output)
            scan_rout.close()
            outputs.extend(scan_outputs)

        return [sampled_note, norm_out_probs] + all_out_probs + outputs

    sequences = []
    non_sequences = []
    outputs_info = [{"initial":T.zeros((n_batch,),'int32'), "taps":[-1]}, None] + [None]*len(specs)
    for spec in specs:
        sequences.extend(spec.sequences)
        non_sequences.extend(spec.non_sequences)
        outputs_info.extend(spec.outputs_info)
    
    result, updates = theano.scan(fn=_scan_fn, sequences=sequences, non_sequences=non_sequences, outputs_info=outputs_info)
    all_chosen = result[0].dimshuffle((1,0))
    all_probs = result[1].dimshuffle((1,0,2))
    indiv_probs = [r.dimshuffle((1,0,2)) for r in result[2:2+len(specs)]]

    return updates, all_chosen, all_probs, indiv_probs
