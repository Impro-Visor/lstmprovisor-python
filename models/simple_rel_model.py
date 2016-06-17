
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy as np

import constants
import input_parts
from relshift_lstm import RelativeShiftLSTMStack
from adam import Adam
from note_encodings import Encoding

class SimpleModel(object):
    def __init__(self, encoding, layer_sizes, shift_mode="drop", dropout=0, setup=False):

        self.encoding = encoding

        parts = [
            input_parts.BeatInputPart(),
            input_parts.PositionInputPart(constants.LOW_BOUND, constants.HIGH_BOUND, 2),
            input_parts.ChordShiftInputPart(),
            input_parts.PassthroughInputPart("last_output", encoding.ENCODING_WIDTH)
        ]
        self.lstmstack = RelativeShiftLSTMStack(parts, layer_sizes, encoding.RAW_ENCODING_WIDTH, encoding.WINDOW_SIZE, dropout, mode=shift_mode)

        self.srng = MRG_RandomStreams(np.random.randint(0, 1024))

        self.update_fun = None
        self.eval_fun = None
        self.gen_fun = None

        if setup:
            print("Setting up train")
            self.setup_train()
            print("Setting up gen")
            self.setup_generate()
            print("Done setting up")

    @property
    def params(self):
        return self.lstmstack.params

    @params.setter
    def params(self, paramlist):
        self.lstmstack.params = paramlist

    def setup_train(self):

        # dimensions: (batch, time, 12)
        chord_types = T.btensor3()

        # dimensions: (batch, time)
        chord_roots = T.imatrix()

        # dimensions: (batch, time)
        relative_pos = T.imatrix()

        # dimesions: (batch, time, output_data)
        encoded_melody = T.btensor3()

        # dimesions: (batch, time)
        correct_notes = T.imatrix()

        n_batch, n_time = relative_pos.shape

        def _build(det_dropout):
            activations = self.lstmstack.do_preprocess_scan( timestep=T.tile(T.arange(n_time), (n_batch,1)) ,
                                                             relative_position=relative_pos,
                                                             cur_chord_type=chord_types,
                                                             cur_chord_root=chord_roots,
                                                             last_output=T.concatenate([T.tile(self.encoding.initial_encoded_form(), (n_batch,1,1)),
                                                                                   encoded_melody[:,:-1,:] ], 1),
                                                             deterministic_dropout=det_dropout)

            out_probs = self.encoding.decode_to_probs(activations, relative_pos, constants.LOW_BOUND, constants.HIGH_BOUND)
            return Encoding.compute_loss(out_probs, correct_notes, True)

        train_loss, train_info = _build(False)
        updates = Adam(train_loss, self.params)

        eval_loss, eval_info = _build(True)

        self.loss_info_keys = list(train_info.keys())

        self.update_fun = theano.function(
            inputs=[chord_types, chord_roots, relative_pos, encoded_melody, correct_notes],
            outputs=[train_loss]+list(train_info.values()),
            updates=updates,
            allow_input_downcast=True)

        self.eval_fun = theano.function(
            inputs=[chord_types, chord_roots, relative_pos, encoded_melody, correct_notes],
            outputs=[eval_loss]+list(eval_info.values()),
            allow_input_downcast=True)

    def _assemble_batch(self, melody, chords):
        encoded_melody = []
        relative_pos = []
        correct_notes = []
        chord_roots = []
        chord_types = []
        for m,c in zip(melody,chords):
            e_m, r_p = self.encoding.encode_melody_and_position(m,c)
            encoded_melody.append(e_m)
            relative_pos.append(r_p)
            correct_notes.append(Encoding.encode_absolute_melody(m, constants.LOW_BOUND, constants.HIGH_BOUND))
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        return (np.array(chord_types, np.float32),
                np.array(chord_roots, np.int32),
                np.array(relative_pos, np.int32),
                np.array(encoded_melody, np.float32),
                np.array(correct_notes, np.int32))

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

        spec = self.lstmstack.prepare_sample_scan(  start_pos=T.alloc(np.array(self.encoding.STARTING_POSITION, np.int32), (n_batch)),
                                                    start_out=T.tile(self.encoding.initial_encoded_form(), (n_batch,1)),
                                                    timestep=T.tile(T.arange(n_time), (n_batch,1)),
                                                    cur_chord_type=chord_types,
                                                    cur_chord_root=chord_roots,
                                                    deterministic_dropout=True )

        def _scan_fn(*inputs):
            # inputs is [ spec_sequences..., last_absolute_position, spec_taps..., spec_non_sequences... ]
            inputs = list(inputs)
            last_absolute_chosen = inputs.pop(len(spec.sequences))
            scan_rout = self.lstmstack.sample_scan_routine(spec, *inputs)

            last_rel_pos, last_out, cur_kwargs = scan_rout.send(None)

            new_pos = self.encoding.get_new_relative_position(last_absolute_chosen, last_rel_pos, last_out, constants.LOW_BOUND, constants.HIGH_BOUND, **cur_kwargs)
            addtl_kwargs = {
                "last_output": last_out
            }

            out_activations = scan_rout.send((new_pos, addtl_kwargs))
            out_probs = self.encoding.decode_to_probs(out_activations,new_pos,constants.LOW_BOUND, constants.HIGH_BOUND)
            sampled_note = Encoding.sample_absolute_probs(self.srng, out_probs)
            encoded_output = self.encoding.note_to_encoding(sampled_note, new_pos, constants.LOW_BOUND, constants.HIGH_BOUND)
            scan_outputs = scan_rout.send(encoded_output)
            scan_rout.close()

            return [sampled_note, out_probs] + scan_outputs

        outputs_info = [{"initial":T.zeros((n_batch,),'int32'), "taps":[-1]}, None] + spec.outputs_info
        result, updates = theano.scan(fn=_scan_fn, sequences=spec.sequences, non_sequences=spec.non_sequences, outputs_info=outputs_info)
        all_chosen = result[0].dimshuffle((1,0))
        all_probs = result[1].dimshuffle((1,0,2))

        self.generate_fun = theano.function(
            inputs=[chord_roots, chord_types],
            updates=updates,
            outputs=all_chosen,
            allow_input_downcast=True)

        self.generate_visualize_fun = theano.function(
            inputs=[chord_roots, chord_types],
            updates=updates,
            outputs=[all_chosen, all_probs],
            allow_input_downcast=True)

    def generate(self, chords):
        assert self.generate_fun is not None, "Need to call setup_generate before generate"

        chord_roots = []
        chord_types = []
        for c in chords:
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        chosen = self.generate_fun(np.array(chord_roots, np.int32),np.array(chord_types, np.float32))
        return [Encoding.decode_absolute_melody(c, constants.LOW_BOUND, constants.HIGH_BOUND) for c in chosen]

    def generate_visualize(self, chords):
        assert self.generate_fun is not None, "Need to call setup_generate before generate"
        chord_roots = []
        chord_types = []
        for c in chords:
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        chosen, all_probs = self.generate_visualize_fun(chord_roots, chord_types)

        melody = [Encoding.decode_absolute_melody(c, constants.LOW_BOUND, constants.HIGH_BOUND) for c in chosen]
        return melody, chosen, all_probs

