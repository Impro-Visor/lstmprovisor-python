
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy as np

import input_parts
from relshift_lstm import RelativeShiftLSTMStack
from adam import Adam

class SimpleModel(object):
    def __init__(self, encoding, layer_sizes, dropout=0, setup=False):

        self.encoding = encoding

        parts = [
            input_parts.BeatInputPart(),
            input_parts.PositionInputPart(encoding.LOW_BOUND, encoding.HIGH_BOUND, 2),
            input_parts.ChordShiftInputPart(),
            input_parts.PassthroughInputPart("last_output", encoding.ENCODING_WIDTH)
        ]
        self.lstmstack = RelativeShiftLSTMStack(parts, layer_sizes, encoding.RAW_ENCODING_WIDTH, encoding.WINDOW_SIZE, dropout)

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
        chords = T.btensor3()

        # dimensions: (batch, time)
        relative_pos = T.imatrix()

        # dimesions: (batch, time, output_data)
        encoded_melody = T.btensor3()

        n_batch, n_time = relative_pos.shape

        def _build(det_dropout):
            activations = self.lstmstack.do_preprocess_scan( timestep=T.tile(T.arange(n_time), (n_batch,1)) ,
                                                             relative_position=relative_pos,
                                                             cur_chord=chords,
                                                             last_output=T.concatenate([T.tile(self.encoding.initial_encoded_form(), (n_batch,1,1)),
                                                                                   encoded_melody[:,:-1,:] ], 1),
                                                             deterministic_dropout=det_dropout)

            out_probs = self.encoding.convert_activations(activations)
            return self.encoding.compute_loss(encoded_melody, out_probs, True)

        train_loss, train_info = _build(False)
        updates = Adam(train_loss, self.params)

        eval_loss, eval_info = _build(True)

        self.loss_info_keys = list(train_info.keys())

        self.update_fun = theano.function(
            inputs=[chords, encoded_melody, relative_pos],
            outputs=[train_loss]+list(train_info.values()),
            updates=updates,
            allow_input_downcast=True)

        self.eval_fun = theano.function(
            inputs=[chords, encoded_melody, relative_pos],
            outputs=[eval_loss]+list(eval_info.values()),
            allow_input_downcast=True)

    def _assemble_batch(self, melody, chords):
        encoded_melody = []
        relative_pos = []
        for m in melody:
            e_m, r_p = self.encoding.encode_melody(m)
            encoded_melody.append(e_m)
            relative_pos.append(r_p)
        return np.array(chords, np.float32), np.array(encoded_melody, np.float32), np.array(relative_pos, np.int32)

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
        chords = T.btensor3()

        n_batch, n_time, _ = chords.shape

        def _sample_fn(out_activations, rel_pos):
            out_probs = self.encoding.convert_activations(out_activations)
            return self.encoding.sample_output(self.srng, rel_pos, out_probs)

        def _out_to_in_fn(rel_pos, last_out, **cur_kwargs):
            return {
                "last_output": last_out
            }

        posns, raw_output, sampled_output, updates = self.lstmstack.do_sample_scan(
                                                        start_pos=T.alloc(np.array(self.encoding.STARTING_POSITION, np.int32), (n_batch)),
                                                        start_out=T.tile(self.encoding.initial_encoded_form(), (n_batch,1)),
                                                        sample_fn=_sample_fn,
                                                        out_to_in_fn=_out_to_in_fn,
                                                        timestep=T.tile(T.arange(n_time), (n_batch,1)),
                                                        cur_chord=chords,
                                                        deterministic_dropout=True )

        self.generate_fun = theano.function(
            inputs=[chords],
            updates=updates,
            outputs=[posns, sampled_output],
            allow_input_downcast=True)

    def generate(self, chords):
        assert self.generate_fun is not None, "Need to call setup_generate before generate"
        posns, outputs = self.generate_fun(chords)
        return [self.encoding.decode_melody(out, pos) for out,pos in zip(outputs, posns)]

