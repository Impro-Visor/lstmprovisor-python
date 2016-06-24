import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

import numpy as np

import constants
import input_parts
from relshift_lstm import RelativeShiftLSTMStack
from queue_managers import QueueManager
from adam import Adam
from note_encodings import Encoding
from .product_model import helper_generate_from_spec
import leadsheet

import itertools
import functools

from theano.compile.nanguardmode import NanGuardMode


class CompressiveAutoencoderModel( object ):
    def __init__(self, queue_manager, encodings, enc_layer_sizes, dec_layer_sizes, inputs=None, shift_modes=None, dropout=0, setup=False, nanguard=False, loss_mode="priority", hide_output=True, unroll_batch_num=None, bounds=constants.BOUNDS):

        self.bounds = bounds

        self.qman = queue_manager

        self.encodings = encodings
        if shift_modes is None:
            shift_modes = ["drop"]*len(encodings)

        if inputs is None:
            inputs = [[
                input_parts.BeatInputPart(),
                input_parts.PositionInputPart(self.bounds.lowbound, self.bounds.highbound, 2),
                input_parts.ChordShiftInputPart()]]*len(self.encodings)

        self.enc_layer_sizes = enc_layer_sizes
        self.dec_layer_sizes = dec_layer_sizes
        self.enc_lstmstacks = []
        self.dec_lstmstacks = []
        for enc_layer_sizes, dec_layer_sizes, encoding, shift_mode, ipt in zip(enc_layer_sizes,dec_layer_sizes,encodings,shift_modes,inputs):
            enc_parts = ipt + [
                input_parts.PassthroughInputPart("cur_input", encoding.ENCODING_WIDTH)
            ]
            enc_lstmstack = RelativeShiftLSTMStack(enc_parts, enc_layer_sizes, self.qman.activation_width, encoding.WINDOW_SIZE, dropout, mode=shift_mode, unroll_batch_num=unroll_batch_num)
            self.enc_lstmstacks.append(enc_lstmstack)

            dec_parts = ipt + [ input_parts.PassthroughInputPart("cur_feature", self.qman.feature_size) ]
            if not hide_output:
                dec_parts.append(input_parts.PassthroughInputPart("last_output", encoding.ENCODING_WIDTH))
            dec_lstmstack = RelativeShiftLSTMStack(dec_parts, dec_layer_sizes, encoding.RAW_ENCODING_WIDTH, encoding.WINDOW_SIZE, dropout, mode=shift_mode, unroll_batch_num=unroll_batch_num)
            self.dec_lstmstacks.append(dec_lstmstack)

        self.srng = MRG_RandomStreams(np.random.randint(1, 1024))

        self.update_fun = None
        self.eval_fun = None
        self.enc_fun = None
        self.dec_fun = None

        self.nanguard = nanguard

        assert loss_mode in ["priority","add","cutoff"], "Invalid loss mode {}".format(loss_mode)
        self.loss_mode = loss_mode
        if setup:
            print("Setting up train")
            self.setup_train()
            print("Setting up encode")
            self.setup_encode()
            print("Setting up decode")
            self.setup_decode()
            print("Done setting up")

    @property
    def params(self):
        return list(itertools.chain(*(lstmstack.params for stackgroup in (self.enc_lstmstacks, self.dec_lstmstacks) for lstmstack in stackgroup)))

    @params.setter
    def params(self, paramlist):
        mycopy = list(paramlist)
        for stackgroup in (self.enc_lstmstacks, self.dec_lstmstacks):
            for lstmstack in stackgroup:
                lstmstack.params = mycopy[:len(lstmstack.params)]
                del mycopy[:len(lstmstack.params)]
        assert len(mycopy) == 0

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
            all_activations = []
            for encoding, enc_lstmstack, encoded_melody, relative_pos in zip(self.encodings, self.enc_lstmstacks, encoded_melodies, relative_posns):
                activations = enc_lstmstack.do_preprocess_scan( timestep=T.tile(T.arange(n_time), (n_batch,1)) ,
                                                            relative_position=relative_pos,
                                                            cur_chord_type=chord_types,
                                                            cur_chord_root=chord_roots,
                                                            cur_input=encoded_melody,
                                                            deterministic_dropout=det_dropout)
                all_activations.append(activations)
            reduced_activations = functools.reduce((lambda x,y: x+y), all_activations)
            queue_loss, feat_strengths, feat_vects, queue_info = self.qman.process(reduced_activations, extra_info=True)
            features = QueueManager.queue_transform(feat_strengths, feat_vects)

            all_out_probs = []
            for encoding, dec_lstmstack, encoded_melody, relative_pos in zip(self.encodings, self.dec_lstmstacks, encoded_melodies, relative_posns):
                activations = dec_lstmstack.do_preprocess_scan( timestep=T.tile(T.arange(n_time), (n_batch,1)) ,
                                                            relative_position=relative_pos,
                                                            cur_chord_type=chord_types,
                                                            cur_chord_root=chord_roots,
                                                            cur_feature=features,
                                                            last_output=T.concatenate([T.tile(encoding.initial_encoded_form(), (n_batch,1,1)),
                                                                                encoded_melody[:,:-1,:] ], 1),
                                                            deterministic_dropout=det_dropout)
                out_probs = encoding.decode_to_probs(activations, relative_pos, self.bounds.lowbound, self.bounds.highbound)
                all_out_probs.append(out_probs)

            reduced_out_probs = functools.reduce((lambda x,y: x*y), all_out_probs)
            normsum = T.sum(reduced_out_probs, 2, keepdims=True)
            normsum = T.maximum(normsum, constants.EPSILON)
            norm_out_probs = reduced_out_probs/normsum
            reconstruction_loss, reconstruction_info = Encoding.compute_loss(norm_out_probs, correct_notes, extra_info=True)

            queue_surrogate_loss_parts = self.qman.surrogate_loss(reconstruction_loss, queue_info)

            if self.loss_mode is "add":
                full_loss = queue_loss + reconstruction_loss
            elif self.loss_mode is "priority":
                full_loss = reconstruction_loss + queue_loss/(1+theano.gradient.disconnected_grad(reconstruction_loss))
            elif self.loss_mode is "cutoff":
                full_loss = T.switch(reconstruction_loss<1, reconstruction_loss+queue_loss, reconstruction_loss)

            full_info = queue_info.copy()
            full_info.update(reconstruction_info)
            full_info["queue_loss"] = queue_loss
            full_info["reconstruction_loss"] = reconstruction_loss

            updates = []
            if queue_surrogate_loss_parts is not None:
                surrogate_loss, addtl_updates = queue_surrogate_loss_parts
                full_loss = full_loss + surrogate_loss
                updates.extend(addtl_updates)
                full_info["surrogate_loss"] = surrogate_loss

            return full_loss, full_info, updates

        train_loss, train_info, train_updates = _build(False)
        adam_updates = Adam(train_loss, self.params)

        eval_loss, eval_info, _ = _build(True)

        self.loss_info_keys = list(train_info.keys())

        self.update_fun = theano.function(
            inputs=[chord_types, chord_roots, correct_notes] + relative_posns + encoded_melodies,
            outputs=[train_loss]+list(train_info.values()),
            updates=train_updates+adam_updates,
            allow_input_downcast=True,
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

        self.eval_fun = theano.function(
            inputs=[chord_types, chord_roots, correct_notes] + relative_posns + encoded_melodies,
            outputs=[eval_loss]+list(eval_info.values()),
            allow_input_downcast=True,
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

    def _assemble_batch(self, melody, chords, with_correct=True):
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

        retlist = [np.array(chord_types, np.float32),
                    np.array(chord_roots, np.int32)]
        if with_correct:
            retlist.append(np.array(correct_notes, np.int32))
        retlist.extend(np.array(x, np.int32) for x in relative_posns)
        retlist.extend(np.array(x, np.int32) for x in encoded_melodies)
        return retlist

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

    def setup_encode(self):

        # dimensions: (batch, time, 12)
        chord_types = T.btensor3()
        # dimensions: (batch, time)
        chord_roots = T.imatrix()
        # dimensions: (batch, time)
        relative_posns = [T.imatrix() for _ in self.encodings]
        # dimesions: (batch, time, output_data)
        encoded_melodies = [T.btensor3() for _ in self.encodings]
        n_batch, n_time = chord_roots.shape

        all_activations = []
        for encoding, enc_lstmstack, encoded_melody, relative_pos in zip(self.encodings, self.enc_lstmstacks, encoded_melodies, relative_posns):
            activations = enc_lstmstack.do_preprocess_scan( timestep=T.tile(T.arange(n_time), (n_batch,1)) ,
                                                        relative_position=relative_pos,
                                                        cur_chord_type=chord_types,
                                                        cur_chord_root=chord_roots,
                                                        cur_input=encoded_melody,
                                                        deterministic_dropout=True )
            all_activations.append(activations)
        reduced_activations = functools.reduce((lambda x,y: x+y), all_activations)
        strengths, vects = self.qman.get_strengths_and_vects(reduced_activations)

        self.encode_fun = theano.function(
            inputs=[chord_types, chord_roots] + relative_posns + encoded_melodies,
            outputs=[strengths, vects],
            allow_input_downcast=True,
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

    def encode(self, chords, melody):
        assert self.encode_fun is not None, "Need to call setup_encode before encode"
        strengths, vects = self.encode_fun(*self._assemble_batch(melody,chords,False))
        return strengths, vects

    def setup_decode(self):

        # dimensions: (batch, time, 12)
        chord_types = T.btensor3()
        # dimensions: (batch, time)
        chord_roots = T.imatrix()
        # dimensions: (batch, time)
        feat_strengths = T.fmatrix()
        # dimensions: (batch, time, feature_size)
        feat_vects = T.ftensor3()
        n_batch, n_time = chord_roots.shape

        features = QueueManager.queue_transform(feat_strengths, feat_vects)

        specs = [lstmstack.prepare_sample_scan(  start_pos=T.alloc(np.array(encoding.STARTING_POSITION, np.int32), (n_batch)),
                                                    start_out=T.tile(encoding.initial_encoded_form(), (n_batch,1)),
                                                    timestep=T.tile(T.arange(n_time), (n_batch,1)),
                                                    cur_chord_type=chord_types,
                                                    cur_chord_root=chord_roots,
                                                    cur_feature=features,
                                                    deterministic_dropout=True )
                    for lstmstack, encoding in zip(self.dec_lstmstacks, self.encodings)]

        updates, all_chosen, all_probs, indiv_probs = helper_generate_from_spec(specs, self.dec_lstmstacks, self.encodings, self.srng, n_batch, n_time, self.bounds)

        self.decode_fun = theano.function(
            inputs=[chord_roots, chord_types, feat_strengths, feat_vects],
            updates=updates,
            outputs=all_chosen,
            allow_input_downcast=True,
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

        self.decode_visualize_fun = theano.function(
            inputs=[chord_roots, chord_types, feat_strengths, feat_vects],
            updates=updates,
            outputs=[all_chosen, all_probs] + indiv_probs,
            allow_input_downcast=True,
            mode=(NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True) if self.nanguard else None))

    def decode(self, chords, feat_strengths, feat_vects):
        assert self.decode_fun is not None, "Need to call setup_decode before decode"
        chord_roots = []
        chord_types = []
        for c in chords:
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        chosen = self.decode_fun(np.array(chord_roots, np.int32), np.array(chord_types, np.float32), feat_strengths, feat_vects)
        return [Encoding.decode_absolute_melody(c, self.bounds.lowbound, self.bounds.highbound) for c in chosen]

    def decode_visualize(self, chords, feat_strengths, feat_vects):
        assert self.decode_visualize_fun is not None, "Need to call setup_decode before decode_visualize"
        chord_roots = []
        chord_types = []
        for c in chords:
            c_roots, c_types = zip(*c)
            chord_roots.append(c_roots)
            chord_types.append(c_types)
        stuff = self.decode_visualize_fun(np.array(chord_roots, np.int32), np.array(chord_types, np.float32), feat_strengths, feat_vects)
        chosen, all_probs = stuff[:2]

        melody = [Encoding.decode_absolute_melody(c, self.bounds.lowbound, self.bounds.highbound) for c in chosen]
        return melody, chosen, all_probs, stuff[2:]

    def setup_produce(self):
        self.setup_encode()
        self.setup_decode()

    def produce(self, chords, melody):
        strengths, vects = self.encode(chords, melody)
        melody, chosen, all_probs, all_info = self.decode_visualize(chords, strengths, vects)
        return melody, chosen, all_probs, (all_info + [strengths, vects])
