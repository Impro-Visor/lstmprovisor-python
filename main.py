import argparse
import time
import sys
import os
import collections

from models import SimpleModel, ProductOfExpertsModel, CompressiveAutoencoderModel
from note_encodings import AbsoluteSequentialEncoding, RelativeJumpEncoding, ChordRelativeEncoding, CircleOfThirdsEncoding
from queue_managers import StandardQueueManager, VariationalQueueManager
import input_parts
import leadsheet
import training
import pickle
import theano
import theano.tensor as T

import numpy as np
import relative_data
import constants

ModelBuilder = collections.namedtuple('ModelBuilder',['name', 'build', 'config_args', 'desc'])
builders = {}

def build_simple(should_setup, check_nan, unroll_batch_num, encode_key, no_per_note):
    if encode_key == "abs":
        enc = AbsoluteSequentialEncoding(constants.BOUNDS.lowbound, constants.BOUNDS.highbound)
        inputs = [input_parts.BeatInputPart(),input_parts.ChordShiftInputPart()]
    elif encode_key == "cot":
        enc = CircleOfThirdsEncoding(constants.BOUNDS.lowbound, (constants.BOUNDS.highbound-constants.BOUNDS.lowbound)//12)
        inputs = [input_parts.BeatInputPart(),input_parts.ChordShiftInputPart()]
    elif encode_key == "rel":
        enc = RelativeJumpEncoding()
        inputs = None
    sizes = [(200,10),(200,10)] if (encode_key == "rel" and not no_per_note) else [(300,0),(300,0)]
    bounds = constants.NoteBounds(48, 84) if encode_key == "cot" else constants.BOUNDS
    return SimpleModel(enc, sizes, bounds=bounds, inputs=inputs, dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)

def config_simple(parser):
    parser.add_argument('encode_key', choices=["abs","cot","rel"], help='Type of encoding to use')
    parser.add_argument('--no_per_note', action="store_true", help='Remove any note memory cells')

builders['simple'] = ModelBuilder('simple', build_simple, config_simple, 'A simple single-LSTM-stack sequential model')

#######################

def build_poex(should_setup, check_nan, unroll_batch_num, no_per_note):
    encs = [RelativeJumpEncoding(), ChordRelativeEncoding()]
    sizes = [[(300,0),(300,0)], [(300,0),(300,0)]] if no_per_note else [[(200,10),(200,10)], [(200,10),(200,10)]]

    return ProductOfExpertsModel(encs, sizes, shift_modes=["drop","roll"],
        dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)

def config_poex(parser):
    parser.add_argument('--no_per_note', action="store_true", help='Remove any note memory cells')

builders['poex'] = ModelBuilder('poex', build_poex, config_poex, 'A product-of-experts LSTM sequential model, using note and chord relative encodings.')

#######################

def build_compae(should_setup, check_nan, unroll_batch_num, encode_key, queue_key, no_per_note, hide_output):
    bounds = constants.NoteBounds(48, 84) if encode_key == "cot" else constants.BOUNDS
    shift_modes = None
    if encode_key == "abs":
        enc = [AbsoluteSequentialEncoding(constants.BOUNDS.lowbound, constants.BOUNDS.highbound)]
        sizes = [[(300,0),(300,0)]]
        inputs = [[input_parts.BeatInputPart(), input_parts.ChordShiftInputPart()]]
    elif encode_key == "cot":
        enc = [CircleOfThirdsEncoding(bounds.lowbound, (bounds.highbound-bounds.lowbound)//12)]
        sizes = [[(300,0),(300,0)]]
        inputs = [[input_parts.BeatInputPart(), input_parts.ChordShiftInputPart()]]
    elif encode_key == "rel":
        enc = [RelativeJumpEncoding()]
        sizes = [[(200,10),(200,10)] if (not no_per_note) else [(300,0),(300,0)]]
        shift_modes=["drop"]
        inputs = None
    elif encode_key == "poex":
        enc = [RelativeJumpEncoding(), ChordRelativeEncoding()]
        sizes = [ [(200,10),(200,10)] if (not no_per_note) else [(300,0),(300,0)] ]*2
        shift_modes=["drop","roll"]
        inputs = None

    if queue_key == "std":
        qman = StandardQueueManager(100, loss_fun=(lambda x: T.log(1+99*x)/T.log(100)))
    elif queue_key == "var":
        qman = VariationalQueueManager(100, loss_fun=(lambda x: T.log(1+99*x)/T.log(100)))


    return CompressiveAutoencoderModel(qman, enc, sizes, sizes, shift_modes=shift_modes, bounds=bounds, hide_output=hide_output, inputs=inputs,
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)

def config_compae(parser):
    parser.add_argument('encode_key', choices=["abs","cot","rel","poex"], help='Type of encoding to use')
    parser.add_argument('queue_key', choices=["std","var"], help='Type of queue manager to use')
    parser.add_argument('--no_per_note', action="store_true", help='Remove any note memory cells')
    parser.add_argument('--hide_output', action="store_true", help='Hide previous outputs from the decoder')

builders['compae'] = ModelBuilder('compae', build_compae, config_compae, 'A compressive autoencoder model.')

###################################################################################################################

def main(modeltype, dataset="dataset", outputdir="output", validation=None, resume=None, check_nan=False, generate=False, generate_over=None, **model_kwargs):
    generate = generate or (generate_over is not None)
    should_setup = not generate
    unroll_batch_num = None if generate else training.BATCH_SIZE

    m = builders[modeltype].build(should_setup, check_nan, unroll_batch_num, **model_kwargs)

    leadsheets = training.find_leadsheets(dataset)

    if resume is not None:
        start_idx, paramfile = resume
        start_idx = int(start_idx)
        m.params = pickle.load( open(paramfile, "rb" ) )
    else:
        start_idx = 0

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    if generate:
        print("Setting up generation")
        m.setup_produce()
        print("Starting to generate")
        start_time = time.process_time()
        if generate_over is not None:
            source, divwidth = generate_over
            if divwidth == 'full':
                divwidth = 0
            elif len(divwidth)>3 and divwidth[-3:] == 'bar':
                divwidth = int(divwidth[:-3])*(constants.WHOLE//constants.RESOLUTION_SCALAR)
            else:
                divwidth = int(divwidth)
            ch,mel = leadsheet.parse_leadsheet(source)
            lslen = leadsheet.get_leadsheet_length(ch,mel)
            if divwidth == 0:
                batch = ([ch],[mel]), source
            else:
                slices = [leadsheet.slice_leadsheet(ch,mel,s,s+divwidth) for s in range(0,lslen,divwidth)]
                batch = list(zip(*slices)), source
            training.generate(m, leadsheets, os.path.join(outputdir, "generated"), with_vis=True, batch=batch)
        else:
            training.generate(m, leadsheets, os.path.join(outputdir, "generated"), with_vis=True)
        end_time = time.process_time()
        print("Generation took {} seconds.".format(end_time-start_time))
    else:
        training.train(m, leadsheets, 50000, outputdir, start_idx, validation_leadsheets=validation)
        pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

parser = argparse.ArgumentParser(description='Train a neural network model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='dataset', help='Path to dataset folder (with .ls files)')
parser.add_argument('--validation', help='Path to validation dataset folder (with .ls files)')
parser.add_argument('--outputdir', default='output', help='Path to output folder')
parser.add_argument('--check_nan', action='store_true', help='Check for nans during execution')
parser.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='Where to restore from: timestep, and file to load')
group = parser.add_mutually_exclusive_group()
group.add_argument('--generate', action='store_true', help="Don't train, just generate. Should be used with restore.")
group.add_argument('--generate_over', nargs=2, metavar=('SOURCE', 'DIV_WIDTH'), default=None, help="Don't train, just generate, and generate over SOURCE chord changes divided into chunks of length DIV_WIDTH (or one contiguous chunk if DIV_WIDTH is 'full'). Can use 'bar' as a unit. Should be used with restore.")

subparsers = parser.add_subparsers(title='Model Types', dest='modeltype', help='Type of model to use. (Note that each model type has additional parameters.)')
for k,b in builders.items():
    cur_parser = subparsers.add_parser(k, help=b.desc)
    b.config_args(cur_parser)

if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    args = vars(parser.parse_args())
    if args["modeltype"] is None:
        parser.print_usage()
    else:
        main(**args)