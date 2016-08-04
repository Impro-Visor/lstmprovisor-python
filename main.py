import argparse
import time
import sys
import os
import collections

from models import SimpleModel, ProductOfExpertsModel, CompressiveAutoencoderModel
from note_encodings import AbsoluteSequentialEncoding, RelativeJumpEncoding, ChordRelativeEncoding, CircleOfThirdsEncoding
from queue_managers import StandardQueueManager, VariationalQueueManager, SamplingVariationalQueueManager, QueuelessVariationalQueueManager, QueuelessStandardQueueManager, NearnessStandardQueueManager, NoiseWrapper
import input_parts
import leadsheet
import training
import pickle
import theano
import theano.tensor as T

import numpy as np
import constants
from util import sliceMaker

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
    parser.add_argument('--per_note', dest="no_per_note", action="store_false", help='Enable note memory cells')

builders['simple'] = ModelBuilder('simple', build_simple, config_simple, 'A simple single-LSTM-stack sequential model')

#######################

def build_poex(should_setup, check_nan, unroll_batch_num, no_per_note, layer_size, num_layers):
    encs = [RelativeJumpEncoding(), ChordRelativeEncoding()]
    sizes = [[(layer_size,0)]*num_layers]*2 if no_per_note else [[(200,10),(200,10)], [(200,10),(200,10)]]

    return ProductOfExpertsModel(encs, sizes, shift_modes=["drop","roll"],
        dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)

def config_poex(parser):
    parser.add_argument('--per_note', dest="no_per_note", action="store_false", help='Enable note memory cells')
    parser.add_argument('--layer_size', type=int, default=300, help='Layer size of the LSTMs. Only works without note memory cells')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers. Only works without note memory cells')

builders['poex'] = ModelBuilder('poex', build_poex, config_poex, 'A product-of-experts LSTM sequential model, using note and chord relative encodings.')

#######################

def build_compae(should_setup, check_nan, unroll_batch_num, encode_key, queue_key, no_per_note, layer_size, num_layers, feature_size, hide_output, sparsity_loss_scale, variational_loss_scale, train_decoder_only, feature_period=None, add_pre_noise=None, add_post_noise=None, loss_mode_priority=False, loss_mode_add=False, loss_mode_cutoff=None, loss_mode_trigger=None):
    bounds = constants.NoteBounds(48, 84) if encode_key == "cot" else constants.BOUNDS
    shift_modes = None
    if encode_key == "abs":
        enc = [AbsoluteSequentialEncoding(constants.BOUNDS.lowbound, constants.BOUNDS.highbound)]
        sizes = [[(layer_size,0)]*num_layers]
        inputs = [[input_parts.BeatInputPart(), input_parts.ChordShiftInputPart()]]
    elif encode_key == "cot":
        enc = [CircleOfThirdsEncoding(bounds.lowbound, (bounds.highbound-bounds.lowbound)//12)]
        sizes = [[(layer_size,0)]*num_layers]
        inputs = [[input_parts.BeatInputPart(), input_parts.ChordShiftInputPart()]]
    elif encode_key == "rel":
        enc = [RelativeJumpEncoding()]
        sizes = [[(200,10),(200,10)] if (not no_per_note) else [[(layer_size,0)]*num_layers]]
        shift_modes=["drop"]
        inputs = None
    elif encode_key == "poex":
        enc = [RelativeJumpEncoding(), ChordRelativeEncoding()]
        sizes = [ [(200,10),(200,10)] if (not no_per_note) else [[(layer_size,0)]*num_layers] ]*2
        shift_modes=["drop","roll"]
        inputs = None

    unscaled_loss_fun = lambda x: T.log(1+99*x)/T.log(100)
    lossfun = lambda x: np.array(sparsity_loss_scale, np.float32) * unscaled_loss_fun(x)
    if queue_key == "std":
        qman = StandardQueueManager(feature_size, loss_fun=lossfun)
    elif queue_key == "var":
        qman = VariationalQueueManager(feature_size, loss_fun=lossfun, variational_loss_scale=variational_loss_scale)
    elif queue_key == "sample_var":
        qman = SamplingVariationalQueueManager(feature_size, loss_fun=lossfun, variational_loss_scale=variational_loss_scale)
    elif queue_key == "queueless_var":
        qman = QueuelessVariationalQueueManager(feature_size, period=feature_period, variational_loss_scale=variational_loss_scale)
    elif queue_key == "queueless_std":
        qman = QueuelessStandardQueueManager(feature_size, period=feature_period)
    elif queue_key == "nearness_std":
        qman = NearnessStandardQueueManager(feature_size, sparsity_loss_scale*10, sparsity_loss_scale, 0.97, loss_fun=unscaled_loss_fun)

    if add_pre_noise is not None or add_post_noise is not None:
        if "queueless" in queue_key:
            pre_mask = sliceMaker[:]
        else:
            pre_mask = sliceMaker[1:]
        qman = NoiseWrapper(qman, add_pre_noise, add_post_noise, pre_mask)

    loss_mode = "add" if loss_mode_add else \
                ("cutoff", loss_mode_cutoff) if loss_mode_cutoff is not None else \
                ("trigger",)+tuple(loss_mode_trigger) if loss_mode_trigger is not None else \
                ("priority", loss_mode_priority if loss_mode_priority is not None else 50)

    return CompressiveAutoencoderModel(qman, enc, sizes, sizes, shift_modes=shift_modes, bounds=bounds, hide_output=hide_output, inputs=inputs,
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num, loss_mode=loss_mode, train_decoder_only=train_decoder_only)

def config_compae(parser):
    parser.add_argument('encode_key', choices=["abs","cot","rel","poex"], help='Type of encoding to use')
    parser.add_argument('queue_key', choices=["std","var","sample_var","queueless_var","queueless_std","nearness_std"], help='Type of queue manager to use')
    parser.add_argument('--per_note', dest="no_per_note", action="store_false", help='Enable note memory cells')
    parser.add_argument('--hide_output', action="store_true", help='Hide previous outputs from the decoder')
    parser.add_argument('--sparsity_loss_scale', type=float, default="1", help='How much to scale the sparsity loss by')
    parser.add_argument('--variational_loss_scale', type=float, default="1", help='How much to scale the variational loss by')
    parser.add_argument('--feature_size', type=int, default="100", help='Size of feature vectors')
    parser.add_argument('--feature_period', type=int, help='If in queueless mode, period of features in timesteps')
    parser.add_argument('--add_pre_noise', type=float, nargs="?", const=1.0, help='Add Gaussian noise to the feature values before applying the activation function')
    parser.add_argument('--add_post_noise', type=float, nargs="?", const=1.0, help='Add Gaussian noise to the feature values after applying the activation function')
    parser.add_argument('--train_decoder_only', action="store_true", help='Only modify the decoder parameters')
    parser.add_argument('--layer_size', type=int, default=300, help='Layer size of the LSTMs. Only works without note memory cells')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers. Only works without note memory cells')
    lossgroup = parser.add_mutually_exclusive_group()
    lossgroup.add_argument('--priority_loss', nargs='?', const=50, dest='loss_mode_priority', type=float, help='Use priority loss scaling mode (with the specified curviness)')
    lossgroup.add_argument('--add_loss', dest='loss_mode_add', action='store_true', help='Use adding loss scaling mode')
    lossgroup.add_argument('--cutoff_loss', dest='loss_mode_cutoff', type=float, metavar="CUTOFF", help='Use cutoff loss scaling mode with the specified per-batch cutoff')
    lossgroup.add_argument('--trigger_loss', dest='loss_mode_trigger', nargs=2, type=float, metavar=("TRIGGER", "RAMP_TIME"), help='Use trigger loss scaling mode with the specified per-batch trigger value and desired ramp-up time')

builders['compae'] = ModelBuilder('compae', build_compae, config_compae, 'A compressive autoencoder model.')

###################################################################################################################

def main(modeltype, batch_size, iterations, segment_len, segment_step, dataset=["dataset"], outputdir="output", validation=None, resume=None, resume_auto=False, check_nan=False, generate=False, generate_over=None, **model_kwargs):
    generate = generate or (generate_over is not None)
    should_setup = not generate
    unroll_batch_num = None if generate else training.BATCH_SIZE

    if generate_over is None:
        training.set_params(batch_size, segment_step, segment_len)
        leadsheets = [training.filter_leadsheets(training.find_leadsheets(d)) for d in dataset]
    else:
        # Don't bother loading leadsheets, we don't need them
        leadsheets = []

    m = builders[modeltype].build(should_setup, check_nan, unroll_batch_num, **model_kwargs)

    if resume_auto:
        paramfile = os.path.join(outputdir,'final_params.p')
        if os.path.isfile(paramfile):
            with open(os.path.join(outputdir,'data.csv'), 'r') as f:
                for line in f:
                    pass
                lastline = line
                start_idx = lastline.split(',')[0]
            print("Automatically resuming from {} after iteration {}.".format(paramfile, start_idx))
            resume = (start_idx, paramfile)
        else:
            print("Didn't find anything to resume. Starting from the beginning...")

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
            elif divwidth == 'debug_firststep':
                divwidth = -1
            elif len(divwidth)>3 and divwidth[-3:] == 'bar':
                divwidth = int(divwidth[:-3])*(constants.WHOLE//constants.RESOLUTION_SCALAR)
            else:
                divwidth = int(divwidth)
            ch,mel = leadsheet.parse_leadsheet(source)
            lslen = leadsheet.get_leadsheet_length(ch,mel)
            if divwidth == 0:
                batch = ([ch],[mel]), [source]
            elif divwidth == -1:
                slices = [leadsheet.slice_leadsheet(ch,mel,0,1)]
                batch = list(zip(*slices)), [source]
            else:
                slices = [leadsheet.slice_leadsheet(ch,mel,s,s+divwidth) for s in range(0,lslen,divwidth)]
                batch = list(zip(*slices)), [source]
            training.generate(m, leadsheets, os.path.join(outputdir, "generated"), with_vis=True, batch=batch)
        else:
            training.generate(m, leadsheets, os.path.join(outputdir, "generated"), with_vis=True)
        end_time = time.process_time()
        print("Generation took {} seconds.".format(end_time-start_time))
    else:
        training.train(m, leadsheets, iterations, outputdir, start_idx, validation_leadsheets=validation)
        pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

def cvt_time(s):
    if len(s)>3 and s[-3:] == "bar":
        return int(s[:-3])*(constants.WHOLE//constants.RESOLUTION_SCALAR)
    else:
        return int(s)

parser = argparse.ArgumentParser(description='Train a neural network model.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', nargs="+", default=['dataset'], help='Path(s) to dataset folder (with .ls files). If multiple are passed, samples randomly from each')
parser.add_argument('--validation', help='Path to validation dataset folder (with .ls files)')
parser.add_argument('--outputdir', default='output', help='Path to output folder')
parser.add_argument('--check_nan', action='store_true', help='Check for nans during execution')
parser.add_argument('--batch_size', type=int, default=10, help='Size of batch')
parser.add_argument('--iterations', type=int, default=50000, help='How many iterations to train')
parser.add_argument('--segment_len', type=cvt_time, default="4bar", help='Length of segment to train on')
parser.add_argument('--segment_step', type=cvt_time, default="1bar", help='Period at which segments may begin')
resume_group = parser.add_mutually_exclusive_group()
resume_group.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='Where to restore from: timestep, and file to load')
resume_group.add_argument('--resume_auto', action='store_true', help='Automatically restore from a previous run using output directory')
gen_group = parser.add_mutually_exclusive_group()
gen_group.add_argument('--generate', action='store_true', help="Don't train, just generate. Should be used with restore.")
gen_group.add_argument('--generate_over', nargs=2, metavar=('SOURCE', 'DIV_WIDTH'), default=None, help="Don't train, just generate, and generate over SOURCE chord changes divided into chunks of length DIV_WIDTH (or one contiguous chunk if DIV_WIDTH is 'full'). Can use 'bar' as a unit. Should be used with restore.")

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
