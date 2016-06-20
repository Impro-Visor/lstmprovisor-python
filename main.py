import argparse

def main(modeltype, dataset="dataset", outputdir="output", resume=None, check_nan=False, generate=False, generate_over=None):
    from models import SimpleModel, ProductOfExpertsModel, CompressiveAutoencoderModel
    from note_encodings import AbsoluteSequentialEncoding, RelativeJumpEncoding, ChordRelativeEncoding
    from queue_managers import StandardQueueManager, VariationalQueueManager
    import input_parts
    import leadsheet
    import training
    import pickle
    import theano
    import theano.tensor as T

    import sys
    import os

    import numpy as np
    import relative_data
    import constants

    generate = generate or (generate_over is not None)
    should_setup = not generate
    unroll_batch_num = None if generate else training.BATCH_SIZE
    model_builders = {
        "simple_abs": (lambda:
            SimpleModel(
                AbsoluteSequentialEncoding(constants.LOW_BOUND, constants.HIGH_BOUND),
                [(300,0),(300,0)],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "simple_rel": (lambda:
            SimpleModel(
                RelativeJumpEncoding(),
                [(200,10),(200,10)],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "simple_rel_npn": (lambda:
            SimpleModel(
                RelativeJumpEncoding(),
                [(300,0),(300,0)],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "poex": (lambda:
            ProductOfExpertsModel(
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                shift_modes=["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "poex_npn": (lambda:
            ProductOfExpertsModel(
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(300,0),(300,0)], [(300,0),(300,0)]],
                shift_modes=["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "compae_std_abs": (lambda:
            CompressiveAutoencoderModel(
                StandardQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [AbsoluteSequentialEncoding(constants.LOW_BOUND, constants.HIGH_BOUND)],
                [[(300,0),(300,0)]],
                [[(300,0),(300,0)]],
                inputs=[[input_parts.BeatInputPart(),
                  input_parts.ChordShiftInputPart()]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "compae_std_rel": (lambda:
            CompressiveAutoencoderModel(
                StandardQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding()],
                [[(200,10),(200,10)]],
                [[(200,10),(200,10)]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "compae_std_poex": (lambda:
            CompressiveAutoencoderModel(
                StandardQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                shift_modes=["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "compae_var_abs": (lambda:
            CompressiveAutoencoderModel(
                VariationalQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [AbsoluteSequentialEncoding(constants.LOW_BOUND, constants.HIGH_BOUND)],
                [[(300,0),(300,0)]],
                [[(300,0),(300,0)]],
                inputs=[[input_parts.BeatInputPart(),
                  input_parts.ChordShiftInputPart()]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "compae_var_rel": (lambda:
            CompressiveAutoencoderModel(
                VariationalQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding()],
                [[(200,10),(200,10)]],
                [[(200,10),(200,10)]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
        "compae_var_poex": (lambda:
            CompressiveAutoencoderModel(
                VariationalQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                shift_modes=["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan, unroll_batch_num=unroll_batch_num)),
    }
    assert modeltype in model_builders, "{} is not a valid model. Try one of {}".format(modeltype, list(model_builders.keys()))
    m = model_builders[modeltype]()

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
        m.setup_generate()
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
    else:
        training.train(m, leadsheets, 50000, outputdir, start_idx)
        pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

parser = argparse.ArgumentParser(description='Train a neural network model.')
parser.add_argument('modeltype', help='Type of model to construct')
parser.add_argument('--dataset', default='dataset', help='Path to dataset folder (with .ls files)')
parser.add_argument('--outputdir', default='output', help='Path to output folder')
parser.add_argument('--check_nan', action='store_true', help='Check for nans during execution')
parser.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='Where to restore from: timestep, and file to load')
group = parser.add_mutually_exclusive_group()
group.add_argument('--generate', action='store_true', help="Don't train, just generate. Should be used with restore.")
group.add_argument('--generate_over', nargs=2, metavar=('SOURCE', 'DIV_WIDTH'), default=None, help="Don't train, just generate, and generate over SOURCE chord changes divided into chunks of length DIV_WIDTH (or one contiguous chunk if DIV_WIDTH is 'full'). Can use 'bar' as a unit. Should be used with restore.")

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))