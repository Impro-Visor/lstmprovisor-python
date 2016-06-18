import argparse

def main(modeltype, dataset="dataset", outputdir="output", resume=None, check_nan=False, generate=False):
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

    should_setup = not generate
    model_builders = {
        "simple_abs": (lambda:
            SimpleModel(
                AbsoluteSequentialEncoding(constants.LOW_BOUND, constants.HIGH_BOUND),
                [(300,0),(300,0)],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "simple_rel": (lambda:
            SimpleModel(
                RelativeJumpEncoding(),
                [(200,10),(200,10)],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "simple_rel_npn": (lambda:
            SimpleModel(
                RelativeJumpEncoding(),
                [(300,0),(300,0)],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "poex": (lambda:
            ProductOfExpertsModel(
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                ["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "poex_npn": (lambda:
            ProductOfExpertsModel(
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(300,0),(300,0)], [(300,0),(300,0)]],
                ["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "compae_std_abs": (lambda:
            CompressiveAutoencoderModel(
                StandardQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [AbsoluteSequentialEncoding(constants.LOW_BOUND, constants.HIGH_BOUND)],
                [[(300,0),(300,0)]],
                [[(300,0),(300,0)]],
                inputs=[[input_parts.BeatInputPart(),
                  input_parts.ChordShiftInputPart()]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "compae_std_rel": (lambda:
            CompressiveAutoencoderModel(
                StandardQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding()],
                [[(200,10),(200,10)]],
                [[(200,10),(200,10)]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "compae_std_poex": (lambda:
            CompressiveAutoencoderModel(
                StandardQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                shift_modes=["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "compae_var_abs": (lambda:
            CompressiveAutoencoderModel(
                VariationalQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [AbsoluteSequentialEncoding(constants.LOW_BOUND, constants.HIGH_BOUND)],
                [[(300,0),(300,0)]],
                [[(300,0),(300,0)]],
                inputs=[[input_parts.BeatInputPart(),
                  input_parts.ChordShiftInputPart()]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "compae_var_rel": (lambda:
            CompressiveAutoencoderModel(
                VariationalQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding()],
                [[(200,10),(200,10)]],
                [[(200,10),(200,10)]],
                shift_modes=["drop"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
        "compae_var_poex": (lambda:
            CompressiveAutoencoderModel(
                VariationalQueueManager(100, loss_fun=(lambda x: T.log(1+x))),
                [RelativeJumpEncoding(), ChordRelativeEncoding()],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                [[(200,10),(200,10)], [(200,10),(200,10)]],
                shift_modes=["drop","roll"],
                dropout=0.5, setup=should_setup, nanguard=check_nan)),
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
        training.generate(m, leadsheets, os.path.join(outputdir, "generated"), with_vis=True)
    else:
        training.train(m, leadsheets, 50000, outputdir, start_idx)
        pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

parser = argparse.ArgumentParser(description='Train a neural network model.')
parser.add_argument('modeltype', help='Type of model to construct')
parser.add_argument('--dataset', default='dataset', help='Path to dataset folder (with .ls files)')
parser.add_argument('--outputdir', default='output', help='Path to output folder')
parser.add_argument('--check_nan', action='store_true', help='Check for nans during execution')
parser.add_argument('--generate', action='store_true', help="Don't train, just generate. Should be used with restore.")
parser.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='Where to restore from: timestep, and file to load')

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))