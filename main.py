import argparse

def main(dataset="dataset", outputdir="output", resume=None, check_nan=False, generate=None, layersizes=None):
    from models import SimpleModel, ProductOfExpertsModel
    from note_encodings import RelativeJumpEncoding, ChordRelativeEncoding
    import leadsheet
    import training
    import pickle

    import sys
    import os

    import numpy as np
    import relative_data

    if layersizes is not None:
        lsizes = eval(layersizes,{},{})
    else:
        lsizes = [[(200,10),(200,10)], [(200,10),(200,10)]]

    m = ProductOfExpertsModel([RelativeJumpEncoding(), ChordRelativeEncoding()], lsizes, ["drop","roll"], dropout=0.5, setup=(generate is None), nanguard=check_nan)

    leadsheets = training.find_leadsheets(dataset)

    if resume is not None:
        start_idx, paramfile = resume
        start_idx = int(start_idx)
        m.params = pickle.load( open(paramfile, "rb" ) )
    else:
        start_idx = 0

    if generate is not None:
        m.setup_generate()
        training.generate(m, leadsheets, generate, with_vis=True)
    else:
        training.train(m, leadsheets, 50000, outputdir, start_idx)
        pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

parser = argparse.ArgumentParser(description='Train a neural network model.')
parser.add_argument('--dataset', default='dataset', help='Path to dataset folder (with .ls files)')
parser.add_argument('--outputdir', default='output', help='Path to output folder')
parser.add_argument('--layersizes', help='Model layer sizes')
parser.add_argument('--check_nan', action='store_true', help='Check for nans during execution')
parser.add_argument('--generate', default=None, metavar='SAMPLEPATH', help="Don't train, just generate. Should be used with restore.")
parser.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='Where to restore from: timestep, and file to load')

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))