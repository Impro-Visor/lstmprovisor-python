import argparse

def main(dataset="dataset", outputdir="output", resume=None, check_nan=False):
    from models import SimpleModel, ProductOfExpertsModel
    from note_encodings import RelativeJumpEncoding, ChordRelativeEncoding
    import leadsheet
    import training
    import pickle

    import sys
    import os

    import numpy as np
    import relative_data

    # (100,10),(100,10)
    # (300,20),(300,20)
    m = ProductOfExpertsModel([RelativeJumpEncoding(), ChordRelativeEncoding()], [[(200,10),(200,10)], [(200,10),(200,10)]], ["drop","roll"], dropout=0.5, setup=True, nanguard=check_nan)

    leadsheets = training.find_leadsheets(dataset)

    if resume is not None:
        start_idx, paramfile = resume
        start_idx = int(start_idx)
        m.params = pickle.load( open(paramfile, "rb" ) )
    else:
        start_idx = 0

    training.train(m, leadsheets, 50000, outputdir, start_idx)

    pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

parser = argparse.ArgumentParser(description='Train a neural network model.')
parser.add_argument('--dataset', default='dataset', help='path to dataset folder (with .ls files)')
parser.add_argument('--outputdir', default='output', help='path to output folder')
parser.add_argument('--check_nan', action='store_true', help='check for nans during execution')
parser.add_argument('--resume', nargs=2, metavar=('TIMESTEP', 'PARAMFILE'), default=None, help='timestep to restore from, and file to load')

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))