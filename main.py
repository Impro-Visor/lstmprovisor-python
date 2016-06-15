from simple_rel_model import SimpleModel
from note_encodings import RelativeJumpEncoding
import leadsheet
import training
import pickle

import sys
import os

import numpy as np
import relative_data

def main(dataset="dataset", outputdir="output"):
    # (100,10),(100,10)
    # (300,20),(300,20)
    m = SimpleModel(RelativeJumpEncoding(), [(200,10),(200,10)], dropout=0.5, setup=True)

    leadsheets = training.find_leadsheets(dataset)

    training.train(m, leadsheets, 1, outputdir)

    pickle.dump( m.params, open( os.path.join(outputdir, "final_params.p"), "wb" ) )

if __name__ == '__main__':
    np.set_printoptions(edgeitems=20)
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1], sys.argv[2])