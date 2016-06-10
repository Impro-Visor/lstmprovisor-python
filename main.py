import model
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
    m = model.Model([(200,10),(200,10)], dropout=0.5, setup=True)

    leadsheets = training.find_leadsheets(dataset)
    leadsheets = training.check_leadsheets(leadsheets)

    training.train(m, leadsheets, 100000, outputdir)

    pickle.dump( m.learned_config, open( os.path.join(outputdir, "final_learned_config.p"), "wb" ) )

def gentest():

    m = model.Model([(100,10),(100,10)], dropout=0.5)
    m.setup_generate()

    leadsheets = training.find_leadsheets("dataset")
    leadsheets = training.check_leadsheets(leadsheets)

    chords = training.get_chords(leadsheets)
    generated_out = m.generate_fun(chords)
    for samplenum, (out, chords) in enumerate(zip((generated_out != 0).astype(np.int8).tolist(), (chords != 0).astype(np.int8).tolist())):
        melody = relative_data.output_form_to_melody(out)
        leadsheet.write_leadsheet(chords, melody, 'output/sample{}_{}.ls'.format(0, samplenum))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        main(sys.argv[1], sys.argv[2])