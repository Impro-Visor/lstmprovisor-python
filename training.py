import numpy as np
import os
import random
import signal

import leadsheet
import constants

import pickle as pickle

import traceback

from pprint import pformat

BATCH_SIZE = 10
SEGMENT_STEP = constants.WHOLE//constants.RESOLUTION_SCALAR
SEGMENT_LEN = 4*SEGMENT_STEP

GEN_BATCH_SIZE = BATCH_SIZE
GEN_SEGMENT_STEP = SEGMENT_STEP
GEN_SEGMENT_LEN = SEGMENT_LEN

VALIDATION_CT = 5

def find_leadsheets(dirpath):
    return [os.path.join(dirpath, fname) for fname in os.listdir(dirpath) if fname[-3:] == '.ls']

def get_batch(leadsheets, with_sample=False):
    """
    Get a batch

    returns: chords, melodies
    """
    sample_fns = [random.choice(leadsheets) for _ in range(BATCH_SIZE)]
    loaded_samples = [leadsheet.parse_leadsheet(lsfn) for lsfn in sample_fns]
    sample_lengths = [leadsheet.get_leadsheet_length(c,m) for c,m in loaded_samples]

    starts = [(0 if l==SEGMENT_LEN else random.randrange(0,l-SEGMENT_LEN,SEGMENT_STEP)) for l in sample_lengths]
    sliced = [leadsheet.slice_leadsheet(c,m,s,s+SEGMENT_LEN) for (c,m),s in zip(loaded_samples, starts)]

    res = list(zip(*sliced))

    sample_sources = ["{}: starting at {} = bar {}".format(fn, start, start/(constants.WHOLE//constants.RESOLUTION_SCALAR)) for fn,start in zip(sample_fns, starts)]

    if with_sample:
        return res, sample_sources
    else:
        return res

def generate(model, leadsheets, filename, with_vis=False, batch=None):
    if batch is None:
        batch = get_batch(leadsheets, True)
    (chords, melody), sample_sources = batch
    generated_out, chosen, vis_probs, vis_info = model.produce(chords, melody)

    if with_vis:
        with open("{}_sources.txt".format(filename), "w") as f:
            f.write('\n'.join(sample_sources))
        np.save('{}_chosen.npy'.format(filename), chosen)
        np.save('{}_probs.npy'.format(filename), vis_probs)
        for i,v in enumerate(vis_info):
            np.save('{}_info_{}.npy'.format(filename,i), v)
    for samplenum, (melody, chords) in enumerate(zip(generated_out, chords)):
        leadsheet.write_leadsheet(chords, melody, '{}_{}.ls'.format(filename, samplenum))

def validate(model, validation_leadsheets):
    accum_loss = None
    accum_infos = None
    for i in range(VALIDATION_CT):
        loss, infos = model.eval(*get_batch(validation_leadsheets))
        if accum_loss is None:
            accum_loss = loss
            accum_infos = infos
        else:
            accum_loss +=  loss
            for k in accum_info.keys():
                accum_loss[k] += accum_infos[k]
    accum_loss /= VALIDATION_CT
    for k in accum_info.keys():
        accum_loss[k] /= VALIDATION_CT
    return accum_loss, accum_info

def train(model,leadsheets,num_updates,outputdir,start=0,validation_leadsheets=None):
    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
        print("Caught interrupt, waiting until safe. Press again to force terminate")
        signal.signal(signal.SIGINT, old_handler)
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start+1,start+num_updates+1):
        if stopflag[0]:
            break
        loss, infos = model.train(*get_batch(leadsheets))
        with open(os.path.join(outputdir,'data.csv'),'a') as f:
            if i == 1:
                f.write("iter, loss, " + ", ".join(k for k,v in sorted(infos.items())) + "\n")
            f.write("{}, {}, ".format(i,loss) + ", ".join(str(v) for k,v in sorted(infos.items())) + "\n")
        if i % 10 == 0:
            print("update {}: {}, info {}".format(i,loss,pformat(infos)))
        if i % 5000 == 0 or i==1:
            generate(model, leadsheets, os.path.join(outputdir,'sample{}'.format(i)))
            pickle.dump(model.params,open(os.path.join(outputdir, 'params{}.p'.format(i)), 'wb'))
            if validation_leadsheets is not None:
                val_loss, val_infos = validate(model, validation_leadsheets)
                print("Validation on {}: {}, info {}".format(i,val_loss,pformat(val_infos)))
                with open(os.path.join(outputdir,'valid_data.csv'),'a') as f:
                    if i == 1:
                        f.write("iter, loss, " + ", ".join(k for k,v in sorted(val_infos.items())) + "\n")
                    f.write("{}, {}, ".format(i,val_loss) + ", ".join(str(v) for k,v in sorted(val_infos.items())) + "\n")
    if not stopflag[0]:
        signal.signal(signal.SIGINT, old_handler)