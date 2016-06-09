import numpy as np
import os
import random
import signal

import leadsheet
import model
import constants
import relative_data

import pickle as pickle

BATCH_SIZE = 10
SEGMENT_STEP = constants.WHOLE//constants.RESOLUTION_SCALAR
SEGMENT_LEN = 4*SEGMENT_STEP

GEN_BATCH_SIZE = BATCH_SIZE
GEN_SEGMENT_STEP = SEGMENT_STEP
GEN_SEGMENT_LEN = SEGMENT_LEN

def find_leadsheets(dirpath):
    return [os.path.join(dirpath, fname) for fname in os.listdir(dirpath) if fname[-3:] == '.ls']

def check_leadsheets(leadsheets):
    good = []
    for lsfn in leadsheets:
        try:
            relative_data.melody_to_network_form(*leadsheet.parse_leadsheet(lsfn))
            good.append(lsfn)
        except relative_data.NoteOutOfRangeException as e:
            print("In ", lsfn)
            print(e.args[0])
    print("Found {} good out of {}.".format(len(good), len(leadsheets)))
    return good

def get_batch(leadsheets):
    sample_fns = random.sample(leadsheets, BATCH_SIZE)
    loaded_samples = [leadsheet.parse_leadsheet(lsfn) for lsfn in sample_fns]

    network_input = [relative_data.melody_to_network_form(c,m) for c,m in loaded_samples]
    input_form, mem_shifts, output_form = list(zip(*network_input))

    sample_lengths = [len(x) for x in mem_shifts]
    # print sample_lengths

    starts = [(0 if l==SEGMENT_LEN else random.randrange(0,l-SEGMENT_LEN,SEGMENT_STEP)) for l in sample_lengths]

    segment_ipt = [x[s:s+SEGMENT_LEN] for x,s in zip(input_form, starts)]
    segment_shifts = [x[s:s+SEGMENT_LEN] for x,s in zip(mem_shifts, starts)]
    segment_opt = [x[s:s+SEGMENT_LEN] for x,s in zip(output_form, starts)]


    segment_ipt = np.array(segment_ipt,np.float32)
    segment_shifts = np.array(segment_shifts,np.int32)
    segment_opt = np.array(segment_opt,np.float32)

    return segment_ipt, segment_shifts, segment_opt

def get_chords(leadsheets):
    sample_fns = random.sample(leadsheets, GEN_BATCH_SIZE)
    loaded_samples = [leadsheet.parse_leadsheet(lsfn) for lsfn in sample_fns]

    chords = [c for c,m in loaded_samples]
    sample_lengths = [len(x) for x in chords]
    starts = [(0 if l==GEN_SEGMENT_LEN else random.randrange(0,l-GEN_SEGMENT_LEN,GEN_SEGMENT_STEP)) for l in sample_lengths]

    chords_input = [x[s:s+SEGMENT_LEN] for x,s in zip(chords, starts)]

    return np.array(chords_input, np.float32)

def train(model,leadsheets,num_updates,start=0):
    stopflag = [False]
    def signal_handler(signame, sf):
        stopflag[0] = True
        print("Caught interrupt, waiting until safe. Press again to force terminate")
        signal.signal(signal.SIGINT, old_handler)
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    for i in range(start,start+num_updates):
        if stopflag[0]:
            break
        loss = model.update_fun(*get_batch(leadsheets))
        if i % 10 == 0:
            print("update {}, loss={}".format(i,loss))
        if i % 500 == 0 or (i % 100 == 0 and i < 1000):
            chords = get_chords(leadsheets)
            generated_out = model.generate_fun(chords)
            for samplenum, (out, chords) in enumerate(zip((generated_out != 0).astype(np.int8).tolist(), (chords != 0).astype(np.int8).tolist())):
                melody = relative_data.output_form_to_melody(out)
                leadsheet.write_leadsheet(chords, melody, 'output/sample{}_{}.ls'.format(i, samplenum))
            pickle.dump(model.learned_config,open('output/params{}.p'.format(i), 'wb'))
    if not stopflag[0]:
        signal.signal(signal.SIGINT, old_handler)