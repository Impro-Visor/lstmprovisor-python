import sys
import leadsheet
import argparse
import constants
import os

def main(file, split, output=None):
    c,m = leadsheet.parse_leadsheet(file)
    lslen = leadsheet.get_leadsheet_length(c,m)
    divwidth = int(split) * (constants.WHOLE//constants.RESOLUTION_SCALAR)
    slices = [leadsheet.slice_leadsheet(c,m,s,s+divwidth) for s in range(0,lslen,divwidth)]
    if output is None:
        output = file + "-split"
    for i, (chords, melody) in enumerate(slices):
        leadsheet.write_leadsheet(chords, melody, '{}_{}.ls'.format(output,i))

parser = argparse.ArgumentParser(description='Split a leadsheet file.')
parser.add_argument('file',help='File to process')
parser.add_argument('split',help='Bars to split at')
parser.add_argument('--output', help='Base name of the output files')

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))