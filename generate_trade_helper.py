import sys
import os
import leadsheet
import argparse
import pickle
import numpy as np
import lscat

def main(filedir):
    files = []
    with open(os.path.join(filedir,'generated_sources.txt'),'r') as f:
        for i,line in enumerate(f):
            files.append(line.split(":")[0])
            files.append(os.path.join(filedir,'generated_{}.ls'.format(i)))
    lscat.main(files, output=os.path.join(filedir,'generated_trades.ls'), verbose=False)

parser = argparse.ArgumentParser(description='Helper to concatenate trades into single leadsheet')
parser.add_argument('filedir', help='Directory to process')

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
