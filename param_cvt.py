import sys
import leadsheet
import argparse
import pickle
import numpy as np

def main(file, keys=None, output=None):
    params = pickle.load(open(file, 'rb'))
    param_vals = [x if isinstance(x,np.ndarray) else x.get_value() for x in params]
    if output is None:
        output = file + "-raw"
    if keys is None:
        key_names = [str(x) for x in range(len(params))]
    else:
        with open(keys,'r') as f:
            key_names = f.readlines()
        assert len(key_names) == len(params), "Wrong number of keys for params! {} keys, {} params".format(len(key_names), len(params))
    for name,val in zip(key_names, param_vals):
        np.savetxt("{}_{}.csv".format(output,name.strip()), val, delimiter=",")

parser = argparse.ArgumentParser(description='Convert a parameters file into CSV files')
parser.add_argument('file', help='File to process')
parser.add_argument('--keys', help='File to load parameter names from')
parser.add_argument('--output', help='Base name of the output files')

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))