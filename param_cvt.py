import sys
import os
import leadsheet
import argparse
import pickle
import numpy as np
import zipfile
import io

def main(file, precision, keys=None, output=None, make_zip=False):
    params = pickle.load(open(file, 'rb'))
    param_vals = [x if isinstance(x,np.ndarray) else x.get_value() for x in params]
    if output is None:
        output = os.path.splitext(file)[0] + (".ctome" if make_zip else "-raw")
    if keys is None:
        key_names = [str(x) for x in range(len(params))]
    else:
        with open(keys,'r') as f:
            key_names = f.readlines()
        assert len(key_names) == len(params), "Wrong number of keys for params! {} keys, {} params".format(len(key_names), len(params))

    fmt = '%.{}e'.format(precision)
    if make_zip:
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zfile:
            for name,val in zip(key_names, param_vals):
                with io.BytesIO() as str_capture:
                    np.savetxt(str_capture, val, fmt=fmt, delimiter=",")
                    zfile.writestr("param_{}.csv".format(name.strip()), str_capture.getvalue())
    else:
        for name,val in zip(key_names, param_vals):
            np.savetxt("{}_{}.csv".format(output,name.strip()), val, fmt=fmt, delimiter=",")

parser = argparse.ArgumentParser(description='Convert a python parameters file into an Impro-Visor connectome file')
parser.add_argument('file', help='File to process')
parser.add_argument('--keys', help='File to load parameter names from')
parser.add_argument('--output', help='Base name of the output files')
parser.add_argument('--precision', default=18, type=int, help='Decimal points of precision to use (default 18)')
parser.add_argument('--raw', dest='make_zip', action='store_false', help='Create individual csv files instead of a connectome file')

if __name__ == '__main__':
    args = parser.parse_args()
    main(**vars(args))
