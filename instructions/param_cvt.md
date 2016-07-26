#param_cvt.py: Convert a python parameters file into a connectome file

```
usage: param_cvt.py [-h] [--keys KEYS] [--output OUTPUT]
                    [--precision PRECISION] [--raw]
                    file

Convert a python parameters file into an Impro-Visor connectome file

positional arguments:
  file                  File to process

optional arguments:
  -h, --help            show this help message and exit
  --keys KEYS           File to load parameter names from
  --output OUTPUT       Base name of the output files
  --precision PRECISION
                        Decimal points of precision to use (default 18)
  --raw                 Create individual csv files instead of a connectome
                        file
```

In python, trained parameters are saved as pickle files containing a list of the model parameter matrices. To convert this to a format that Impro-Visor can read, we encode each model parameter matrix as a .csv file, and name it according to a key file, which describes the order that the parameters appear in the list. These .csv files are zipped together and given the extension .ctome, which can be loaded by Impro-Visor.

## Examples

Convert a product-of-experts network into a connectome file:

```
$ python3 param_cvt.py --keys param_keys/poex_keys.txt output_poex/final_params.p
```

Convert a compressing autoencoder network into a connectome file:

```
$ python3 param_cvt.py --keys param_keys/ae_poex_keys.txt output_compae/final_params.p
```
