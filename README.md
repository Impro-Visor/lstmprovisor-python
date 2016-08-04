# Using LSTMprovisor

This project allows you to train LSTM-based neural network models of jazz music in the .ls format, as well as convert the models into connectome files for use in Impro-Visor.

## Dependencies and Setup

In order to run this project, you will need to install [Python 3.5][] (or later). You will also need the libraries `theano`, `theano-lstm`, `sexpdata`, and `matplotlib`, which you can install with

```
pip3 install theano theano-lstm sexpdata matplotlib
```

[Python 3.5]: https://www.python.org/downloads/

Note that Theano depends on SciPy. If you do not already have SciPy, `pip3` should install SciPy automatically when you install Theano, but if that fails, you can download SciPy from [their website][scipy]. Alternately, you can install a Python 3.5 distribution that already has SciPy installed, such as [Anaconda][].
 
[scipy]: http://scipy.org/install.html
[Anaconda]: https://www.continuum.io/downloads

Before using the scripts, you will also need to make a file called `.theanorc` in your home directory, with the following contents:

```
[global]
floatX=float32

[mode]=FAST_RUN
```

For additional Theano configuration options, including instructions on how to use the GPU, see the [theano config documentation][configdoc].

[configdoc]: http://deeplearning.net/software/theano/library/config.html

## Scripts

Python scripts are provided to accomplish certain tasks. Each script can be invoked using the `python3` executable, for example

```
python3 SCRIPTNAME.py [arguments]
```

- [main.py](instructions/main.md): The main entry point for the project. Trains different types of model, as well as generating samples from them.
- [param_cvt.py](instructions/param_cvt.md): Converts trained connectomes from pickle format (.p) to Impro-Visor connectome format (.ctome)
- [plot_internal_state.py](instructions/plot_internal_state.md): Plots the internal state of a network, produced by main.py in generation mode.
- [plot_data.py](instructions/plot_data.md): Plots a .csv file as a graph, allowing you to visualize the training loss of a network
- [lscat.py](instructions/lscat.md): Concatenates leadsheets together for easier viewing.
- [lssplit.py](instructions/lssplit.md): Splits leadsheets into multiple pieces.
- [generate_trade_helper.py](instructions/generate_trade_helper.md): Interleaves generated output with the original input in a single leadsheet, for use in autoencoder models.

Detailed instruction pages for each script are available in the instructions subdirectory, and each script will display a help message if the script is given the `-h` argument.

## Examples

Some general examples follow. See detailed instruction pages for more in-depth examples.

Train a product-of-experts generative model on a directory of leadsheet fileswith path `datasets/my_dataset`, automatically resuming training if previously interrupted. By default, each leadsheet file will be split into 4-bar chunks starting at each bar.

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset --resume-auto poex
```

Generate some leadsheets using the trained product-of-experts model, sampling from the dataset:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset/generated --resume 0 output_my_dataset/final_params.p --generate poex
```

Visualize the internal state of the network for the first generated leadsheet:

```
$ python3 plot_internal_state.py output_my_dataset/generated 0
```

Plot the training progress of the network:

```
$ python3 plot_data.py output_my_dataset/data.csv
```

Convert the trained network into a connectome file:

```
$ python3 param_cvt.py --keys param_keys/poex_keys.txt output_my_dataset/final_params.p
```

Train a compressing autoencoder on the same dataset, using fixed features and product-of-experts for compatibility with Impro-Visor:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset_compae --resume-auto compae poex queueless_std --feature_period 24 --add_loss
```

Run the autoencoder on some leadsheets from the dataset, and then combine the input and output into a trading summary leadsheet:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset_compae/generated --resume 0 output_my_dataset_compae/final_params.p --generate compae poex queueless_std --feature_period 24 --add_loss
$ python3 generate_trade_helper.py output_my_dataset_compae/generated
```

Convert the trained autoencoder into a connectome file:

```
$ python3 param_cvt.py --keys param_keys/ae_poex_keys.txt output_my_dataset_compae/final_params.p
```
