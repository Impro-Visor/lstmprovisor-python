# main.py: Train or generate from a neural network model

This script allows you to actually interact with a neural network model. You can run the script with

```
$ python3 main.py [general arguments] MODELTYPE [model-specific arguments]
```

## General Arguments
```
  -h, --help            show this help message and exit
  --dataset DATASET [DATASET ...]
                        Path(s) to dataset folder (with .ls files). If
                        multiple are passed, samples randomly from each
                        (default: ['dataset'])
  --validation VALIDATION
                        Path to validation dataset folder (with .ls files)
                        (default: None)
  --outputdir OUTPUTDIR
                        Path to output folder (default: output)
  --check_nan           Check for nans during execution (default: False)
  --batch_size BATCH_SIZE
                        Size of batch (default: 10)
  --iterations ITERATIONS
                        How many iterations to train (default: 50000)
  --segment_len SEGMENT_LEN
                        Length of segment to train on (default: 4bar)
  --segment_step SEGMENT_STEP
                        Period at which segments may begin (default: 1bar)
  --resume TIMESTEP PARAMFILE
                        Where to restore from: timestep, and file to load
                        (default: None)
  --resume_auto         Automatically restore from a previous run using output
                        directory (default: False)
  --generate            Don't train, just generate. Should be used with
                        restore. (default: False)
  --generate_over SOURCE DIV_WIDTH
                        Don't train, just generate, and generate over SOURCE
                        chord changes divided into chunks of length DIV_WIDTH
                        (or one contiguous chunk if DIV_WIDTH is 'full'). Can
                        use 'bar' as a unit. Should be used with restore.
                        (default: None)
```

## Model Types


###`simple`: A simple model
This is a simple class of model, not available in Impro-Visor.

`simple`-specific arguments:
```
usage: main.py simple [-h] [--per_note] {abs,cot,rel}

positional arguments:
  {abs,cot,rel}  Type of encoding to use

optional arguments:
  -h, --help     show this help message and exit
  --per_note     Enable note memory cells
```
The three types of encoding in this mode are

- `abs`: An absolute encoding, where each distinct pitch has a distinct bit in the note representation
- `cot`: The circles-of-thirds representation, where pitches are determined by which circles of major and minor thirds that pitch is in. This encoding was originally described by Judy A. Franklin in [Recurrent Neural Networks and Pitch Representations for Music Tasks ](http://cs.smith.edu/~jfrankli/papers/FLAIRS04FranklinJ.pdf).
- `rel`: An interval-relative encoding, where each note is encoded as the size and direction of interval between this note and the previous one.

###`poex`: A product-of-experts generative model.
This is the type of generative model available in Impro-Visor.

`poex`-specific arguments:
```
  -h, --help            show this help message and exit
  --per_note  Enable note memory cells
  --layer_size LAYER_SIZE
                        Layer size of the LSTMs. Only works without note
                        memory cells
  --num_layers NUM_LAYERS
                        Number of LSTM layers. Only works without note memory
                        cells
```
`--per_note` enables memory cells which are fixed to particular notes and do not shift with the rest of the network. Although these were investigated as being potentially useful, they did not give a significant advantage and were not implemented in Impro-Visor for simplicity. If you wish to train a model for Impro-Visor, do not use the `--per_note` flag.

###`compae`: A compressing autoencoder model
This is the type of trading model available in Impro-Visor.

`compae`-specific arguments:
```
usage: main.py [general arguments] compae ENCODING MANAGER [compae optional arguments]

positional arguments:
  {abs,cot,rel,poex}    Type of encoding to use
  {std,var,sample_var,queueless_var,queueless_std,nearness_std}
                        Type of queue manager to use

optional arguments:
  -h, --help            show this help message and exit
  --per_note            Enable note memory cells
  --hide_output         Hide previous outputs from the decoder
  --sparsity_loss_scale SPARSITY_LOSS_SCALE
                        How much to scale the sparsity loss by
  --variational_loss_scale VARIATIONAL_LOSS_SCALE
                        How much to scale the variational loss by
  --feature_size FEATURE_SIZE
                        Size of feature vectors
  --feature_period FEATURE_PERIOD
                        If in queueless mode, period of features in timesteps
  --add_pre_noise [ADD_PRE_NOISE]
                        Add Gaussian noise to the feature values before
                        applying the activation function
  --add_post_noise [ADD_POST_NOISE]
                        Add Gaussian noise to the feature values after
                        applying the activation function
  --train_decoder_only  Only modify the decoder parameters
  --layer_size LAYER_SIZE
                        Layer size of the LSTMs. Only works without note
                        memory cells
  --num_layers NUM_LAYERS
                        Number of LSTM layers. Only works without note memory
                        cells
  --priority_loss [LOSS_MODE_PRIORITY]
                        Use priority loss scaling mode (with the specified
                        curviness)
  --add_loss            Use adding loss scaling mode
  --cutoff_loss CUTOFF  Use cutoff loss scaling mode with the specified per-
                        batch cutoff
  --trigger_loss TRIGGER RAMP_TIME
                        Use trigger loss scaling mode with the specified per-
                        batch trigger value and desired ramp-up time
```

In order to be compatible with Impro-Visor, you must use the encoding `poex` and the queue manger `queueless_std` or `queueless_var`, and you must not use the flags `--per_note` or `--hide_output`. Other modes are available for training and generation within python, but were not implemented in Impro-Visor. Additionally, `--feature_period` should be 24, corresponding to the duration of a half note.

Other than `poex`, which matches the `poex` generative model, the other encoding modes correspond to the encodings for the `simple` model.

There are a variety of queue managers available:

- `std`: The standard, variable-sized feature model, where the network decides where to output features, and is encouraged to output few features.
- `var`: A variational version of the variable-sized feature model, where the features are latent variables that are sampled from a normal distribution, and that distribution is regularized to be similar to a unit normal distribution.
- `sample_var`: Like `var`, but instead of allowing a feature to be output with fractional strength, the network samples from the feature strength to decide where the features are, and then is trained using a variant of reinforcement learning.
- `queueless_std`: A fixed-size feature model, with features repeating at a fixed interval.
- `queueless_var`: A variational version of the fixed-size feature model.
- `nearness_std`: A version of the variable-sized feature model where features that are close together are penalized more than features that are far away.

For some queue mangers, there are multiple loss values: the reconstruction loss, as well as some extra loss. The `--sparsity_loss_scale` and `--variational_loss_scale` allow you to scale the importance of these losses, and the loss modes `--add_loss`, `--priority_loss`, `--cutoff_loss`, and `--trigger_loss` determine how the losses are balanced:

- `--add_loss` simply adds the losses
- `--priority_loss` attempts to scale the losses so that the largest loss is most important
- `--cutoff_loss` ignores the extra loss unless the reconstruction loss is small enough
- `--trigger_loss` waits until the reconstruction loss becomes small enough, and then interpolates the extra loss from having no influence to having full influence (as in `--add_loss` mode)

## Examples

Train a product-of-experts generative model on a directory of leadsheet fileswith path `datasets/my_dataset`, automatically resuming training if previously interrupted. Each leadsheet file will be split into 4-bar chunks starting at each bar.

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset --resume_auto poex
```

Train a product-of-experts generative model as before, but split each leadsheet into 8-bar chunks, starting at each multiple of 4 bars:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset  --resume_auto --segment_len 8bar --segment_step 4bar poex
```

Generate some leadsheets using a trained product-of-experts model, sampling from the dataset:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset/generated --resume 0 output_my_dataset/final_params.p --generate poex
```

As above, but generate over a particular piece `my_generate_target.ls` in 4 bar chunks:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset/generated --resume 0 output_my_dataset/final_params.p --generate_over my_generate_target.ls 4bar poex
```

As above, but generate over the whole piece in a single run of the network (without breaking into chunks)

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset/generated --resume 0 output_my_dataset/final_params.p --generate_over my_generate_target.ls full poex
```

Train a compressing autoencoder on the same dataset, using fixed features and product-of-experts for compatibility with Impro-Visor:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset_compae --resume_auto compae poex queueless_std --feature_period 24 --add_loss
```

As above, but train a variational autoencoder instead of a standard one, scaling the variational loss by 0.01 and only enforcing the variational loss after the reconstruction loss drops below 4 per sample:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset_compae --resume_auto compae poex queueless_var --feature_period 24 --trigger_loss 4 2000
```

As above, but with a standard autoencoder and variable-size features:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset_compae --resume_auto compae poex std --trigger_loss 4 2000
```

Run the autoencoder and generate some output:

```
$ python3 main.py --dataset datasets/my_dataset --outputdir output_my_dataset_compae/generated --resume 0 output_my_dataset_compae/final_params.p --generate compae poex queueless_std --feature_period 24 --add_loss
```

