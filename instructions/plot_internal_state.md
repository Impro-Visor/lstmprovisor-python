#plot_internal_state.py: Plot the internal state of a network

```
usage: plot_internal_state.py [-h] folder idx

Plot the internal state of a network

positional arguments:
  folder      Directory with the generated files
  idx         Zero-based index of the output to visualize

optional arguments:
  -h, --help  show this help message and exit
```

Using matplotlib, this script plots the internal state of the network while generating a particular piece. To use the script, first run `main.py` with the `--generate` or `--generate_over` arguments. This will output a series of files to the desired output directory. You must then pass that directory to this utility, along with an index of the piece to view. For instance, if running main.py gives you a directory `generated_stuff`, with files

```
generated_stuff/generated_0.ls
generated_stuff/generated_1.ls
generated_stuff/generated_10.ls
generated_stuff/generated_11.ls
generated_stuff/generated_2.ls
generated_stuff/generated_3.ls
generated_stuff/generated_4.ls
generated_stuff/generated_5.ls
generated_stuff/generated_6.ls
generated_stuff/generated_7.ls
generated_stuff/generated_8.ls
generated_stuff/generated_9.ls
generated_stuff/generated_chosen.npy
generated_stuff/generated_info_0.npy
generated_stuff/generated_info_1.npy
generated_stuff/generated_probs.npy
generated_stuff/generated_sources.txt
```

and you want to view the state of the network while generating `generated_6.ls`, you can run

```
$ python3 plot_internal_state.py generated_stuff 6
```
