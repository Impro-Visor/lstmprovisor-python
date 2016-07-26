#plot_data.py: Plot a csv data file

```
usage: plot_data.py [-h] fn

Plot a .csv file

positional arguments:
  fn          File to plot

optional arguments:
  -h, --help  show this help message and exit
```

This script is designed to plot the `data.csv` file produced by `main.py` during training. You can pass this script the path to the `data.csv` file to visualize it:

```
$ python3 plot_data.py output_my_dataset/data.csv
```
