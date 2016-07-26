#lssplit.py: Split leadsheets

```
usage: lssplit.py [-h] [--output OUTPUT] file split

Split a leadsheet file.

positional arguments:
  file             File to process
  split            Bars to split at

optional arguments:
  -h, --help       show this help message and exit
  --output OUTPUT  Base name of the output files
```

To use this script, pass a single leadsheet file, and a number of bars. This script will chop up the input into chunks of length `split`.

For instance, to split up `large.ls` into chunks of 4 bars, use
```
$ python3 lssplit.py large.ls 4 --output smaller
```

which will produce files `smaller_0.ls`, `smaller_1.ls`, etc.