#lscat.py: Concatenate leadsheets

```
usage: lscat.py [-h] [--output OUTPUT] [--verbose] files [files ...]

Concatenate leadsheet files.

positional arguments:
  files            Files to process

optional arguments:
  -h, --help       show this help message and exit
  --output OUTPUT  Name of the output file
  --verbose        Be verbose about processing
```

To use this script, pass a list of leadsheet files. These files will then be concatenated together into a new leadsheet file. It is recommended that you also use the `--output` argument to specify an output filename, but otherwise a filename is generated based on the first concatenated file.

For example, to concatenate `a.ls`, `b.ls`, and `c.ls`, you can run

```
$ python3 lscat.py a.ls b.ls c.ls --output combined.ls
```
