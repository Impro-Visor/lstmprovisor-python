#generate_trade_helper.py: Generate a trading summary leadsheet

```
usage: generate_trade_helper.py [-h] filedir

Helper to concatenate trades into single leadsheet

positional arguments:
  filedir     Directory to process

optional arguments:
  -h, --help  show this help message and exit
```

To use this script, pass the path to a generation output directory produced by `main.py`. This script will read the generated files and produce a new file `generated_trades.py` which alternates between the source pieces and the generated output produced by the network.