# Phenix project - Happy meal algorithm

The main goal of this repository is to create balanced meals from a list of products


## Requirements
The numpy and [openfoodfacts](https://github.com/openfoodfacts/openfoodfacts-python) packages are required to properly use the repo.
Tested on the following version:
```python
import sys
import numpy, openfoodfacts
print('Python %s' % '.'.join(map(str, sys.version_info[:3])))
print('Numpy %s, Openfoodfacts %s' % (numpy.__version__, openfoodfacts.__version__))
```
```console
Python 3.6.5
Numpy 1.15.0, Openfoodfacts 0.1.0
```

## How to use it
Depending on the feature you wish to test, you may need an Internet connection (product information)

### Product information

Pass a barcode (EAN) as argument of the demo script:
```bash
python product_info_demo.py 3392460480827
```

### Meal balancing

Run a simulation with the demo script similarly as below:
```bash
python tetris_demo.py 1000 --item_max_qty 100. --portion_size 500. --overflow_thresh 0.2 --underflow_thresh 0.1
```

which should return the result of the algorithm:
```bash
------------
RESULT
------------
40 batches for 1 persons (portion of 500.0): 502 items
1 batches for 3 persons (portion of 500.0): 27 items
3 batches for 2 persons (portion of 500.0): 65 items
Average batch loss: 41.91587070338677
Number of remaining items: 201 portioned, 4 unportioned
Number of large items: 201
Number of unindentified items: 0
```

Many parameters can be adjusted with the previous arguments.
To check the full list of arguments and their meaning, use the help of the parser:
```bash
python tetris_demo.py -h
```

## Useful links
[Git Cheat Sheet](https://data-for-good.slack.com/files/U284ZS6JW/FDSA71Q0H/git-cheatsheet.pdf) <br>
[Phenix Trello](https://trello.com/b/X9SX81OU/algo-matching-db-open-food-fact) <br>
[Data4Good Github](https://github.com/dataforgoodfr/) <br>


## TODO
- [x] Implement an EAN - product information function
- [x] Implement a naive meal balancing algorithm
- [ ] Explore brute force and smart optimisation techniques
