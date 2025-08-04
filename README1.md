# FAONet

FAONet is a Python package to analyze bipartite trade networks using data from FAOSTAT.  
It supports modular loading, filtering, network creation, and metrics computation.

## Installation

```bash
pip install .
```

## Usage

```python
from faonet.io import load_and_merge_csv
from faonet.filtering import filter_top_percentile
from faonet.network import build_bipartite_network
```
