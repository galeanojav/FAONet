# FAONet

FAONet is a Python package to analyze bipartite trade networks using data from FAOSTAT.  
It supports modular loading, filtering, network creation, and metrics computation.

FAONet is a project that leverages data from FAOSTAT to build and analyze bipartite networks in the context of international agricultural trade. The main goal is to model relationships between countries and products (or other relevant entities) as a complex network and explore its structural properties.

**Features:**
- Processing of FAOSTAT datasets
- Construction of bipartite networks (e.g., country-product, exporter-importer)
- Analysis of structural properties (degree, strength, clustering, communities)
- Interactive visualization and summary statistics

**Applications:**
- Agricultural economics
- Resilience analysis of food supply networks
- International trade structure exploration

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


