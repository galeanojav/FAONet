
import pytest
import pandas as pd
import networkx as nx
from faonet.metrics import compute_degree, compute_strength
from faonet.graphs import build_bipartite_graph
from faonet.clustering import compute_bipartite_clustering

def test_build_bipartite_graph_and_degree():
    data = {
        'Reporter Countries': ['A', 'A', 'B', 'C'],
        'Partner Countries': ['X', 'Y', 'Y', 'Z'],
        'Value': [10, 20, 30, 40]
    }
    df = pd.DataFrame(data)
    B, reporters, partners = build_bipartite_graph(df)
    df_degrees = compute_degree(B, reporters, partners)
    assert "Degree" in df_degrees.columns
    assert len(df_degrees) == len(B.nodes)

def test_compute_strength():
    B = nx.Graph()
    B.add_edge('A', 'X', weight=10)
    B.add_edge('A', 'Y', weight=5)
    B.add_edge('B', 'Y', weight=7)
    reporters = {'A', 'B'}
    partners = {'X', 'Y'}
    df_strength = compute_strength(B, reporters, partners)
    assert "Strength" in df_strength.columns
    assert df_strength.loc['A', 'Strength'] == 15
    assert df_strength.loc['B', 'Strength'] == 7

def test_clustering_does_not_fail():
    B = nx.Graph()
    B.add_edge('A', 'X', weight=1)
    B.add_edge('A', 'Y', weight=1)
    B.add_edge('B', 'X', weight=1)
    B.add_edge('B', 'Y', weight=1)
    df_clust = compute_bipartite_clustering(B)
    assert "C4b" in df_clust.columns
