import networkx as nx

def export_gml(G, filepath):
    """Export the graph to a GML file.

    Args:
        G (networkx.Graph): NetworkX graph.
        filepath (str): Output path.
    """
    nx.write_gml(G, filepath)
