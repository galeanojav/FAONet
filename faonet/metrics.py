import pandas as pd

def degree_by_group(G, group_nodes):
    """Compute degree for a group of nodes.

    Args:
        G (networkx.Graph): Network.
        group_nodes (iterable): Set of nodes.

    Returns:
        pd.DataFrame: Degree per node.
    """
    degrees = {n: G.degree(n) for n in group_nodes}
    return pd.DataFrame(degrees.items(), columns=["Node", "Degree"])
