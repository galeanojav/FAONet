import pandas as pd
import networkx as nx

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



def compute_degree_and_strength(B, reporters, partners):
    """
    Compute degree and strength (sum of weights) for nodes in a bipartite network.

    Args:
        B (networkx.Graph): Bipartite network with 'weight' attribute on edges.
        reporters (set): Nodes from group 0 (e.g., exporters).
        partners (set): Nodes from group 1 (e.g., importers).

    Returns:
        tuple: (df_exporters, df_importers)
            - df_exporters (pd.DataFrame): Degree and strength for reporter nodes.
            - df_importers (pd.DataFrame): Degree and strength for partner nodes.
    """
    # Compute strength: sum of edge weights per node
    strength = {
        node: sum(data['weight'] for _, _, data in B.edges(node, data=True))
        for node in B.nodes()
    }

    # Compute degree using built-in function
    degree = dict(B.degree())

    # Separate by node group
    exporters_strength = {node: strength[node] for node in reporters}
    importers_strength = {node: strength[node] for node in partners}
    exporters_degree = {node: degree[node] for node in reporters}
    importers_degree = {node: degree[node] for node in partners}

    # Create dataframes
    df_exporters = pd.DataFrame({
        "Degree": pd.Series(exporters_degree),
        "Strength": pd.Series(exporters_strength)
    }).dropna()

    df_importers = pd.DataFrame({
        "Degree": pd.Series(importers_degree),
        "Strength": pd.Series(importers_strength)
    }).dropna()

    return df_exporters, df_importers