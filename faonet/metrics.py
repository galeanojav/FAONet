import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite

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



def compute_betweenness_all(G):
    """
    Compute betweenness centrality for bipartite network and its projections,
    using both real weights and inverted weights (for shortest path interpretation).

    Args:
        G (networkx.Graph): Bipartite graph with edge attribute 'weight'.

    Returns:
        pd.DataFrame: DataFrame with betweenness centralities per node.
    """
    # Identify bipartite sets
    exportadores = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
    importadores = set(G) - exportadores

    # Invert weights for shortest-path based betweenness
    G_inv = G.copy()
    for u, v, d in G_inv.edges(data=True):
        peso = d.get("weight", 1)
        d["inv_weight"] = 1 / peso if peso > 0 else 0

    # Betweenness in original bipartite network
    bet_bip = nx.betweenness_centrality(G, weight="weight")
    bet_bip_inv = nx.betweenness_centrality(G_inv, weight="inv_weight")

    # Projected graphs
    proy_exp = bipartite.weighted_projected_graph(G, exportadores)
    proy_imp = bipartite.weighted_projected_graph(G, importadores)

    # Betweenness in projections (real weights)
    bet_proy_exp = nx.betweenness_centrality(proy_exp, weight="weight")
    bet_proy_imp = nx.betweenness_centrality(proy_imp, weight="weight")

    # Invert weights in projections
    for _, _, d in proy_exp.edges(data=True):
        d["inv_weight"] = 1 / d["weight"] if d["weight"] > 0 else 0
    for _, _, d in proy_imp.edges(data=True):
        d["inv_weight"] = 1 / d["weight"] if d["weight"] > 0 else 0

    bet_proy_exp_inv = nx.betweenness_centrality(proy_exp, weight="inv_weight")
    bet_proy_imp_inv = nx.betweenness_centrality(proy_imp, weight="inv_weight")

    # Build results
    nodos = list(G.nodes())
    df_bet = pd.DataFrame({
        "node": nodos,
        "bipartite_set": [G.nodes[n].get("bipartite") for n in nodos],
        "betweenness_bipartite": [bet_bip.get(n, 0) for n in nodos],
        "betweenness_bipartite_inv": [bet_bip_inv.get(n, 0) for n in nodos],
        "betweenness_proj_exporters": [bet_proy_exp.get(n, None) for n in nodos],
        "betweenness_proj_exporters_inv": [bet_proy_exp_inv.get(n, None) for n in nodos],
        "betweenness_proj_importers": [bet_proy_imp.get(n, None) for n in nodos],
        "betweenness_proj_importers_inv": [bet_proy_imp_inv.get(n, None) for n in nodos],
    })

    return df_bet