import networkx as nx

def build_bipartite_network(df, reporter_col, partner_col, weight_col):
    """Build a bipartite network from trade data.

    Args:
        df (pd.DataFrame): Trade DataFrame.
        reporter_col (str): Exporting countries column.
        partner_col (str): Importing countries column.
        weight_col (str): Trade value column.

    Returns:
        networkx.Graph: Bipartite network.
        set: Exporter nodes.
        set: Importer nodes.
    """
    B = nx.Graph()
    reporters = set(df[reporter_col])
    partners = set(df[partner_col])

    B.add_nodes_from(reporters, bipartite=0)
    B.add_nodes_from(partners, bipartite=1)

    for _, row in df.iterrows():
        B.add_edge(row[reporter_col], row[partner_col], weight=row[weight_col])

    return B, reporters, partners

def remove_zero_weight_edges(G):
    """Remove edges with zero weight from the graph.

    Args:
        G (networkx.Graph): Input graph.

    Returns:
        networkx.Graph: Cleaned graph.
    """
    zero_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("weight", 1) == 0]
    G.remove_edges_from(zero_edges)
    return G
