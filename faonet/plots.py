import matplotlib.pyplot as plt
import networkx as nx

def plot_trade_scatter(df, x_col='Reporter Country Code (M49)', y_col='Partner Country Code (M49)', 
                       value_col='Value', step=10, cmap='viridis', alpha=0.8, figsize=(8, 6)):
    """
    Plot a scatter plot of trade interactions between reporter and partner countries.

    Args:
        df (pd.DataFrame): DataFrame containing trade data.
        x_col (str): Column name for x-axis (reporter country codes).
        y_col (str): Column name for y-axis (partner country codes).
        value_col (str): Column name used for coloring the points.
        step (int): Step for showing ticks (default is every 10th).
        cmap (str): Matplotlib colormap.
        alpha (float): Transparency level for points.
        figsize (tuple): Size of the plot.

    Returns:
        matplotlib.axes.Axes: The plot axes object.
    """
    ax = df.plot(kind='scatter', x=x_col, y=y_col, s=32, c=value_col, 
                 cmap=cmap, alpha=alpha, figsize=figsize)

    # Define ticks
    x_ticks = df[x_col].unique()
    y_ticks = df[y_col].unique()
    ax.set_xticks(x_ticks[::step])
    ax.set_yticks(y_ticks[::step])

    # Style
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

    return ax


def plot_bipartite_network(B, group0_nodes, title=None, figsize=(12, 8), node_size=700, font_size=10):
    """
    Plot a bipartite network using NetworkX with edge weights as colors.

    Args:
        B (networkx.Graph): Bipartite graph with 'weight' attribute on edges.
        group0_nodes (list or set): Nodes from group 0 (e.g., exporters) used to compute layout.
        title (str): Optional title for the plot.
        figsize (tuple): Figure size in inches.
        node_size (int): Size of the nodes.
        font_size (int): Font size for node labels.

    Returns:
        matplotlib.axes.Axes: The plot axes object.
    """
    # Layout
    pos = nx.bipartite_layout(B, group0_nodes)

    # Extract weights
    edges = B.edges(data=True)
    weights = [d['weight'] for (_, _, d) in edges]
    max_weight = max(weights) if weights else 1  # avoid division by zero

    # Create figure
    plt.figure(figsize=figsize)
    nx.draw(
        B, pos, with_labels=True, node_size=node_size, font_size=font_size,
        edge_color=weights,
        width=[w / max_weight * 5 for w in weights],
        edge_cmap=plt.cm.Blues
    )

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()

    return plt.gca()
