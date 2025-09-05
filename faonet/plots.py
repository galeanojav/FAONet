import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import seaborn as sns





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



def plot_bipartite_network2(B, group0_nodes, title=None, figsize=(12, 8), node_size=700, font_size=10):
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
    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.bipartite_layout(B, group0_nodes)

    # Extract weights
    edges = B.edges(data=True)
    weights = [d['weight'] for (_, _, d) in edges]
    max_weight = max(weights) if weights else 1  # avoid division by zero

    # Draw network
    nx.draw(
        B, pos, ax=ax, with_labels=True, node_size=node_size, font_size=font_size,
        edge_color=weights,
        width=[w / max_weight * 5 for w in weights],
        edge_cmap=plt.cm.Blues
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax



def plot_degree_bar(df, country_col="Reporter Country", degree_col="Degree", 
                    title="Node Degree", xlabel="Country", ylabel="Degree", 
                    color="blue", alpha=0.7, figsize=(12, 6), rotation=90):
    """
    Plot a bar chart of node degrees (e.g., exporters or importers) in the bipartite network.

    Args:
        df (pd.DataFrame): DataFrame with at least two columns: one for countries, one for degree values.
        country_col (str): Name of the column with country/node names.
        degree_col (str): Name of the column with degree values.
        title (str): Plot title.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        color (str): Color of the bars.
        alpha (float): Transparency of the bars.
        figsize (tuple): Figure size.
        rotation (int): Rotation angle for x-tick labels.

    Returns:
        matplotlib.axes.Axes: The plot axes object.
    """
    df_sorted = df.sort_values(by=degree_col, ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df_sorted[country_col], df_sorted[degree_col], color=color, alpha=alpha)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=rotation)
    plt.tight_layout()
    plt.show()

    return ax


def plot_degree_comparison(df_reporters, df_partners,
                           reporter_country_col="Reporter Country",
                           partner_country_col="Partner Country",
                           degree_col="Degree",
                           figsize=(12, 10),
                           reporter_color="blue",
                           partner_color="orange",
                           alpha=0.7,
                           rotation=90,
                           use_log_scale=False):
    """
    Plot side-by-side scatter plots comparing degree of reporter and partner countries.

    Args:
        df_reporters (pd.DataFrame): DataFrame of reporter countries with degree values.
        df_partners (pd.DataFrame): DataFrame of partner countries with degree values.
        reporter_country_col (str): Column name for reporter country names.
        partner_country_col (str): Column name for partner country names.
        degree_col (str): Column name with degree values.
        figsize (tuple): Size of the figure.
        reporter_color (str): Color for reporter scatter plot.
        partner_color (str): Color for partner scatter plot.
        alpha (float): Transparency of scatter points.
        rotation (int): Rotation angle for x-axis labels.
        use_log_scale (bool): Whether to use log scale on the y-axis.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    df_reporters_sorted = df_reporters.sort_values(by=degree_col, ascending=False)
    df_partners_sorted = df_partners.sort_values(by=degree_col, ascending=False)

    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)

    axs[0].scatter(df_reporters_sorted[reporter_country_col],
                   df_reporters_sorted[degree_col],
                   color=reporter_color, alpha=alpha)
    axs[0].set_xlabel("Exporter Countries")
    axs[0].set_ylabel("Degree (Number of Connections)")
    axs[0].set_title("Degree - Reporter Countries")
    axs[0].tick_params(axis='x', rotation=rotation)
    if use_log_scale:
        axs[0].set_yscale('log')

    axs[1].scatter(df_partners_sorted[partner_country_col],
                   df_partners_sorted[degree_col],
                   color=partner_color, alpha=alpha)
    axs[1].set_xlabel("Importer Countries")
    axs[1].set_title("Degree - Partner Countries")
    axs[1].tick_params(axis='x', rotation=rotation)
    if use_log_scale:
        axs[1].set_yscale('log')

    plt.tight_layout()
    plt.show()
    return


def plot_degree_by_rank(df_reporters, df_partners,
                        reporter_label="Exporters",
                        partner_label="Importers",
                        degree_col="Degree",
                        figsize=(10, 5),
                        reporter_color="blue",
                        partner_color="orange",
                        alpha=0.7,
                        use_log_y=True,
                        use_log_x=False,
                        title="Node Degree by Rank",
                        xlabel="Rank",
                        ylabel="Degree (Number of Connections)"):
    """
    Plot degrees of reporter and partner countries by descending rank.

    Args:
        df_reporters (pd.DataFrame): DataFrame of reporter nodes with degree values.
        df_partners (pd.DataFrame): DataFrame of partner nodes with degree values.
        reporter_label (str): Label for reporter nodes in legend.
        partner_label (str): Label for partner nodes in legend.
        degree_col (str): Column containing degree values.
        figsize (tuple): Size of the figure.
        reporter_color (str): Color for exporter plot.
        partner_color (str): Color for importer plot.
        alpha (float): Transparency of points.
        use_log_y (bool): Whether to use log scale for y-axis.
        use_log_x (bool): Whether to use log scale for x-axis.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.

    Returns:
        matplotlib.axes.Axes: The plot axes object.
    """
    df_reporters_sorted = df_reporters.sort_values(by=degree_col, ascending=False)
    df_partners_sorted = df_partners.sort_values(by=degree_col, ascending=False)

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(range(1, len(df_reporters_sorted) + 1),
               df_reporters_sorted[degree_col],
               color=reporter_color,
               label=reporter_label,
               alpha=alpha)

    ax.scatter(range(1, len(df_partners_sorted) + 1),
               df_partners_sorted[degree_col],
               color=partner_color,
               label=partner_label,
               alpha=alpha)

    if use_log_y:
        ax.set_yscale("log")
    if use_log_x:
        ax.set_xscale("log")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    return ax


def plot_weight_matrix(df, row="Partner Countries", col="Reporter Countries", 
                       value="Value", cmap="coolwarm", figsize=(20, 15), 
                       title="Weighted Adjacency Matrix (Trade Volume)"):
    """
    Plot a heatmap of the weighted bipartite adjacency matrix.

    Args:
        df (pd.DataFrame): Filtered trade DataFrame.
        row (str): Column name for matrix rows (e.g., importers).
        col (str): Column name for matrix columns (e.g., exporters).
        value (str): Column with weights (e.g., trade volume).
        cmap (str): Seaborn colormap.
        figsize (tuple): Size of the figure.
        title (str): Plot title.

    Returns:
        matplotlib.axes.Axes: The heatmap Axes object.
    """
    # Build matrix
    matrix = df.pivot(index=row, columns=col, values=value)

    # Sort rows/cols by total weights
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index,
                        matrix.sum(axis=0).sort_values(ascending=False).index]

    # Plot
    plt.figure(figsize=figsize)
    ax = sns.heatmap(matrix, cmap=cmap, annot=False, linewidths=0.5)

    plt.xlabel(col)
    plt.ylabel(row)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return ax


def plot_top_betweenness(df, col, title=None, color="steelblue", top_n=10, label_col="node", xlabel="Betweenness Centrality"):
    """
    Plot a horizontal bar chart of the top N nodes by betweenness centrality.

    Args:
        df (pd.DataFrame): DataFrame with betweenness values and node labels.
        col (str): Column name containing betweenness centrality values.
        title (str): Title of the plot.
        color (str): Color of the bars.
        top_n (int): Number of top nodes to show.
        label_col (str): Column with node names (default: 'node').
        xlabel (str): Label for x-axis.

    Returns:
        matplotlib.axes.Axes: The plot axes object.
    """
    top = df.sort_values(by=col, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top[label_col], top[col], color=color)
    ax.set_xlabel(xlabel)
    ax.set_title(title or f"Top {top_n} Nodes by {col}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()

    return ax