import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns

def plot_trade_scatter(df, x_col='Reporter Country Code (M49)', y_col='Partner Country Code (M49)', 
                       value_col='Value', step=10, cmap='viridis', alpha=0.8, figsize=(8, 6)):
    """
    Plot a scatter plot of trade interactions between reporter and partner countries.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trade data.
    x_col : str
        Column name for x-axis (e.g. reporter country codes).
    y_col : str
        Column name for y-axis (e.g. partner country codes).
    value_col : str
        Column name used for point color intensity (e.g. trade value).
    step : int
        Interval of tick marks on the axes (e.g. show every 10th value).
    cmap : str
        Colormap to use for the scatter points.
    alpha : float
        Transparency level for the points.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.axes.Axes
        The plot axes object.
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
    Plot a bipartite network using NetworkX with edge weights shown as color intensity.

    Parameters
    ----------
    B : networkx.Graph
        Bipartite graph with 'weight' attributes on edges.
    group0_nodes : list or set
        Nodes from one bipartite group (used for layout positioning).
    title : str, optional
        Title of the plot.
    figsize : tuple
        Figure size in inches.
    node_size : int
        Size of the nodes in the plot.
    font_size : int
        Font size for node labels.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object of the plot.
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



def plot_bipartite_network_enhanced(
    B,
    group0_nodes,
    title=None,
    figsize=(14, 9),
    exporter_color="#0072B2",
    importer_color="#D55E00",
    edge_cmap="Greys",
    edge_alpha=0.45,
    edge_width_scale=4.0,
    label_top_n=10,
    font_size=13,
    title_font_size=18,
    node_size_mode="strength",
    node_size_range=(250, 1400),
    default_node_size=700,
    partition_gap=1.2,
    label_offset=0.04,
    label_min_gap=0.055,
    draw_label_connectors=True,
    x_margin_left=0.18,
    x_margin_right=0.32,
    y_margin=0.06,
    show_axis=False,
    save_path=None,
    save_dpi=300,
    save_bbox_inches="tight",
):
    """
    Plot a cleaner bipartite network with partition-specific colors, selective labels,
    grayscale edges, and node sizes based on degree or strength.

    Parameters
    ----------
    B : networkx.Graph
        Bipartite graph with optional 'weight' attributes on edges.
    group0_nodes : list or set
        Nodes in the left partition (typically exporters).
    title : str, optional
        Title of the plot.
    figsize : tuple
        Figure size in inches.
    exporter_color : str
        Color for exporter nodes. Default uses Okabe-Ito blue.
    importer_color : str
        Color for importer nodes. Default uses Okabe-Ito vermillion.
    edge_cmap : str
        Matplotlib colormap for edges.
    edge_alpha : float
        Transparency of edges.
    edge_width_scale : float
        Maximum width multiplier for edges based on weights.
    label_top_n : int
        Number of top exporters and top importers to label.
    font_size : int
        Font size for labels.
    title_font_size : int
        Font size for the title.
    node_size_mode : {"strength", "degree", None}
        Metric used to scale node sizes. If None, all nodes use `default_node_size`.
    node_size_range : tuple
        Minimum and maximum node size when scaling is enabled.
    default_node_size : int
        Node size when `node_size_mode` is None.
    partition_gap : float
        Horizontal separation between the two node partitions.
    label_offset : float
        Horizontal offset used to place labels outside the nodes.
    label_min_gap : float
        Minimum vertical separation enforced between labels on the same side.
    draw_label_connectors : bool
        Whether to draw thin connector lines from shifted labels to their nodes.
    x_margin_left : float
        Extra horizontal margin on the left side of the plot.
    x_margin_right : float
        Extra horizontal margin on the right side of the plot.
    y_margin : float
        Extra vertical margin around the layout.
    show_axis : bool
        Whether to display axes.
    save_path : str or None
        If provided, save the figure to this path.
    save_dpi : int
        Resolution used when saving the figure.
    save_bbox_inches : str
        Bounding box option passed to `savefig`.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object of the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    group0_nodes = list(group0_nodes)
    group0_set = set(group0_nodes)
    group1_nodes = [node for node in B.nodes if node not in group0_set]

    pos = nx.bipartite_layout(B, group0_nodes)
    pos = {
        node: (
            -partition_gap / 2 if node in group0_set else partition_gap / 2,
            coords[1],
        )
        for node, coords in pos.items()
    }

    edge_data = list(B.edges(data=True))
    weights = [data.get("weight", 1.0) for _, _, data in edge_data]
    max_weight = max(weights) if weights else 1.0
    edge_widths = [0.5 + (weight / max_weight) * edge_width_scale for weight in weights]

    degree_dict = dict(B.degree())
    strength_dict = dict(B.degree(weight="weight"))

    if node_size_mode == "degree":
        size_metric = degree_dict
    elif node_size_mode == "strength":
        size_metric = strength_dict
    else:
        size_metric = None

    if size_metric:
        metric_values = np.array(list(size_metric.values()), dtype=float)
        metric_min = metric_values.min()
        metric_max = metric_values.max()

        if metric_max == metric_min:
            uniform_size = float(np.mean(node_size_range))
            node_sizes = {node: uniform_size for node in B.nodes}
        else:
            low, high = node_size_range
            node_sizes = {
                node: low + (size_metric[node] - metric_min) * (high - low) / (metric_max - metric_min)
                for node in B.nodes
            }
    else:
        node_sizes = {node: default_node_size for node in B.nodes}

    nx.draw_networkx_edges(
        B,
        pos,
        ax=ax,
        edge_color=weights if weights else "0.7",
        edge_cmap=plt.get_cmap(edge_cmap),
        width=edge_widths,
        alpha=edge_alpha,
    )

    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=group0_nodes,
        node_color=exporter_color,
        node_size=[node_sizes[node] for node in group0_nodes],
        ax=ax,
        linewidths=0.8,
        edgecolors="white",
    )

    nx.draw_networkx_nodes(
        B,
        pos,
        nodelist=group1_nodes,
        node_color=importer_color,
        node_size=[node_sizes[node] for node in group1_nodes],
        ax=ax,
        linewidths=0.8,
        edgecolors="white",
    )

    ranking_metric = size_metric if size_metric is not None else degree_dict
    top_exporters = sorted(group0_nodes, key=lambda node: ranking_metric[node], reverse=True)[:label_top_n]
    top_importers = sorted(group1_nodes, key=lambda node: ranking_metric[node], reverse=True)[:label_top_n]

    def _spread_label_positions(nodes):
        if not nodes:
            return {}

        sorted_nodes = sorted(nodes, key=lambda node: pos[node][1])
        label_y = {}
        previous_y = None

        for node in sorted_nodes:
            current_y = pos[node][1]
            if previous_y is None:
                adjusted_y = current_y
            else:
                adjusted_y = max(current_y, previous_y + label_min_gap)
            label_y[node] = adjusted_y
            previous_y = adjusted_y

        top_limit = max(y_values) if y_values else None
        bottom_limit = min(y_values) if y_values else None

        if top_limit is not None and sorted_nodes:
            overflow = label_y[sorted_nodes[-1]] - top_limit
            if overflow > 0:
                for node in sorted_nodes:
                    label_y[node] -= overflow

        if bottom_limit is not None and sorted_nodes:
            underflow = bottom_limit - label_y[sorted_nodes[0]]
            if underflow > 0:
                for node in sorted_nodes:
                    label_y[node] += underflow

        return label_y

    y_values = [coords[1] for coords in pos.values()]
    exporter_label_y = _spread_label_positions(top_exporters)
    importer_label_y = _spread_label_positions(top_importers)

    for node in top_exporters:
        x, y = pos[node]
        label_y = exporter_label_y[node]
        ax.text(
            x - label_offset,
            label_y,
            str(node),
            fontsize=font_size,
            color="black",
            ha="right",
            va="center",
            clip_on=False,
        )
        if draw_label_connectors and abs(label_y - y) > 1e-9:
            ax.plot(
                [x - 0.01, x - label_offset + 0.01],
                [y, label_y],
                color="0.5",
                linewidth=0.8,
                alpha=0.8,
                solid_capstyle="round",
                zorder=1,
            )

    for node in top_importers:
        x, y = pos[node]
        label_y = importer_label_y[node]
        ax.text(
            x + label_offset,
            label_y,
            str(node),
            fontsize=font_size,
            color="black",
            ha="left",
            va="center",
            clip_on=False,
        )
        if draw_label_connectors and abs(label_y - y) > 1e-9:
            ax.plot(
                [x + 0.01, x + label_offset - 0.01],
                [y, label_y],
                color="0.5",
                linewidth=0.8,
                alpha=0.8,
                solid_capstyle="round",
                zorder=1,
            )

    ax.set_xlim(-partition_gap / 2 - x_margin_left, partition_gap / 2 + x_margin_right)

    if y_values:
        ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)

    if title:
        ax.set_title(title, fontsize=title_font_size, pad=16)

    if not show_axis:
        ax.set_axis_off()

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)

    return ax


def plot_degree_bar(df, country_col="Reporter Country", degree_col="Degree", 
                    title="Node Degree", xlabel="Country", ylabel="Degree", 
                    color="blue", alpha=0.7, figsize=(12, 6), rotation=90):
    """
    Plot a bar chart of node degrees (e.g., exporters or importers) in a bipartite network.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing node information with degree values.
    country_col : str
        Name of the column with country or node names.
    degree_col : str
        Name of the column with degree values.
    title : str
        Title of the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    color : str
        Color used for the bars.
    alpha : float
        Transparency level for the bars (0 to 1).
    figsize : tuple
        Size of the figure in inches (width, height).
    rotation : int
        Rotation angle of the x-axis tick labels.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object of the created plot.
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
    Plot side-by-side scatter plots comparing the degree of reporter and partner countries.

    Parameters
    ----------
    df_reporters : pandas.DataFrame
        DataFrame containing degree values for reporter (exporter) nodes.
    df_partners : pandas.DataFrame
        DataFrame containing degree values for partner (importer) nodes.
    reporter_country_col : str
        Column name for reporter (exporter) country names.
    partner_country_col : str
        Column name for partner (importer) country names.
    degree_col : str
        Column name containing the degree values.
    figsize : tuple
        Size of the entire figure in inches (width, height).
    reporter_color : str
        Color used for the reporter scatter plot.
    partner_color : str
        Color used for the partner scatter plot.
    alpha : float
        Transparency level for the scatter points.
    rotation : int
        Rotation angle for x-axis tick labels.
    use_log_scale : bool
        If True, apply logarithmic scale to the y-axis.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the two subplots.
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
    Plot degree values of reporter and partner countries sorted by rank in descending order.

    Parameters
    ----------
    df_reporters : pandas.DataFrame
        DataFrame containing degree values for reporter (exporter) nodes.
    df_partners : pandas.DataFrame
        DataFrame containing degree values for partner (importer) nodes.
    reporter_label : str
        Label for reporter nodes (used in legend).
    partner_label : str
        Label for partner nodes (used in legend).
    degree_col : str
        Column name containing degree values.
    figsize : tuple
        Size of the figure in inches (width, height).
    reporter_color : str
        Color used for reporter points and fit line.
    partner_color : str
        Color used for partner points and fit line.
    alpha : float
        Transparency level for the scatter points.
    use_log_y : bool
        Whether to use log scale for the y-axis.
    use_log_x : bool
        Whether to use log scale for the x-axis.
    title : str
        Title of the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object of the plot.
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


def plot_degree_rank_multiyear(
    exporter_data,
    importer_data,
    years=None,
    degree_col="Degree",
    figsize=(16, 6),
    colors=None,
    marker="o",
    linewidth=2.2,
    markersize=5.5,
    alpha=0.95,
    use_log_y=True,
    use_log_x=False,
    exporter_title="Exporters",
    importer_title="Importers",
    suptitle=None,
    xlabel="Rank",
    ylabel="Degree",
    legend_title="Year",
    title_font_size=18,
    panel_title_font_size=15,
    label_font_size=13,
    tick_font_size=11,
    legend_font_size=11,
    grid_alpha=0.25,
    save_path=None,
    save_dpi=300,
    save_bbox_inches="tight",
):
    """
    Plot rank-degree curves for multiple years in two side-by-side panels:
    exporters on the left and importers on the right.

    Parameters
    ----------
    exporter_data : dict
        Dictionary mapping year -> DataFrame for exporter nodes.
    importer_data : dict
        Dictionary mapping year -> DataFrame for importer nodes.
    years : list, optional
        Ordered list of years to plot. If None, uses the sorted intersection
        of years present in both dictionaries.
    degree_col : str
        Column name containing the degree-like metric to plot.
    figsize : tuple
        Figure size in inches.
    colors : list or None
        List of colors to use for the yearly curves.
    marker : str
        Marker style for the lines.
    linewidth : float
        Line width for the curves.
    markersize : float
        Marker size for the curves.
    alpha : float
        Transparency level for the curves.
    use_log_y : bool
        Whether to use logarithmic scaling on the y-axis.
    use_log_x : bool
        Whether to use logarithmic scaling on the x-axis.
    exporter_title : str
        Title of the exporter panel.
    importer_title : str
        Title of the importer panel.
    suptitle : str, optional
        Figure-level title.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    legend_title : str
        Title for the legend.
    title_font_size : int
        Font size for the figure title.
    panel_title_font_size : int
        Font size for panel titles.
    label_font_size : int
        Font size for axis labels.
    tick_font_size : int
        Font size for axis tick labels.
    legend_font_size : int
        Font size for the legend.
    grid_alpha : float
        Transparency of the grid lines.
    save_path : str or None
        If provided, save the figure to this path.
    save_dpi : int
        Resolution used when saving the figure.
    save_bbox_inches : str
        Bounding box option passed to `savefig`.

    Returns
    -------
    tuple
        (fig, axes) where axes contains the two subplot axes.
    """
    if years is None:
        years = sorted(set(exporter_data).intersection(importer_data))

    if not years:
        raise ValueError("No overlapping years were provided in exporter_data and importer_data.")

    if colors is None:
        colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#000000"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)

    panel_specs = [
        (axes[0], exporter_data, exporter_title),
        (axes[1], importer_data, importer_title),
    ]

    for ax, data_dict, panel_title in panel_specs:
        for idx, year in enumerate(years):
            if year not in data_dict:
                continue

            df_sorted = data_dict[year].sort_values(by=degree_col, ascending=False).reset_index(drop=True)
            ranks = range(1, len(df_sorted) + 1)

            ax.plot(
                ranks,
                df_sorted[degree_col],
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                alpha=alpha,
                color=colors[idx % len(colors)],
                label=str(year),
            )

        if use_log_y:
            ax.set_yscale("log")
        if use_log_x:
            ax.set_xscale("log")

        ax.set_title(panel_title, fontsize=panel_title_font_size, pad=10)
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        ax.set_ylabel(ylabel, fontsize=label_font_size)
        ax.tick_params(axis="both", labelsize=tick_font_size)
        ax.grid(True, which="major", alpha=grid_alpha)
        ax.legend(title=legend_title, fontsize=legend_font_size, title_fontsize=legend_font_size)

    if suptitle:
        fig.suptitle(suptitle, fontsize=title_font_size, y=1.02)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)

    return fig, axes


def plot_weight_matrix(df, row="Partner Countries", col="Reporter Countries", 
                       value="Value", cmap="coolwarm", figsize=(20, 15), 
                       title="Weighted Adjacency Matrix (Trade Volume)",
                       save_path=None, save_dpi=300, save_bbox_inches="tight"):
    """
    Plot a heatmap of the weighted bipartite adjacency matrix.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing filtered trade data with exporter, importer, and weight columns.
    row : str
        Column name to use as rows of the matrix (typically importers).
    col : str
        Column name to use as columns of the matrix (typically exporters).
    value : str
        Column containing the weight or value of the trade relationship.
    cmap : str
        Colormap used for the heatmap.
    figsize : tuple
        Size of the figure in inches (width, height).
    title : str
        Title of the plot.
    save_path : str or None
        If provided, save the figure to this path.
    save_dpi : int
        Resolution used when saving the figure.
    save_bbox_inches : str
        Bounding box option passed to `savefig`.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes object of the resulting heatmap.
    """
    # Build matrix
    matrix = df.pivot(index=row, columns=col, values=value)

    # Sort rows/cols by total weights
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index,
                        matrix.sum(axis=0).sort_values(ascending=False).index]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, cmap=cmap, annot=False, linewidths=0.5, ax=ax)

    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_title(title)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)

    plt.show()

    return ax


def plot_top_betweenness(df, col, title=None, color="steelblue", top_n=10, label_col="node", xlabel="Betweenness Centrality"):
    """
    Plot a horizontal bar chart of the top N nodes ranked by betweenness centrality.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing betweenness values and node labels.
    col : str
        Column name containing betweenness centrality scores.
    title : str, optional
        Title of the plot (default is None).
    color : str
        Color of the bars in the chart.
    top_n : int
        Number of top-ranking nodes to display.
    label_col : str
        Column name with node identifiers (default is 'node').
    xlabel : str
        Label for the x-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object of the plot.
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


def plot_betweenness_heatmap(
    betweenness_data,
    metric_col,
    years=None,
    label_col="node",
    mode="rank",
    top_n=10,
    nodes=None,
    ascending=False,
    cmap="YlOrRd",
    figsize=(8, 6),
    title=None,
    cbar_label=None,
    annot=True,
    fmt=".0f",
    save_path=None,
    save_dpi=300,
    save_bbox_inches="tight",
):
    """
    Plot a heatmap showing the evolution of betweenness values or ranks across years.

    Parameters
    ----------
    betweenness_data : dict
        Dictionary mapping year -> DataFrame containing betweenness results.
    metric_col : str
        Column name with the betweenness metric to visualize.
    years : list, optional
        Ordered list of years to include. If None, uses sorted keys.
    label_col : str
        Column name with node labels.
    mode : {"rank", "value"}
        Whether to plot within-year rank or raw metric values. Rank=1 is the most central. 
    top_n : int
        Number of nodes to include when `nodes` is not provided.
    nodes : list or None
        Specific node labels to include. If None, selects the top nodes by average score.
    ascending : bool
        Sorting direction for metric values when computing ranks.
    cmap : str
        Colormap for the heatmap.
    figsize : tuple
        Figure size in inches.
    title : str, optional
        Title of the plot.
    cbar_label : str, optional
        Label for the color bar.
    annot : bool
        Whether to annotate heatmap cells.
    fmt : str
        Format string for annotations.
    save_path : str or None
        If provided, save the figure to this path.
    save_dpi : int
        Resolution used when saving the figure.
    save_bbox_inches : str
        Bounding box option passed to `savefig`.

    Returns
    -------
    tuple
        (ax, matrix) where `matrix` is the DataFrame displayed in the heatmap.
    """
    if years is None:
        years = sorted(betweenness_data)

    if not years:
        raise ValueError("No years were provided in betweenness_data.")

    frames = []
    for year in years:
        if year not in betweenness_data:
            continue

        df_year = betweenness_data[year][[label_col, metric_col]].copy()
        df_year["year"] = year

        if mode == "rank":
            df_year["display_value"] = df_year[metric_col].rank(
                method="min",
                ascending=ascending
            )
        elif mode == "value":
            df_year["display_value"] = df_year[metric_col]
        else:
            raise ValueError("mode must be either 'rank' or 'value'.")

        frames.append(df_year)

    combined = pd.concat(frames, ignore_index=True)

    if nodes is None:
        selector = (
            combined.groupby(label_col)["display_value"]
            .mean()
            .sort_values(ascending=(mode == "rank"))
        )
        selected_nodes = list(selector.head(top_n).index)
    else:
        selected_nodes = list(nodes)

    matrix = (
        combined[combined[label_col].isin(selected_nodes)]
        .pivot(index=label_col, columns="year", values="display_value")
        .reindex(selected_nodes)
    )

    if mode == "rank":
        fmt = ".0f"
        if cbar_label is None:
            cbar_label = "Rank"
    elif cbar_label is None:
        cbar_label = metric_col

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap=cmap,
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        cbar_kws={"label": cbar_label},
        ax=ax,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.set_title(title or f"Betweenness {mode.capitalize()} Heatmap")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)

    return ax, matrix


def plot_mean_clustering_ratio_vs_degree(
    df,
    degree_col="degree",
    ratio_col="C4_rate",
    type_col="tipo",
    node_col="node",
    show_labels=False,
    label_max_names_per_line=3,
    label_font_size=6,
    exporter_label="Exporters",
    importer_label="Importers",
    save_path=None,
    save_dpi=300,
    save_bbox_inches="tight",
):
    """
    Plot the mean clustering ratio ⟨C4b^w / C4b⟩ versus node degree for each node type.

    The function groups nodes by degree and computes the average clustering ratio per group,
    optionally displaying node labels.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the clustering ratio, node degree, type and identifier columns.
    degree_col : str
        Column name with node degrees.
    ratio_col : str
        Column name with clustering ratio (e.g., C4b^w / C4b).
    type_col : str
        Column name indicating node type (e.g., 'Exporter' or 'Importer').
    node_col : str
        Column name with node identifiers (used for optional annotations).
    show_labels : bool
        Whether to annotate each point with its corresponding node names.
    label_max_names_per_line : int
        Maximum number of node names to place on each line of an annotation.
    label_font_size : int
        Font size used for annotations.
    exporter_label : str
        Label used in the legend for exporter nodes.
    importer_label : str
        Label used in the legend for importer nodes.
    save_path : str or None
        If provided, save the figure to this path.
    save_dpi : int
        Resolution used when saving the figure.
    save_bbox_inches : str
        Bounding box option passed to `savefig`.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object of the plot.
    """
    # Group by type and degree
    grouped = (
        df.groupby([type_col, degree_col])
        .agg({
            ratio_col: "mean",
            node_col: lambda x: ', '.join(x)
        })
        .reset_index()
        .rename(columns={node_col: "nodos"})
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each type separately
    for tipo in grouped[type_col].unique():
        subset = grouped[grouped[type_col] == tipo]
        legend_label = exporter_label if tipo.lower().startswith("export") else importer_label
        ax.plot(subset[degree_col], subset[ratio_col],
                label=legend_label,
                marker='o' if tipo.lower().startswith("export") else 's',
                linestyle='-')

        if show_labels:
            x_offset = -8 if tipo.lower().startswith("export") else 8
            y_offset = 6 if tipo.lower().startswith("export") else -6
            for _, row in subset.iterrows():
                node_names = [name.strip() for name in row["nodos"].split(",") if name.strip()]
                label_lines = [
                    ", ".join(node_names[i:i + label_max_names_per_line])
                    for i in range(0, len(node_names), label_max_names_per_line)
                ]
                label_text = "\n".join(label_lines)
                ax.annotate(
                    label_text,
                    (row[degree_col], row[ratio_col]),
                    xytext=(x_offset, y_offset),
                    textcoords="offset points",
                    fontsize=label_font_size,
                    ha="right" if tipo.lower().startswith("export") else "left",
                    va="bottom" if tipo.lower().startswith("export") else "top",
                )

    # Labels and styling
    ax.set_xlabel("Degree")
    ax.set_ylabel("⟨C4b^w / C4b⟩")
    ax.set_title("Mean clustering ratio ⟨C4b^w / C4b⟩ vs. Degree by node type")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)

    return ax


def plot_clustering_ratio_multiyear(
    clustering_data,
    years=None,
    degree_col="degree",
    ratio_col="C4_rate",
    type_col="tipo",
    exporter_label="Exporter",
    importer_label="Importer",
    figsize=(16, 6),
    colors=None,
    marker="o",
    linewidth=2.2,
    markersize=5.5,
    alpha=0.95,
    use_log_x=False,
    use_log_y=False,
    suptitle=None,
    exporter_title="Exporters",
    importer_title="Importers",
    xlabel="Degree",
    ylabel="Mean weighted/unweighted clustering ratio",
    legend_title="Year",
    title_font_size=18,
    panel_title_font_size=15,
    label_font_size=13,
    tick_font_size=11,
    legend_font_size=11,
    grid_alpha=0.25,
    save_path=None,
    save_dpi=300,
    save_bbox_inches="tight",
):
    """
    Plot the mean clustering ratio versus degree for multiple years in two panels:
    exporters on the left and importers on the right.

    Parameters
    ----------
    clustering_data : dict
        Dictionary mapping year -> DataFrame produced by `compute_bipartite_clustering`.
    years : list, optional
        Ordered list of years to plot. If None, uses the sorted keys of `clustering_data`.
    degree_col : str
        Column name with node degree.
    ratio_col : str
        Column name with clustering ratio.
    type_col : str
        Column name indicating node type.
    exporter_label : str
        Label used to identify exporter rows in `type_col`.
    importer_label : str
        Label used to identify importer rows in `type_col`.
    figsize : tuple
        Figure size in inches.
    colors : list or None
        List of colors to use for the yearly curves.
    marker : str
        Marker style for the lines.
    linewidth : float
        Line width for the curves.
    markersize : float
        Marker size for the curves.
    alpha : float
        Transparency level for the curves.
    use_log_x : bool
        Whether to use logarithmic scaling on the x-axis.
    use_log_y : bool
        Whether to use logarithmic scaling on the y-axis.
    suptitle : str, optional
        Figure-level title.
    exporter_title : str
        Title of the exporter panel.
    importer_title : str
        Title of the importer panel.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    legend_title : str
        Title for the legend.
    title_font_size : int
        Font size for the figure title.
    panel_title_font_size : int
        Font size for panel titles.
    label_font_size : int
        Font size for axis labels.
    tick_font_size : int
        Font size for axis tick labels.
    legend_font_size : int
        Font size for the legend.
    grid_alpha : float
        Transparency of the grid lines.
    save_path : str or None
        If provided, save the figure to this path.
    save_dpi : int
        Resolution used when saving the figure.
    save_bbox_inches : str
        Bounding box option passed to `savefig`.

    Returns
    -------
    tuple
        (fig, axes) where axes contains the two subplot axes.
    """
    if years is None:
        years = sorted(clustering_data)

    if not years:
        raise ValueError("No years were provided in clustering_data.")

    if colors is None:
        colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#000000"]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    panel_specs = [
        (axes[0], exporter_label, exporter_title),
        (axes[1], importer_label, importer_title),
    ]

    for ax, node_type, panel_title in panel_specs:
        for idx, year in enumerate(years):
            if year not in clustering_data:
                continue

            df_year = clustering_data[year]
            subset = df_year[df_year[type_col] == node_type].copy()
            grouped = (
                subset.groupby(degree_col)[ratio_col]
                .mean()
                .reset_index()
                .sort_values(by=degree_col)
            )

            if grouped.empty:
                continue

            ax.plot(
                grouped[degree_col],
                grouped[ratio_col],
                marker=marker,
                linewidth=linewidth,
                markersize=markersize,
                alpha=alpha,
                color=colors[idx % len(colors)],
                label=str(year),
            )

        if use_log_x:
            ax.set_xscale("log")
        if use_log_y:
            ax.set_yscale("log")

        ax.set_title(panel_title, fontsize=panel_title_font_size, pad=10)
        ax.set_xlabel(xlabel, fontsize=label_font_size)
        ax.set_ylabel(ylabel, fontsize=label_font_size)
        ax.tick_params(axis="both", labelsize=tick_font_size)
        ax.grid(True, which="major", alpha=grid_alpha)
        ax.legend(title=legend_title, fontsize=legend_font_size, title_fontsize=legend_font_size)

    if suptitle:
        fig.suptitle(suptitle, fontsize=title_font_size, y=1.02)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=save_dpi, bbox_inches=save_bbox_inches)

    return fig, axes
