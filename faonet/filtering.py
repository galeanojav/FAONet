def filter_top_percentile(df, value_column="Value", percentile=0.9):
    """Filter the DataFrame by cumulative value percentile.

    Args:
        df (pd.DataFrame): Original DataFrame.
        value_column (str): Column with values.
        percentile (float): Accumulated threshold (0 < percentile <= 1).

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    df_sorted = df.sort_values(by=value_column, ascending=False)
    total_value = df_sorted[value_column].sum()
    df_sorted["cumsum"] = df_sorted[value_column].cumsum()
    df_sorted["cumperc"] = df_sorted["cumsum"] / total_value
    return df_sorted[df_sorted["cumperc"] <= percentile].copy()
