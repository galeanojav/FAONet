import pandas as pd

def load_and_merge_csv(filepaths):
    """Load and concatenate multiple FAOSTAT CSV files.

    Args:
        filepaths (list of str): Paths to the CSV files.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    dataframes = [pd.read_csv(path) for path in filepaths]
    return pd.concat(dataframes, ignore_index=True)

def save_dataframe(df, filepath):
    """Save a DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filepath (str): Path to the output file.
    """
    df.to_csv(filepath, index=False)
