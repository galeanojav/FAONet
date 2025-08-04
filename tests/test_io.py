import pandas as pd
from faonet.io import load_and_merge_csv

def test_load_and_merge_csv(tmp_path):
    file1 = tmp_path / "file1.csv"
    file2 = tmp_path / "file2.csv"
    df1 = pd.DataFrame({"A": [1, 2]})
    df2 = pd.DataFrame({"A": [3, 4]})
    df1.to_csv(file1, index=False)
    df2.to_csv(file2, index=False)

    df_merged = load_and_merge_csv([file1, file2])
    assert len(df_merged) == 4
