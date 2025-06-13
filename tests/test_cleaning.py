import pandas as pd
from app.utils.cleaning import clean_and_merge_files

def test_clean_and_merge():
    df1 = pd.DataFrame({"airlie_id": [1], "val": [2]})
    df2 = pd.DataFrame({"airlie_id": [1], "name": ["A"]})
    merged = clean_and_merge_files(df1, df2)
    assert "name" in merged.columns