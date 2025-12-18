import pandas as pd

from utils import (
    filter_by_parent_edu,
    make_long,
    map_and_order_edu,
    compute_percent
)

# Fixtures 

def dummy_df():
    return pd.DataFrame({
        "Mother_edu_code": [2, 3, 2, 5],
        "Father_edu_code": [2, 3, 5, 5],
        "Target": ["Graduate", "Dropout", "Graduate", "Dropout"],
        "Age": [20, 21, 22, 23]
    })
   
# Tests

# Testing filtering function
def test_filter_by_parent_edu():
    df = dummy_df()
    filtered = filter_by_parent_edu(df, mother_code=2, father_code=2)

    assert len(filtered) == 1
    assert filtered.iloc[0]["Mother_edu_code"] == 2
    assert filtered.iloc[0]["Father_edu_code"] == 2

# Testing the long_df function
def test_make_long():
    df = dummy_df()
    long_df = make_long(df, value_col="Age")

    # Row count should double
    assert len(long_df) == len(df) * 2

    # Required columns
    assert {"Parent", "Edu_code", "Age"}.issubset(long_df.columns)

    # Parent labels
    assert set(long_df["Parent"]) == {"Mother", "Father"}

# Testing the mapping function
def test_map_and_order_edu():
    df = pd.DataFrame({
        "Edu_code": [2, 3, 5]
    })

    labels = {
        2: "Elementary",
        3: "Middle",
        5: "High"
    }

    order = ["Elementary", "Middle", "High"]

    mapped = map_and_order_edu(
        df,
        edu_col="Edu_code",
        label_col="Education Level",
        labels=labels,
        order=order
    )

    assert mapped["Education Level"].dtype.name == "category"
    assert list(mapped["Education Level"].cat.categories) == order

def test_compute_percent_sums_to_100():
    df = pd.DataFrame({
        "Parent": ["Mother", "Mother", "Father", "Father"],
        "Age": [20, 21, 22, 23],
        "Edu_code": [2, 3, 5, 5],
        "Education Level": ["Test Level"] * 4,
        "Target": ["A", "B", "A", "B"]  
    })

    percent_df = compute_percent(df, target_col="Target")

    # Ensure all percentages sum to 100 within each group
    grouped = percent_df.groupby(["Parent", "Education Level"])["Percent"]
    for _, group in grouped:
        total = group.sum()
        assert round(total, 5) == 100.0, f"Percent sum = {total}, expected 100"

    # Check that all columns exist
    for col in ["Parent", "Education Level", "Target", "Percent"]:
        assert col in percent_df.columns, f"Missing column: {col}"

