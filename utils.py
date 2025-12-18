# utils.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Functions to explore data

def print_counts(df, col, label):
    """
    Computes the value counts

    Parameters:
    - df: DataFrame
    - col: The column
    - label: Label to print the value counts

    Returns:
    - counts: The value counts Series
    """
    counts = df[col].value_counts()

    print(f"{label} Count")
    print(counts)
    print()

    return counts


def filter_by_parent_edu(df, mother_code, father_code=None):
    """
    Filters the DataFrame based on the education codes

    Parameters:
    - df: DataFrame
    - mother_code: The mother's education code
    - father_code: The father's education code

    Returns:
    - df: DataFrame with the filtered education codes
    """
    if father_code is None:
        father_code = mother_code

    return df[
        (df["Mother_edu_code"] == mother_code) &
        (df["Father_edu_code"] == father_code)
    ]


def filter_one_parent(df, parent, parent_code):
    """
    Filters the DataFrame based on a single parent's education code

    Parameters:
    - df: DataFrame
    - parent: The parent column to filter for
    - parent_code: The parent's education code 

    Returns:
    - df: DataFrame with the filtered education code
    """
    return df[df[parent] == parent_code]


def print_percent(count_series, parent_label, parent_edu_label, pos_label, neg_label):
    """
    Computes and prints percentages from a count Series

    Parameters:
    - count_series: Series containing counts (two values expected)
    - parent_label: The parent you are computing for
    - parent_edu_label: The parent's education label
    - pos_label: The label for the first value in the series
    - neg_label: The label for the second value in the series

    Returns:
    - None: Prints out statements of the percentages
    """
    total = count_series.iloc[0] + count_series.iloc[1]

    zero_percent = (count_series.iloc[0] / total) * 100
    one_percent = (count_series.iloc[1] / total) * 100

    print(f"Students whose {parent_label} completed {parent_edu_label} and {pos_label}: {zero_percent:.2f}%")
    print(f"Students whose {parent_label} completed {parent_edu_label} and {neg_label}: {one_percent:.2f}%")
    print()



# Education labels & ordering

EDU_LABELS = {
    2: "Some Level of Elementary School",
    3: "Completed Elementary School",
    4: "Completed Middle School",
    5: "Completed High School",
    7: "Completed their Bachelors Degree"
}

EDU_ORDER = [
    "Some Level of Elementary School",
    "Completed Elementary School",
    "Completed Middle School",
    "Completed High School",
    "Completed their Bachelors Degree"
]


# Color palettes

OUTCOME_PALETTE = {"Dropout": "red", "Graduate": "green"}
MEAN_MEDIAN_PALETTE = {"Mean": "steelblue", "Median": "lightblue"}
TUITION_PALETTE = {"To date": "lightcoral", "Not to date": "firebrick"}
DEBT_PALETTE = {"Has Debt": "darkorange", "No Debt": "lemonchiffon"}
SCHOLARSHIP_PALETTE = {"Scholarship": "mediumpurple", "No Scholarship": "thistle"}


def make_long(df, value_col, parent_cols=["Mother_edu_code", "Father_edu_code"]):
    """
    Converts a wide DataFrame with Mother and Father columns into a long format for plotting.

    Parameters:
    - df: original wide DataFrame
    - value_cols: list of columns to keep
    - mother_col: column for mother's education code
    - father_col: column for father's education code

    Returns:
    - long_df: long-format DataFrame with columns:
        - Parent ("Mother" / "Father")
        - Edu_code (education code)
        - All value_cols
    """
    long_df = pd.concat([
        df[[value_col, parent_cols[0]]]
            .rename(columns={parent_cols[0]: "Edu_code"})
            .assign(Parent="Mother"),
        df[[value_col, parent_cols[1]]]
            .rename(columns={parent_cols[1]: "Edu_code"})
            .assign(Parent="Father")
    ])
    return long_df

def map_and_order_edu(df, edu_col="Edu_code", label_col="Education Level",
                      labels=None, order=None):
    """
    Maps the dataframe from the code to the label 
    
    Parameters:
    - df: original DataFrame
    - edu_code: The codes for the education level
    - label_col: The education level label

    Returns:
    - df: dataframe with the values mapped
    """
    df[label_col] = df[edu_col].map(labels)
    df[label_col] = pd.Categorical(df[label_col], categories=order, ordered=True)
    return df

    
def compute_summary_stats(df, value_col):
    """
    Computes summary statistics for the plots

    Parameters:
    - df: dataframe
    - value_cols: list of columns to compute stats on

    Returns:
    - summary_long: Maps the Mean and Median values
    """
    summary_df = df.groupby(["Parent", "Education Level"], observed=True).agg(
        Mean_Age=(value_col, "mean"),
        Median_Age=(value_col, "median")
    ).reset_index()

    summary_long = summary_df.melt(
        id_vars=["Parent", "Education Level"],
        value_vars=["Mean_Age", "Median_Age"],
        var_name="Statistic",
        value_name=value_col
    )

    summary_long["Statistic"] = summary_long["Statistic"].map({"Mean_Age": "Mean", "Median_Age": "Median"})

    return summary_long


def compute_percent(df, target_col="Target"):
    """
    Computes the percentages for the labels
    
    Parameters:
    - df: DataFrame
    - target_col: The "Target" column

    Returns:
    - counts: The percentage that Dropped out or Graduated
    """
    counts = (
        df.groupby(["Parent", "Education Level", target_col], observed=True)
        .size()
        .reset_index(name="Count")
    )
    counts["Percent"] = counts["Count"] / counts.groupby(["Parent", "Education Level"], observed=True)["Count"].transform("sum") * 100
    return counts

    
def plot_bar(df, x, y, hue=None, col=None, title="", ylabel="", xlabel="",
             rotate_x=30, show_values=False, percent=False, save_path=None, palette=None):
    """
    Plots a Seaborn bar chart with options for custom palette, black outlines, values, and facets.
    """
    g = sns.catplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        col=col,
        kind="bar",
        height=5,
        aspect=1.3,
        errorbar=None,
        palette=palette
    )

    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=rotate_x)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel(xlabel)

        # Black outlines for bars
        for patch in ax.patches:
            patch.set_edgecolor("black")
            patch.set_linewidth(1)

        # Values on top
        if show_values:
            for p in ax.patches:
                height = p.get_height()
                if height > 0:
                    label = f"{height:.1f}%" if percent else f"{height:.1f}"
                    ax.text(
                        p.get_x() + p.get_width()/2,
                        height + 0.5,
                        label,
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )

    g.set_titles("{col_name}")
    g.fig.suptitle(title, fontsize=16, y=1.03)

    # Legend outside
    if hue is not None:
        sns.move_legend(
            g,
            "center right",
            bbox_to_anchor=(1.05, 0.5),
            title=hue
        )

    plt.tight_layout()

    if save_path:
        g.fig.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()
    return g
