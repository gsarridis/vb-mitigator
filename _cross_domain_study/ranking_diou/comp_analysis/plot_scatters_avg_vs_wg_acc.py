import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_method_means_simple(
    filename: str, column1: str, column2: str, dataset_name: str = None
):
    """
    Reads a CSV file (Method, column1, column2), plots a scatter of column1 vs column2
    per method, and saves the figure.

    Args:
        filename (str): Path to CSV file.
        column1 (str): Name of column for X-axis (e.g., 'Avg Acc').
        column2 (str): Name of column for Y-axis (e.g., 'WG Acc').
        dataset_name (str, optional): Name to use for saving the figure. Defaults to filename without extension.
    """

    # Load CSV
    df = pd.read_csv(filename)

    # If no dataset_name provided, use filename without extension
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(filename))[0]

    # Compute mean per method (though each method has only one row, still safe)
    grouped = df.groupby("Method")[[column1, column2]].mean().reset_index()

    print("Mean values per method:")
    print(grouped)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(grouped[column1], grouped[column2])

    # Label each method
    for _, row in grouped.iterrows():
        plt.text(
            row[column1],
            row[column2],
            row["Method"],
            fontsize=10,
            ha="right",
            va="bottom",
        )

    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(f"{column1} vs {column2} per Method — {dataset_name}")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    out_path = os.path.join(
        os.path.dirname(filename), f"plot_wg_vs_avg_acc_{dataset_name}.png"
    )
    plt.savefig(out_path, dpi=300)
    print(f"Figure saved to: {out_path}")

    plt.close()


plot_method_means_simple(
    filename="results_chexpert_nih.csv",
    column1="Avg Acc",
    column2="WG Acc",
    dataset_name="CheXpert_NIH",
)

plot_method_means_simple(
    filename="results_bias_in_bios.csv",
    column1="Avg Acc",
    column2="WG Acc",
    dataset_name="bias_in_bios",
)

plot_method_means_simple(
    filename="results_urbansounds8k.csv",
    column1="Avg Acc",
    column2="WG Acc",
    dataset_name="urbansounds8k",
)

plot_method_means_simple(
    filename="results_accent_archive.csv",
    column1="Avg Acc",
    column2="WG Acc",
    dataset_name="speech_accent_archive",
)

plot_method_means_simple(
    filename="results_jigsaw.csv",
    column1="Official Set Acc",
    column2="Generated Set Acc",
    dataset_name="jigsaw",
)
