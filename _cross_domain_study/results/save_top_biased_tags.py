import pandas as pd
import numpy as np
import os

# List of datasets
datasets = ["celeba", "waterbirds", "urbancars", "imagenet9"]

# Process each dataset
for dataset in datasets:
    # Define file path
    file_path = f"./data/{dataset}/overperforming_tags.csv"

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    # Load CSV
    df = pd.read_csv(file_path)

    # Compute weighted score
    df["Score"] = df["Acc"] * np.log1p(df["Samples"])

    # Get top-10 tags per class
    top_tags = (
        df.groupby("Class")
        .apply(lambda x: x.nlargest(20, "Score"))
        .reset_index(drop=True)
    )

    # Save results
    output_path = f"./data/{dataset}/top_biased_tags.csv"
    top_tags.to_csv(output_path, index=False)

    print(f"Processed {dataset}: saved top tags to {output_path}")
