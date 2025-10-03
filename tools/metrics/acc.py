import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np

acc_dict = {"best": "high", "performance": "accuracy"}


def acc(data_dict):
    predictions = data_dict["predictions"]
    targets = data_dict["targets"]
    accuracy = (predictions == targets).mean() * 100
    return {"accuracy": accuracy}


def acc_per_class(data_dict):
    predictions = data_dict["predictions"]
    targets = data_dict["targets"]
    accuracy = (predictions == targets).mean() * 100
    # Get unique classes
    classes = np.unique(targets)

    # Initialize a dictionary to store accuracy per class
    acc_dict = {}

    # Calculate accuracy per class
    for i in classes:
        # Find indices of the current class in targets
        class_indices = targets == i

        # Calculate accuracy for this class
        class_accuracy = (
            predictions[class_indices] == targets[class_indices]
        ).mean() * 100
        acc_dict[f"accuracy_{i}"] = class_accuracy
    acc_dict[f"accuracy"] = accuracy

    return acc_dict
