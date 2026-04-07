from collections import defaultdict
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from torch.utils.data.sampler import WeightedRandomSampler
from my_datasets.utils import get_sampling_weights
from datasets import load_dataset


class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, biases=None, embeddings=None):
        self.biases = biases
        self.texts = texts
        self.targets = labels
        self.embeddings = embeddings

        if self.embeddings is not None:
            # Ensure embeddings are a PyTorch tensor
            self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)

        # Ensure labels are a PyTorch tensor and unsqueezed for BCEWithLogitsLoss
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        if self.biases is not None:
            # Ensure embeddings are a PyTorch tensor
            self.biases = torch.tensor(self.biases, dtype=torch.long)
        else:
            self.biases = self.targets  # workaround in order not to have nans in biases

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # Return embeddings if available, otherwise return text for on-the-fly encoding
        input_data = (
            self.embeddings[idx] if self.embeddings is not None else self.texts[idx]
        )

        return {
            "index": idx,
            "inputs": input_data,
            "targets": self.targets[idx],
            "bias": self.biases[idx],
        }


def compute_or_load_embeddings(texts, split_name, model, cache_dir, device):
    # os.makedirs(cache_dir, exist_ok=True)
    # emb_path = os.path.join(cache_dir, f"{split_name}_embeddings.npy")

    # if os.path.exists(emb_path):
    #     print(f"Loading cached embeddings for '{split_name}' from: {emb_path}")
    #     embeddings = np.load(emb_path)
    # else:
    #     print(f"Generating embeddings for '{split_name}'...")
    #     embeddings = model.encode(
    #         texts,
    #         show_progress_bar=True,
    #         convert_to_numpy=True,
    #         device=str(device),
    #     )
    # np.save(emb_path, embeddings)
    # print(f"Saved embeddings to: {emb_path}")
    print(f"Generating embeddings for '{split_name}'...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        device=str(device),
    )
    return embeddings


def get_bias_in_bios_loaders(
    root,
    target="profession",
    bias="gender",
    batch_size=128,
    num_workers=4,
    encoder_name="all-MiniLM-L6-v2",
    sampler=None,
):

    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Data Loading ---

    train_dataset = load_dataset("LabHC/bias_in_bios", split="train", cache_dir=root)
    X_train_text = train_dataset["hard_text"]
    y_train = train_dataset[target]
    b_train = train_dataset[bias]

    test_dataset = load_dataset("LabHC/bias_in_bios", split="test", cache_dir=root)
    X_test_text = test_dataset["hard_text"]
    y_test = test_dataset[target]
    b_test = test_dataset[bias]

    val_dataset = load_dataset("LabHC/bias_in_bios", split="dev", cache_dir=root)
    X_val_text = val_dataset["hard_text"]
    y_val = val_dataset[target]
    b_val = val_dataset[bias]

    print("\n--- Target Class Distribution (Identity_hate) ---")
    print("Training Set:")
    print(pd.Series(y_train).value_counts(normalize=True))
    print("Validation Set:")
    print(pd.Series(y_val).value_counts(normalize=True))
    print("\nTest Set:")
    print(pd.Series(y_test).value_counts(normalize=True))

    # --- 2. Generate Embeddings using Sentence-BERT ---
    print(f"\n--- Loading Sentence-BERT model: '{encoder_name}' for embeddings ---")
    embedding_model = SentenceTransformer(encoder_name)
    embedding_model.to(device)  # Move embedding model to GPU if available
    print("Embedding model loaded successfully.")

    print("\n--- Generating Embeddings (This may take a while) ---")

    X_train_embeddings = compute_or_load_embeddings(
        X_train_text, "train", embedding_model, root, device
    )
    X_val_embeddings = compute_or_load_embeddings(
        X_val_text, "val", embedding_model, root, device
    )
    X_test_embeddings = compute_or_load_embeddings(
        X_test_text, "test", embedding_model, root, device
    )

    # --- 4. PyTorch DataLoaders ---
    print("\n--- Creating PyTorch DataLoaders ---")

    # Create TensorDatasets
    train_dataset = CustomTextDataset(
        X_train_text, y_train, b_train, embeddings=X_train_embeddings
    )
    val_dataset = CustomTextDataset(
        X_val_text, y_val, b_val, embeddings=X_val_embeddings
    )
    test_dataset = CustomTextDataset(
        X_test_text, y_test, b_test, embeddings=X_test_embeddings
    )

    if sampler == "weighted":
        weights = get_sampling_weights(
            train_dataset.targets,
            train_dataset.biases,
        )
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
        )
    else:
        sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    print("DataLoaders created.")
    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )
