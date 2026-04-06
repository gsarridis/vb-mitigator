#!/usr/bin/env python3
"""
Interactive script to correct model metadata (arch_family, patch_size, training_data).

Usage:
    python correct_model_metadata.py --input results.csv --output corrected_metadata.csv
"""

import argparse
import pandas as pd
import sys
from collections import OrderedDict


def get_user_input(prompt, default=None, existing_options=None):
    """
    Get user input with default value and existing options.

    Args:
        prompt: The prompt to display
        default: Default value (press Enter to accept)
        existing_options: List of existing values to choose from

    Returns:
        User's input or default value (always as string)
    """
    # Convert default to string
    default_str = str(default) if default is not None else ""
    if default_str == "nan" or default_str == "None":
        default_str = ""

    # Display existing options if available
    if existing_options and len(existing_options) > 0:
        print(f"\n  Existing options:")
        for i, opt in enumerate(existing_options, 1):
            print(f"    [{i}] {opt}")

    # Build prompt with default
    if default_str.strip():
        full_prompt = f"{prompt} [default: {default_str}]: "
    else:
        full_prompt = f"{prompt}: "

    user_input = input(full_prompt).strip()

    # Handle empty input (use default)
    if user_input == "":
        return default_str

    # Handle numeric selection from existing options
    if existing_options and user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(existing_options):
            selected = existing_options[idx]
            print(f"  → Selected: {selected}")
            return str(selected)

    return user_input


def main():
    parser = argparse.ArgumentParser(description="Interactively correct model metadata")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file")
    parser.add_argument(
        "--output", "-o", default="corrected_metadata.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--start", "-s", type=int, default=0, help="Start from model index (0-based)"
    )
    args = parser.parse_args()

    # Load CSV
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)

    # Check required columns
    required_cols = ["model_id"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            sys.exit(1)

    # Initialize columns if they don't exist
    for col in ["arch_family", "patch_size", "training_data"]:
        if col not in df.columns:
            df[col] = ""

    # Track unique values seen (using OrderedDict to preserve order)
    seen_arch_families = OrderedDict()
    seen_training_data = OrderedDict()
    seen_patch_sizes = OrderedDict()

    # Pre-populate with existing values
    for val in df["arch_family"].dropna().unique():
        if str(val).strip() and str(val) != "nan":
            seen_arch_families[str(val)] = True
    for val in df["training_data"].dropna().unique():
        if str(val).strip() and str(val) != "nan":
            seen_training_data[str(val)] = True
    for val in df["patch_size"].dropna().unique():
        if str(val).strip() and str(val) != "nan":
            seen_patch_sizes[str(val)] = True

    # Results storage
    results = []

    # Try to load existing progress
    try:
        existing_results = pd.read_csv(args.output)
        processed_ids = set(existing_results["model_id"].tolist())
        results = existing_results.to_dict("records")
        print(f"Loaded {len(results)} previously processed entries from {args.output}")

        # Update seen values from existing results
        for r in results:
            if r.get("arch_family") and str(r["arch_family"]) != "nan":
                seen_arch_families[str(r["arch_family"])] = True
            if r.get("training_data") and str(r["training_data"]) != "nan":
                seen_training_data[str(r["training_data"])] = True
            if r.get("patch_size") and str(r["patch_size"]) != "nan":
                seen_patch_sizes[str(r["patch_size"])] = True
    except FileNotFoundError:
        processed_ids = set()
        print("Starting fresh (no existing output file found)")

    total = len(df)

    print(f"\nTotal models: {total}")
    print(f"Already processed: {len(processed_ids)}")
    print(f"Remaining: {total - len(processed_ids)}")
    print("\nInstructions:")
    print("  - Press Enter to accept the default value")
    print("  - Type a number to select from existing options")
    print("  - Type a new value to enter it")
    print("  - Type 'q' to save and quit")
    print("  - Type 's' to skip this model")
    print("-" * 60)

    skipped = 0

    for idx, row in df.iterrows():
        model_id = row["model_id"]

        # Skip already processed
        if model_id in processed_ids:
            continue

        # Skip if before start index
        if idx < args.start:
            continue

        print(f"\n[{idx + 1}/{total}] Model: {model_id}")
        print(f"  Current values:")
        print(f"    arch_family: {row.get('arch_family', '')}")
        print(f"    patch_size: {row.get('patch_size', '')}")
        print(f"    training_data: {row.get('training_data', '')}")

        # Get arch_family
        arch_options = list(seen_arch_families.keys())
        arch_input = get_user_input(
            "  arch_family",
            default=row.get("arch_family", ""),
            existing_options=arch_options if arch_options else None,
        )

        if arch_input.lower() == "q":
            print("\nSaving and quitting...")
            break
        if arch_input.lower() == "s":
            print("  Skipped.")
            skipped += 1
            continue

        # Get patch_size
        patch_options = list(seen_patch_sizes.keys())
        patch_input = get_user_input(
            "  patch_size",
            default=row.get("patch_size", ""),
            existing_options=patch_options if patch_options else None,
        )

        if patch_input.lower() == "q":
            print("\nSaving and quitting...")
            break
        if patch_input.lower() == "s":
            print("  Skipped.")
            skipped += 1
            continue

        # Get training_data
        data_options = list(seen_training_data.keys())
        data_input = get_user_input(
            "  training_data",
            default=row.get("training_data", ""),
            existing_options=data_options if data_options else None,
        )

        if data_input.lower() == "q":
            print("\nSaving and quitting...")
            break
        if data_input.lower() == "s":
            print("  Skipped.")
            skipped += 1
            continue

        # Store result
        result = {
            "model_id": model_id,
            "arch_family": arch_input,
            "patch_size": patch_input,
            "training_data": data_input,
        }
        results.append(result)
        processed_ids.add(model_id)

        # Update seen values
        if arch_input and str(arch_input) != "nan":
            seen_arch_families[str(arch_input)] = True
        if patch_input and str(patch_input) != "nan":
            seen_patch_sizes[str(patch_input)] = True
        if data_input and str(data_input) != "nan":
            seen_training_data[str(data_input)] = True

        # Auto-save every 10 entries
        if len(results) % 10 == 0:
            results_df = pd.DataFrame(results)
            results_df.to_csv(args.output, index=False)
            print(f"  [Auto-saved {len(results)} entries to {args.output}]")

    # Final save
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(results)} entries to {args.output}")
        print(f"Skipped: {skipped}")
    else:
        print("\nNo entries to save.")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total models in input: {total}")
    print(
        f"Processed in this session: {len(results) - len([r for r in results if r['model_id'] in processed_ids])}"
    )
    print(f"Total in output file: {len(results)}")
    print(f"Remaining: {total - len(processed_ids)}")

    print(f"\nUnique arch_families: {list(seen_arch_families.keys())}")
    print(f"Unique patch_sizes: {list(seen_patch_sizes.keys())}")
    print(f"Unique training_data: {list(seen_training_data.keys())}")


if __name__ == "__main__":
    main()
