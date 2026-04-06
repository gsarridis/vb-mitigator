#!/usr/bin/env python3
"""
Merge corrected metadata into master results CSV.

Usage:
    python merge_corrected_metadata.py --master results.csv --corrections corrected_metadata.csv --output updated_results.csv
"""

import argparse
import pandas as pd
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Merge corrected metadata into master CSV"
    )
    parser.add_argument("--master", "-m", required=True, help="Master results CSV file")
    parser.add_argument(
        "--corrections", "-c", required=True, help="Corrected metadata CSV file"
    )
    parser.add_argument("--output", "-o", required=True, help="Output CSV file")
    parser.add_argument(
        "--dry-run", "-n", action="store_true", help="Show changes without saving"
    )
    args = parser.parse_args()

    # Load files
    print(f"Loading master file: {args.master}")
    master_df = pd.read_csv(args.master)
    print(f"  → {len(master_df)} rows")

    print(f"Loading corrections file: {args.corrections}")
    corrections_df = pd.read_csv(args.corrections)
    print(f"  → {len(corrections_df)} rows")

    # Check for model_id column
    if "model_id" not in master_df.columns:
        print("Error: 'model_id' column not found in master file")
        sys.exit(1)
    if "model_id" not in corrections_df.columns:
        print("Error: 'model_id' column not found in corrections file")
        sys.exit(1)

    # Create lookup dict from corrections
    corrections_dict = corrections_df.set_index("model_id").to_dict("index")

    # Track changes
    changes = []
    not_found = []

    # Columns to update
    update_cols = ["arch_family", "patch_size", "training_data"]

    # Ensure columns exist in master
    for col in update_cols:
        if col not in master_df.columns:
            master_df[col] = ""

    # Apply corrections
    for idx, row in master_df.iterrows():
        model_id = row["model_id"]

        if model_id in corrections_dict:
            correction = corrections_dict[model_id]
            model_changes = []

            for col in update_cols:
                if col in correction:
                    old_val = (
                        str(row.get(col, "")) if pd.notna(row.get(col, "")) else ""
                    )
                    new_val = str(correction[col]) if pd.notna(correction[col]) else ""

                    # Skip if new value is empty or same as old
                    if new_val and new_val != "nan" and old_val != new_val:
                        model_changes.append(
                            {"column": col, "old": old_val, "new": new_val}
                        )
                        master_df.at[idx, col] = new_val

            if model_changes:
                changes.append({"model_id": model_id, "changes": model_changes})
        else:
            not_found.append(model_id)

    # Report changes
    print("\n" + "=" * 70)
    print("CHANGES SUMMARY")
    print("=" * 70)

    if changes:
        print(f"\nModels updated: {len(changes)}")
        for change in changes:
            print(f"\n  {change['model_id']}:")
            for c in change["changes"]:
                print(f"    {c['column']}: '{c['old']}' → '{c['new']}'")
    else:
        print("\nNo changes to apply.")

    # Report models in corrections but not in master
    corrections_ids = set(corrections_df["model_id"])
    master_ids = set(master_df["model_id"])
    in_corrections_not_master = corrections_ids - master_ids

    if in_corrections_not_master:
        print(
            f"\n⚠ Models in corrections but not in master ({len(in_corrections_not_master)}):"
        )
        for mid in sorted(in_corrections_not_master):
            print(f"    {mid}")

    # Report models in master but not in corrections
    in_master_not_corrections = master_ids - corrections_ids
    if in_master_not_corrections:
        print(
            f"\n⚠ Models in master but not in corrections ({len(in_master_not_corrections)}):"
        )
        for mid in sorted(in_master_not_corrections):
            print(f"    {mid}")

    # Save or show
    if args.dry_run:
        print(f"\n[DRY RUN] Would save to: {args.output}")
    else:
        master_df.to_csv(args.output, index=False)
        print(f"\n✓ Saved updated results to: {args.output}")

    # Final stats
    print("\n" + "=" * 70)
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Total models in master: {len(master_df)}")
    print(f"Total corrections available: {len(corrections_df)}")
    print(f"Models updated: {len(changes)}")
    print(f"Total field changes: {sum(len(c['changes']) for c in changes)}")

    # Show unique values after merge
    print(
        f"\nUnique arch_family values: {sorted(master_df['arch_family'].dropna().unique())}"
    )
    print(
        f"Unique training_data values: {sorted(master_df['training_data'].dropna().unique())}"
    )
    print(
        f"Unique patch_size values: {sorted(master_df['patch_size'].dropna().unique())}"
    )


if __name__ == "__main__":
    main()
