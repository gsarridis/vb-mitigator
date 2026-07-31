"""
Merge per-dataset ranking CSVs into a single performances.csv.

Reads:  <ranking_dir>/ranking_<dataset>.csv  (one per dataset)
Writes: performances.csv  (all datasets combined, with display names)

The output has columns: method, dataset, mean, std
  - method  : display name (from METHOD_MAP, or original if not mapped)
  - dataset : display name (from DATASET_MAP, or original if not mapped)
  - mean    : mean of the fairness metric
  - std     : std  of the fairness metric

Edit METHOD_MAP and DATASET_MAP below to set your display names.

Usage
-----
  python s3_build_performances.py --datasets bias_in_bios toxic urbansounds58 speech_accent_archive chexpert_nih ucf101 --ranking-dir ./rankings --output performances.csv
"""

import argparse
import os
import sys
import pandas as pd


# ---------------------------------------------------------------------------
# DISPLAY NAME MAPPINGS  — edit these to match your setup
# ---------------------------------------------------------------------------

# Maps CSV method name  →  display name shown in performances.csv
METHOD_MAP: dict[str, str] = {
    "erm": "Vanilla",
    "bb": "BB",
    "debian": "Debian",
    "di": "DI",
    "end": "EnD",
    "groupdro": "GroupDro",
    "jtt": "JTT",
    "lff": "LfF",
    "sd": "SD",
    "flac": "FLAC",
    "badd": "BAdd",
    "mavias": "MAVias",
    "maviasb": "MAVias",
    "bias_ensemble": "BE",
    "nsf": "NSF",
    "bpa": "BPA",
    "george": "GEORGE",
    # add more here ...
}

# Maps CSV dataset name  →  display name shown in performances.csv
DATASET_MAP: dict[str, str] = {
    "bias_in_bios": "Bias in Bios",
    "urbansounds58": "UrbanSounds8k",
    "speech_accent_archive": "Speech Accent Archive",
    "toxic": "Jigsaw Toxic Comments",
    "chexpert_nih": "CheXpert+NIH",
    "ucf101": "UCF101+SCUBA",
    # add more here ...
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge ranking CSVs into a single performances.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "Dataset identifiers to include (must match ranking_<dataset>.csv filenames). "
            "Example: bias_in_bios jigsaw urbansounds"
        ),
    )
    p.add_argument(
        "--ranking-dir",
        default=".",
        help="Directory containing ranking_<dataset>.csv files (default: current dir).",
    )
    p.add_argument(
        "--output",
        default="performances.csv",
        help="Path for the output performances.csv (default: performances.csv).",
    )
    p.add_argument(
        "--keep-rank-col",
        action="store_true",
        help="Keep the 'rank' column in the output (dropped by default).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def load_ranking(ranking_dir: str, dataset: str) -> pd.DataFrame:
    path = os.path.join(ranking_dir, f"ranking_{dataset}.csv")
    if not os.path.exists(path):
        sys.exit(f"ERROR: File not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = {"method", "dataset", "mean", "std"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {missing}")
    return df


def apply_maps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["method"] = df["method"].map(lambda m: METHOD_MAP.get(m, m))
    df["dataset"] = df["dataset"].map(lambda d: DATASET_MAP.get(d, d))
    return df


def build_performances(datasets: list[str], ranking_dir: str) -> pd.DataFrame:
    frames = []
    for dataset in datasets:
        print(f"Loading ranking_{dataset}.csv ...")
        df = load_ranking(ranking_dir, dataset)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    combined = apply_maps(combined)
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    performances = build_performances(args.datasets, args.ranking_dir)

    # Columns to keep
    cols = ["method", "dataset", "mean", "std"]
    if args.keep_rank_col and "rank" in performances.columns:
        cols.append("rank")
    performances = performances[cols]

    performances.to_csv(args.output, index=False, float_format="%.1f")
    print(f"\nperformances.csv written to: {args.output}")
    print(f"  Rows : {len(performances)}")
    print(f"  Methods  : {sorted(performances['method'].unique())}")
    print(f"  Datasets : {sorted(performances['dataset'].unique())}")


if __name__ == "__main__":
    main()
