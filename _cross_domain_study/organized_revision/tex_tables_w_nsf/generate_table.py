"""
Generate a LaTeX results table from full_results_<dataset>.csv files.

Metrics are specified as "csv_name:Display Name" pairs, e.g.:
  "avg_acc:Avg Acc"  or just  "avg_acc"  (display name defaults to csv name)

Methods are renamed via --method-map and ordered via --method-order.
The baseline is always the first row; --method-order covers all other methods.
Methods not listed in --method-order are appended alphabetically at the end.

Usage examples
--------------

# 2 datasets, 2 metrics each, full method renaming and ordering
python generate_table.py \
  --datasets accent_archive urbansounds \
  --metrics "avg_acc:Avg Acc,wg_acc:WG Acc" "avg_acc:Avg Acc,wg_acc:WG Acc" \
  --dataset-display-names "Accent Archive" "UrbanSounds" \
  --baseline erm --baseline-display Vanilla \
  --method-map bb=BB debian=Debian di=DI end=EnD groupdro=GroupDro \
              jtt=JTT lff=LfF sd=SD flac=FLAC badd=BAdd mavias=MAVias \
  --method-order bb debian di end groupdro jtt lff sd flac badd mavias \
  --caption "Performance comparison on Speech Accent Archive and UrbanSounds8k." \
  --label "tab:audio_results"
  

# 1 dataset, 1 metric
python generate_table.py \
  --datasets bias_in_bios \
  --metrics "avg_acc:Avg Acc" \
  --baseline erm --baseline-display Vanilla \
  --method-order jtt debian lff

# 2 datasets, different metric counts
python generate_table.py \
  --datasets accent_archive bias_in_bios \
  --metrics "avg_acc:Avg Acc,wg_acc:WG Acc" "macro_f1:Macro F1" \
  --dataset-display-names "Accent Archive" "Bias in Bios" \
  --baseline erm --baseline-display Vanilla


Table 3 
python generate_table.py \
--datasets toxic bias_in_bios \
--metrics "Accuracy Official Set:Official Set Acc,Accuracy Generated Set:Generated Set Acc" "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" \
--dataset-display-names "Jigsaw Toxic Comments" "Bias in Bios" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE nsf=NSF george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd nsf george bias_ensemble bpa flac badd mavias \
--caption "Performance comparison on Jigsaw Toxic Comments and Bias in Bios datasets" \
--label "tab:text_results" \
--csv-dir ../full_results_v2 \
--output text_results.tex

Table 4 
python generate_table.py \
--datasets speech_accent_archive urbansounds58 \
--metrics "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" \
--dataset-display-names "Accent Archive" "UrbanSounds" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE nsf=NSF george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd nsf george bias_ensemble bpa flac badd mavias \
--caption "Performance comparison on Speech Accent Archive and UrbanSounds8k." \
--label "tab:audio_results" \
--csv-dir ../full_results_v2 \
--output audio_results.tex

Table 5 
python generate_table.py \
--datasets chexpert_nih ucf101 \
--metrics "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" "test_accuracy:Acc" \
--dataset-display-names "CheXpert + NIH" "UCF101+SCUBA" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE nsf=NSF george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd nsf george bias_ensemble bpa flac badd mavias \
--caption "erformance on CheXpert + NIH and UCF101+SCUBA datasets." \
--label "tab:medical_video_results" \
--csv-dir ../full_results_v2 \
--output medical_video_results.tex

"""

import argparse
import os
import sys
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate LaTeX table from CSV results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset identifiers used in filenames (full_results_<dataset>.csv).",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help=(
            "Comma-separated metric specs per dataset, one entry per dataset. "
            "Each spec is either 'csv_col' or 'csv_col:Display Name'. "
            'Example: "avg_acc:Avg Acc,wg_acc:WG Acc" "macro_f1:Macro F1"'
        ),
    )
    p.add_argument(
        "--dataset-display-names",
        nargs="+",
        default=None,
        help="Human-readable dataset names for the table header. Defaults to dataset ids.",
    )
    p.add_argument(
        "--baseline",
        default="erm",
        help="Method name in the CSV used as the baseline row (default: erm).",
    )
    p.add_argument(
        "--baseline-display",
        default="Vanilla",
        help="Display name for the baseline row (default: Vanilla).",
    )
    p.add_argument(
        "--method-map",
        nargs="+",
        default=None,
        help=(
            "Rename methods for display: csv_name=DisplayName pairs. "
            "Example: jtt=JTT lff=LfF groupdro=GroupDro"
        ),
    )
    p.add_argument(
        "--method-order",
        nargs="+",
        default=None,
        help=(
            "Ordered list of method CSV names (excluding baseline). "
            "Methods not listed are appended alphabetically. "
            "Example: bb debian di jtt lff"
        ),
    )
    p.add_argument(
        "--csv-dir",
        default=None,
        help="Directory containing the CSV files (default: current dir).",
    )
    p.add_argument(
        "--output",
        default="results_table.tex",
        help="Output .tex file (default: results_table.tex).",
    )
    p.add_argument(
        "--caption", default="Performance comparison.", help="Table caption."
    )
    p.add_argument("--label", default="tab:results", help="Table label.")
    p.add_argument(
        "--no-delta",
        action="store_true",
        help="Suppress the coloured (+/-) delta annotations.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric spec parsing
# ---------------------------------------------------------------------------


def parse_metric_specs(raw: str) -> list[tuple[str, str]]:
    """
    Parse a comma-separated metric spec string into (csv_col, display_name) pairs.
    Each token is either "csv_col" or "csv_col:Display Name".
    """
    specs = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            csv_col, display = token.split(":", 1)
            specs.append((csv_col.strip(), display.strip()))
        else:
            specs.append((token, token))
    return specs


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------


def load_dataset(csv_path: str, csv_cols: list[str]) -> pd.DataFrame:
    """Load one CSV and return per-method mean±std for each metric column."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    missing = [c for c in csv_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metric column(s) {missing} not found in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )

    agg = df.groupby("Method")[csv_cols].agg(["mean", "std"])
    # Flatten: (csv_col, stat) -> "csv_col__stat"
    agg.columns = ["__".join(c) for c in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def fmt_cell(
    mean: float, std: float, baseline_mean: float | None, show_delta: bool
) -> str:
    # Round to displayed precision
    mean_r = round(mean, 1)
    std_r = round(std, 1)

    cell = rf"{mean_r:.1f}\std{{{std_r:.1f}}}"

    if show_delta and baseline_mean is not None:
        baseline_r = round(baseline_mean, 1)

        delta = mean_r - baseline_r
        sign = "+" if delta >= 0 else ""
        colour = "blue" if delta >= 0 else "red"

        cell += rf" {{\color{{{colour}}}({sign}{delta:.1f})}}"

    return cell


def get_stats(df: pd.DataFrame, method: str, csv_col: str):
    row = df[df["Method"] == method]
    if row.empty:
        return None, None
    return row[f"{csv_col}__mean"].values[0], row[f"{csv_col}__std"].values[0]


def build_table(
    dataset_display_names: list[str],
    metric_specs_per_dataset: list[list[tuple[str, str]]],
    all_data: list[pd.DataFrame],
    baseline_method: str,
    baseline_display: str,
    method_map: dict[str, str],
    method_order: list[str] | None,
    show_delta: bool,
    caption: str,
    label: str,
) -> str:

    # Collect all non-baseline methods across all datasets
    all_methods: set[str] = set()
    for df in all_data:
        all_methods.update(df["Method"].tolist())
    all_methods.discard(baseline_method)

    def display_method(m: str) -> str:
        return method_map.get(m, m)

    # Build ordered row list
    if method_order:
        seen: set[str] = set()
        rows: list[str] = []
        for m in method_order:
            if m in all_methods and m not in seen:
                rows.append(m)
                seen.add(m)
        for m in sorted(all_methods):
            if m not in seen:
                rows.append(m)
    else:
        rows = sorted(all_methods)

    # Column spec
    col_spec = "l" + "".join("l" * len(specs) for specs in metric_specs_per_dataset)

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: dataset spans
    header1 = [r"\multirow{2}{*}{Method}"]
    for dname, specs in zip(dataset_display_names, metric_specs_per_dataset):
        n = len(specs)
        if n == 1:
            header1.append(dname)
        else:
            header1.append(rf"\multicolumn{{{n}}}{{c}}{{{dname}}}")
    lines.append(" & ".join(header1) + r" \\")

    # Cmidrules
    cmidrules = []
    col = 2
    for specs in metric_specs_per_dataset:
        n = len(specs)
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    lines.append(" ".join(cmidrules))

    # Header row 2: metric display names
    header2 = [""]  # blank under "Method"
    for specs in metric_specs_per_dataset:
        header2.extend(disp for _, disp in specs)
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    # Baseline row
    baseline_cells = [baseline_display]
    for df, specs in zip(all_data, metric_specs_per_dataset):
        for csv_col, _ in specs:
            mean, std = get_stats(df, baseline_method, csv_col)
            baseline_cells.append(
                "--" if mean is None else fmt_cell(mean, std, None, False)
            )
    lines.append(" & ".join(baseline_cells) + r" \\")

    # Method rows
    for method in rows:
        cells = [display_method(method)]
        for df, specs in zip(all_data, metric_specs_per_dataset):
            for csv_col, _ in specs:
                mean, std = get_stats(df, method, csv_col)
                if mean is None:
                    cells.append("--")
                else:
                    b_mean, _ = get_stats(df, baseline_method, csv_col)
                    cells.append(fmt_cell(mean, std, b_mean, show_delta))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preamble hint
# ---------------------------------------------------------------------------

PREAMBLE_HINT = """\
% Add to your LaTeX preamble:
% \\usepackage{booktabs}
% \\usepackage{multirow}
% \\usepackage{xcolor}
% \\newcommand{\\std}[1]{{\\scriptsize$\\pm$#1}}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if len(args.metrics) != len(args.datasets):
        sys.exit(
            f"ERROR: --metrics has {len(args.metrics)} entries but "
            f"--datasets has {len(args.datasets)}. They must match."
        )

    metric_specs_per_dataset = [parse_metric_specs(m) for m in args.metrics]

    dataset_display_names = args.dataset_display_names or args.datasets
    if len(dataset_display_names) != len(args.datasets):
        sys.exit("ERROR: --dataset-display-names count must match --datasets count.")

    # Parse method map
    method_map: dict[str, str] = {}
    if args.method_map:
        for entry in args.method_map:
            if "=" not in entry:
                sys.exit(
                    f"ERROR: --method-map entries must be key=value, got: {entry!r}"
                )
            k, v = entry.split("=", 1)
            method_map[k.strip()] = v.strip()

    # Load CSVs
    all_data = []
    for dataset, specs in zip(args.datasets, metric_specs_per_dataset):
        csv_path = os.path.join(args.csv_dir, f"full_results_{dataset}.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"ERROR: File not found: {csv_path}")
        csv_cols = [col for col, _ in specs]
        print(f"Loading {csv_path}  (columns: {csv_cols})")
        all_data.append(load_dataset(csv_path, csv_cols))

    latex = build_table(
        dataset_display_names=dataset_display_names,
        metric_specs_per_dataset=metric_specs_per_dataset,
        all_data=all_data,
        baseline_method=args.baseline,
        baseline_display=args.baseline_display,
        method_map=method_map,
        method_order=args.method_order,
        show_delta=not args.no_delta,
        caption=args.caption,
        label=args.label,
    )

    with open(args.output, "w") as f:
        f.write(PREAMBLE_HINT + "\n" + latex + "\n")

    print(f"\nTable written to: {args.output}")
    print("\n--- Preview ---\n")
    print(latex)


if __name__ == "__main__":
    main()
