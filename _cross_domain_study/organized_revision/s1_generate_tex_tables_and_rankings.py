"""
Generate a LaTeX results table from full_results_<dataset>.csv files,
and optionally write per-dataset ranking CSVs based on a designated
fairness metric.

Metrics are specified as "csv_name:Display Name" pairs, e.g.:
  "avg_acc:Avg Acc"  or just  "avg_acc"  (display name defaults to csv name)

Methods are renamed via --method-map and ordered via --method-order.
The baseline is always the first row; --method-order covers all other methods.
Methods not listed in --method-order are appended alphabetically at the end.

Ranking CSVs
------------
When --fairness-metrics and --ranking-out-dir are provided, the script writes
one file per dataset:

  <ranking_out_dir>/ranking_<dataset>.csv

with columns: method, dataset, mean, std, rank

  - method  : CSV method name (not the display name)
  - dataset : dataset identifier
  - mean    : mean of the fairness metric across seeds
  - std     : std  of the fairness metric across seeds
  - rank    : 1 = best (highest mean), includes the baseline

--fairness-metrics follows the same "csv_col" or "csv_col:Display Name"
syntax as --metrics, one entry per dataset.

Usage examples
--------------

# 2 datasets, 2 metrics each, full method renaming, ordering, and ranking
python generate_table.py \\
  --datasets accent_archive urbansounds \\
  --metrics "avg_acc:Avg Acc,wg_acc:WG Acc" "avg_acc:Avg Acc,wg_acc:WG Acc" \\
  --dataset-display-names "Accent Archive" "UrbanSounds" \\
  --baseline erm --baseline-display Vanilla \\
  --method-map bb=BB debian=Debian di=DI end=EnD groupdro=GroupDro \\
              jtt=JTT lff=LfF sd=SD flac=FLAC badd=BAdd mavias=MAVias \\
  --method-order bb debian di end groupdro jtt lff sd flac badd mavias \\
  --fairness-metrics "wg_acc:WG Acc" "wg_acc:WG Acc" \\
  --ranking-out-dir ./rankings \\
  --caption "Performance comparison on Speech Accent Archive and UrbanSounds8k." \\
  --label "tab:audio_results"

# 1 dataset, 1 metric, with ranking
python generate_table.py \\
  --datasets bias_in_bios \\
  --metrics "avg_acc:Avg Acc" \\
  --baseline erm --baseline-display Vanilla \\
  --method-order jtt debian lff \\
  --fairness-metrics "avg_acc" \\
  --ranking-out-dir ./rankings

# 2 datasets, different metric counts, different fairness metrics
python generate_table.py \\
  --datasets accent_archive bias_in_bios \\
  --metrics "avg_acc:Avg Acc,wg_acc:WG Acc" "macro_f1:Macro F1" \\
  --dataset-display-names "Accent Archive" "Bias in Bios" \\
  --baseline erm --baseline-display Vanilla \\
  --fairness-metrics "wg_acc:WG Acc" "macro_f1:Macro F1" \\
  --ranking-out-dir ./rankings


Table 3 
python s1_generate_tex_tables_and_rankings.py \
--datasets toxic bias_in_bios \
--metrics "Accuracy Official Set:Official Set Acc,Accuracy Generated Set:Generated Set Acc" "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" \
--dataset-display-names "Jigsaw Toxic Comments" "Bias in Bios" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE nsf=NSF george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd nsf george bias_ensemble bpa flac badd mavias \
--caption "Performance comparison on Jigsaw Toxic Comments and Bias in Bios datasets" \
--label "tab:text_results" \
--csv-dir ./full_results_v2 \
--output ./tex_tables/text_results.tex \
--fairness-metrics "Accuracy Generated Set:fm" "Test Worst Group Accuracy:fm" \
--ranking-out-dir ./rankings

Table 4 
python s1_generate_tex_tables_and_rankings.py \
--datasets speech_accent_archive urbansounds58 \
--metrics "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" \
--dataset-display-names "Accent Archive" "UrbanSounds" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE nsf=NSF george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd nsf george bias_ensemble bpa flac badd mavias \
--caption "Performance comparison on Speech Accent Archive and UrbanSounds8k." \
--label "tab:audio_results" \
--csv-dir ./full_results_v2 \
--output ./tex_tables/audio_results.tex \
--fairness-metrics "Test Worst Group Accuracy:fm" "Test Worst Group Accuracy:fm" \
--ranking-out-dir ./rankings

Table 5 
python s1_generate_tex_tables_and_rankings.py \
--datasets chexpert_nih ucf101 \
--metrics "Test Overall Accuracy:Avg Acc,Test Worst Group Accuracy:WG Acc" "test_accuracy:Acc" \
--dataset-display-names "CheXpert + NIH" "UCF101+SCUBA" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE nsf=NSF george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd nsf george bias_ensemble bpa flac badd mavias \
--caption "Performance on CheXpert + NIH and UCF101+SCUBA datasets." \
--label "tab:medical_video_results" \
--csv-dir ./full_results_v2 \
--output ./tex_tables/medical_video_results.tex \
--fairness-metrics "Test Worst Group Accuracy:fm" "test_accuracy:fm" \
--ranking-out-dir ./rankings

appendix table on original scuba
Table 5 
python s1_generate_tex_tables_and_rankings.py \
--datasets ucf101_org \
--metrics "test_accuracy:Acc" \
--dataset-display-name "UCF101+SCUBA" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd george bias_ensemble bpa flac badd mavias \
--caption "Performance on UCF101+SCUBA dataset." \
--label "tab:video_org_results" \
--csv-dir ./full_results_v2 \
--output ./tex_tables/video_org_results.tex \
--fairness-metrics "test_accuracy:fm" \
--ranking-out-dir ./rankings

new methods natural images 
python s1_generate_tex_tables_and_rankings.py \
--datasets celeba waterbirds urbancars \
--metrics "test_overall:Avg Acc,test_worst_group_accuracy:WG Acc" "test_overall:Avg Acc,test_worst_group_accuracy:WG Acc" "test_overall:Avg Acc,test_worst_group_accuracy:WG Acc" \
--dataset-display-names "CelebA" "Waterbirds" "UrbanCars" \
--baseline erm --baseline-display Vanilla \
--method-map bias_ensemble=BE nsf=NSF george=GEORGE bpa=BPA \
--method-order nsf george bias_ensemble bpa \
--caption "Performance on CheXpert + NIH and UCF101+SCUBA datasets." \
--label "tab:natural_images_results" \
--csv-dir ./full_results_v2 \
--output ./tex_tables/natural_images_results.tex \
--fairness-metrics "test_worst_group_accuracy:fm" "test_worst_group_accuracy:fm" "test_worst_group_accuracy:fm" \
--ranking-out-dir ./rankings

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
        default=".",
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
    # --- ranking ---
    p.add_argument(
        "--fairness-metrics",
        nargs="+",
        default=None,
        help=(
            "One fairness metric spec per dataset used for ranking. "
            "Same 'csv_col' or 'csv_col:Display Name' format as --metrics. "
            "Must be provided together with --ranking-out-dir."
        ),
    )
    p.add_argument(
        "--ranking-out-dir",
        default=None,
        help=(
            "Directory where ranking_<dataset>.csv files are written. "
            "Created automatically if it does not exist. "
            "Must be provided together with --fairness-metrics."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric spec parsing
# ---------------------------------------------------------------------------


def parse_metric_spec(raw: str) -> tuple[str, str]:
    """Parse a single 'csv_col' or 'csv_col:Display Name' token."""
    raw = raw.strip()
    if ":" in raw:
        csv_col, display = raw.split(":", 1)
        return csv_col.strip(), display.strip()
    return raw, raw


def parse_metric_specs(raw: str) -> list[tuple[str, str]]:
    """Parse a comma-separated string of metric specs."""
    return [parse_metric_spec(t) for t in raw.split(",") if t.strip()]


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
        header1.append(dname if n == 1 else rf"\multicolumn{{{n}}}{{c}}{{{dname}}}")
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
    header2 = [""]
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
# Ranking CSV output
# ---------------------------------------------------------------------------


def write_ranking_csv(
    out_path: str,
    dataset: str,
    agg_df: pd.DataFrame,
    fairness_col: str,
) -> None:
    """
    Build and write ranking_<dataset>.csv.

    Ranks all methods (including the baseline) by mean of fairness_col,
    highest = rank 1.
    """
    mean_col = f"{fairness_col}__mean"
    std_col = f"{fairness_col}__std"

    if mean_col not in agg_df.columns:
        raise ValueError(
            f"Fairness metric column '{fairness_col}' not found in aggregated data "
            f"for dataset '{dataset}'.\n"
            f"Available aggregated columns: {list(agg_df.columns)}"
        )

    ranking = agg_df[["Method", mean_col, std_col]].copy()
    ranking = ranking.rename(
        columns={mean_col: "mean", std_col: "std", "Method": "method"}
    )
    ranking["dataset"] = dataset
    ranking["rank"] = ranking["mean"].rank(method="min", ascending=False).astype(int)
    ranking = ranking.sort_values("rank").reset_index(drop=True)
    ranking = ranking[["method", "dataset", "mean", "std", "rank"]]

    ranking.to_csv(out_path, index=False, float_format="%.6f")
    print(f"  Ranking written to: {out_path}")


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

    # --- Basic validation ---
    if len(args.metrics) != len(args.datasets):
        sys.exit(
            f"ERROR: --metrics has {len(args.metrics)} entries but "
            f"--datasets has {len(args.datasets)}. They must match."
        )

    do_ranking = args.fairness_metrics is not None or args.ranking_out_dir is not None
    if do_ranking:
        if args.fairness_metrics is None:
            sys.exit("ERROR: --ranking-out-dir requires --fairness-metrics.")
        if args.ranking_out_dir is None:
            sys.exit("ERROR: --fairness-metrics requires --ranking-out-dir.")
        if len(args.fairness_metrics) != len(args.datasets):
            sys.exit(
                f"ERROR: --fairness-metrics has {len(args.fairness_metrics)} entries but "
                f"--datasets has {len(args.datasets)}. They must match."
            )

    metric_specs_per_dataset = [parse_metric_specs(m) for m in args.metrics]

    # Parse fairness metric specs (single spec per dataset, not comma-separated)
    fairness_specs: list[tuple[str, str]] = []
    if do_ranking:
        for raw in args.fairness_metrics:
            fairness_specs.append(parse_metric_spec(raw))

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

    # Load CSVs — collect all columns needed (table metrics + fairness metric)
    all_data = []
    for i, (dataset, specs) in enumerate(zip(args.datasets, metric_specs_per_dataset)):
        csv_path = os.path.join(args.csv_dir, f"full_results_{dataset}.csv")
        if not os.path.exists(csv_path):
            sys.exit(f"ERROR: File not found: {csv_path}")

        table_cols = [col for col, _ in specs]
        extra_cols = []
        if do_ranking:
            f_col = fairness_specs[i][0]
            if f_col not in table_cols:
                extra_cols.append(f_col)

        all_cols = table_cols + extra_cols
        print(f"Loading {csv_path}  (columns: {all_cols})")
        all_data.append(load_dataset(csv_path, all_cols))

    # Build and write LaTeX table
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

    # Write ranking CSVs
    if do_ranking:
        os.makedirs(args.ranking_out_dir, exist_ok=True)
        print(f"\nWriting ranking CSVs to: {args.ranking_out_dir}")
        for dataset, agg_df, (f_col, f_disp) in zip(
            args.datasets, all_data, fairness_specs
        ):
            out_path = os.path.join(args.ranking_out_dir, f"ranking_{dataset}.csv")
            print(f"  Fairness metric for '{dataset}': {f_col} ({f_disp})")
            write_ranking_csv(out_path, dataset, agg_df, f_col)


if __name__ == "__main__":
    main()
