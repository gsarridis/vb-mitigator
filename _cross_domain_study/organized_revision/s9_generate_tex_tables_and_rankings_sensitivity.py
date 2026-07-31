"""
Generate a LaTeX results table from full_results_<dataset>.csv files,
and optionally write per-dataset ranking CSVs based on a designated
fairness metric, and sensitivity line plots (one per table metric).

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

Sensitivity Plots
-----------------
When --sensitivity-ratios is provided, the script produces one PDF per metric
(using the display name to derive the filename) showing each method as a
connected scatter line across the sensitivity ratios.

  --sensitivity-ratios 70 80 90
  --plots-out-dir      ./plots          (default: current dir)

The number of ratios must equal the number of datasets.

Usage example
-------------

python s9_generate_tex_tables_and_rankings_sensitivity.py \
--datasets chexpert_nih_s07 chexpert_nih_s08 chexpert_nih_s09 \
--metrics "test_overall:Avg Acc,test_worst_group_accuracy:WG Acc" "test_overall:Avg Acc,test_worst_group_accuracy:WG Acc" "test_overall:Avg Acc,test_worst_group_accuracy:WG Acc" \
--dataset-display-names "CheXpert + NIH 70%" "CheXpert + NIH 80%" "CheXpert + NIH 90%" \
--baseline erm --baseline-display Vanilla \
--method-map bb=BB debian=Debian di=DI end=EnD jtt=JTT lff=LfF groupdro=GroupDro \
            sd=SD flac=FLAC badd=BAdd maviasb=MAVIAS bias_ensemble=BE \
            nsf=NSF george=GEORGE bpa=BPA \
--method-order bb debian di end groupdro jtt lff sd nsf george bias_ensemble bpa flac badd maviasb \
--caption "Sensitivity on CheXpert + NIH" \
--label "tab:med_sens_results" \
--csv-dir ./full_results_sensitivity \
--output ./tex_tables/chexpert_sensitivity_results.tex \
--ranking-out-dir ./rankings_sensitivity \
--fairness-metrics \
  "test_overall:fm" \
  "test_overall:fm" \
  "test_overall:fm" \
--sensitivity-ratios 70 80 90 \
--plots-out-dir ./plots_sensitivity
"""

import argparse
import os
import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


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
    # --- sensitivity plots ---
    p.add_argument(
        "--sensitivity-ratios",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Numeric sensitivity ratios corresponding to each dataset, "
            "used as the x-axis of the sensitivity plots. "
            "Must match the number of --datasets. Example: 70 80 90"
        ),
    )
    p.add_argument(
        "--plots-out-dir",
        default=".",
        help="Directory where sensitivity plot PDFs are saved (default: current dir).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Metric spec parsing
# ---------------------------------------------------------------------------


def parse_metric_spec(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    if ":" in raw:
        csv_col, display = raw.split(":", 1)
        return csv_col.strip(), display.strip()
    return raw, raw


def parse_metric_specs(raw: str) -> list[tuple[str, str]]:
    return [parse_metric_spec(t) for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------


def load_dataset(csv_path: str, csv_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    missing = [c for c in csv_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Metric column(s) {missing} not found in {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )
    agg = df.groupby("Method")[csv_cols].agg(["mean", "std"])
    agg.columns = ["__".join(c) for c in agg.columns]
    return agg.reset_index()


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------


def fmt_cell(
    mean: float, std: float, baseline_mean: float | None, show_delta: bool
) -> str:
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


def avg_consecutive_delta(
    method: str, csv_col: str, all_data: list[pd.DataFrame]
) -> float | None:
    """
    Compute the average of consecutive differences in `csv_col` means across datasets.
    Returns None if fewer than 2 data points are available for this method.
    E.g. for ratios [70, 80, 90]: avg of [(val_80 - val_70), (val_90 - val_80)]
    """
    means = []
    for df in all_data:
        mean, _ = get_stats(df, method, csv_col)
        if mean is not None:
            means.append(round(mean, 1))
    if len(means) < 2:
        return None
    deltas = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    return sum(deltas) / len(deltas)


def fmt_avg_delta(delta: float, baseline_delta: float | None) -> str:
    """Format an avg-delta cell: signed value only, no colours or annotations."""
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"


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
    sensitivity_ratios: list[float] | None = None,
) -> str:
    all_methods: set[str] = set()
    for df in all_data:
        all_methods.update(df["Method"].tolist())
    all_methods.discard(baseline_method)

    def display_method(m: str) -> str:
        return method_map.get(m, m)

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

    # Determine whether to add avg-delta columns.
    # We add one "Avg Δ" column per unique metric (csv_col), but only when
    # sensitivity_ratios are provided and there are at least 2 datasets.
    add_avg_delta = sensitivity_ratios is not None and len(all_data) >= 2

    # Collect unique metrics in order of first appearance across datasets
    unique_metrics: list[tuple[str, str]] = []  # (csv_col, display)
    seen_cols: set[str] = set()
    for specs in metric_specs_per_dataset:
        for csv_col, disp in specs:
            if csv_col not in seen_cols:
                unique_metrics.append((csv_col, disp))
                seen_cols.add(csv_col)

    n_avg_delta_cols = len(unique_metrics) if add_avg_delta else 0

    col_spec = (
        "l"
        + "".join("l" * len(specs) for specs in metric_specs_per_dataset)
        + "l" * n_avg_delta_cols
    )

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # --- Header row 1 ---
    header1 = [r"\multirow{2}{*}{Method}"]
    for dname, specs in zip(dataset_display_names, metric_specs_per_dataset):
        n = len(specs)
        header1.append(dname if n == 1 else rf"\multicolumn{{{n}}}{{c}}{{{dname}}}")
    if add_avg_delta:
        n_d = len(unique_metrics)
        label_text = "Avg $\Delta$"
        header1.append(
            label_text if n_d == 1 else rf"\multicolumn{{{n_d}}}{{c}}{{{label_text}}}"
        )
    lines.append(" & ".join(header1) + r" \\")

    # --- Cmidrules ---
    cmidrules = []
    col = 2
    for specs in metric_specs_per_dataset:
        n = len(specs)
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col + n - 1}}}")
        col += n
    if add_avg_delta and n_avg_delta_cols > 0:
        cmidrules.append(rf"\cmidrule(lr){{{col}-{col + n_avg_delta_cols - 1}}}")
    lines.append(" ".join(cmidrules))

    # --- Header row 2 ---
    header2 = [""]
    for specs in metric_specs_per_dataset:
        header2.extend(disp for _, disp in specs)
    if add_avg_delta:
        header2.extend(disp for _, disp in unique_metrics)
    lines.append(" & ".join(header2) + r" \\")
    lines.append(r"\midrule")

    # Pre-compute baseline avg deltas for each unique metric
    baseline_avg_deltas: dict[str, float | None] = {}
    if add_avg_delta:
        for csv_col, _ in unique_metrics:
            baseline_avg_deltas[csv_col] = avg_consecutive_delta(
                baseline_method, csv_col, all_data
            )

    # --- Baseline row ---
    baseline_cells = [baseline_display]
    for df, specs in zip(all_data, metric_specs_per_dataset):
        for csv_col, _ in specs:
            mean, std = get_stats(df, baseline_method, csv_col)
            baseline_cells.append(
                "--" if mean is None else fmt_cell(mean, std, None, False)
            )
    if add_avg_delta:
        for csv_col, _ in unique_metrics:
            d = baseline_avg_deltas[csv_col]
            baseline_cells.append("--" if d is None else fmt_avg_delta(d, None))
    lines.append(" & ".join(baseline_cells) + r" \\")

    # --- Method rows ---
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
        if add_avg_delta:
            for csv_col, _ in unique_metrics:
                d = avg_consecutive_delta(method, csv_col, all_data)
                b_d = baseline_avg_deltas[csv_col]
                cells.append("--" if d is None else fmt_avg_delta(d, b_d))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ranking CSV output
# ---------------------------------------------------------------------------


def write_ranking_csv(
    out_path: str, dataset: str, agg_df: pd.DataFrame, fairness_col: str
) -> None:
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
# Sensitivity plots
# ---------------------------------------------------------------------------

# Distinct markers to cycle through
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "p", "<", ">", "H", "8"]


def _safe_filename(display_name: str) -> str:
    """Convert a metric display name to a safe filename stem."""
    return display_name.lower().replace(" ", "_").replace("/", "_")


def plot_sensitivity(
    all_data: list[pd.DataFrame],
    metric_specs_per_dataset: list[list[tuple[str, str]]],
    sensitivity_ratios: list[float],
    baseline_method: str,
    baseline_display: str,
    method_map: dict[str, str],
    method_order: list[str] | None,
    plots_out_dir: str,
) -> None:
    """
    For each unique metric (identified by csv_col across datasets), produce one
    PDF with connected scatter lines — one line per method — over the x-axis
    defined by sensitivity_ratios.
    """
    os.makedirs(plots_out_dir, exist_ok=True)

    # Build the set of unique metrics, preserving the display name from the
    # first dataset that mentions each csv_col.
    # metric_index[csv_col] = display_name
    metric_index: dict[str, str] = {}
    for specs in metric_specs_per_dataset:
        for csv_col, display in specs:
            if csv_col not in metric_index:
                metric_index[csv_col] = display

    # Collect all methods in display order (baseline first, then the rest)
    all_methods_csv: set[str] = set()
    for df in all_data:
        all_methods_csv.update(df["Method"].tolist())

    def display_method(m: str) -> str:
        if m == baseline_method:
            return baseline_display
        return method_map.get(m, m)

    non_baseline = all_methods_csv - {baseline_method}
    if method_order:
        seen: set[str] = set()
        ordered: list[str] = []
        for m in method_order:
            if m in non_baseline and m not in seen:
                ordered.append(m)
                seen.add(m)
        for m in sorted(non_baseline):
            if m not in seen:
                ordered.append(m)
    else:
        ordered = sorted(non_baseline)

    all_methods_ordered = [baseline_method] + ordered  # baseline always first

    n_methods = len(all_methods_ordered)

    # Use a qualitative colormap with enough colours
    cmap = cm.get_cmap("tab20", max(n_methods, 1))
    colour_for = {m: cmap(i) for i, m in enumerate(all_methods_ordered)}

    x = np.array(sensitivity_ratios)

    for csv_col, metric_display in metric_index.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for idx, method in enumerate(all_methods_ordered):
            y_vals = []
            y_err = []
            x_valid = []

            for ratio, df, specs in zip(
                sensitivity_ratios, all_data, metric_specs_per_dataset
            ):
                # Only plot this point if the metric exists for this dataset
                if not any(c == csv_col for c, _ in specs):
                    continue
                mean, std = get_stats(df, method, csv_col)
                if mean is None:
                    continue
                x_valid.append(ratio)
                y_vals.append(mean)
                y_err.append(std if std is not None else 0.0)

            if not x_valid:
                continue

            x_arr = np.array(x_valid)
            y_arr = np.array(y_vals)
            e_arr = np.array(y_err)

            marker = _MARKERS[idx % len(_MARKERS)]
            colour = colour_for[method]
            label = display_method(method)

            ax.plot(
                x_arr,
                y_arr,
                marker=marker,
                color=colour,
                linewidth=1.6,
                markersize=6,
                label=label,
            )
            ax.fill_between(
                x_arr, y_arr - e_arr, y_arr + e_arr, color=colour, alpha=0.10
            )

        ax.set_xlabel("Sensitivity Ratio (%)", fontsize=12)
        ax.set_ylabel(metric_display, fontsize=12)
        ax.set_title(f"Sensitivity Analysis — {metric_display}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r)}%" for r in x])
        ax.grid(True, linestyle="--", alpha=0.4)

        # Place legend outside the plot on the right
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            fontsize=9,
            frameon=True,
        )

        fig.tight_layout()

        fname = _safe_filename(metric_display)
        out_path = os.path.join(plots_out_dir, f"sensitivity_{fname}.pdf")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Plot saved to: {out_path}")


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

    do_plots = args.sensitivity_ratios is not None
    if do_plots and len(args.sensitivity_ratios) != len(args.datasets):
        sys.exit(
            f"ERROR: --sensitivity-ratios has {len(args.sensitivity_ratios)} entries but "
            f"--datasets has {len(args.datasets)}. They must match."
        )

    metric_specs_per_dataset = [parse_metric_specs(m) for m in args.metrics]

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

    # Load CSVs
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
        sensitivity_ratios=args.sensitivity_ratios,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
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

    # Sensitivity plots
    if do_plots:
        print(f"\nGenerating sensitivity plots in: {args.plots_out_dir}")
        plot_sensitivity(
            all_data=all_data,
            metric_specs_per_dataset=metric_specs_per_dataset,
            sensitivity_ratios=args.sensitivity_ratios,
            baseline_method=args.baseline,
            baseline_display=args.baseline_display,
            method_map=method_map,
            method_order=args.method_order,
            plots_out_dir=args.plots_out_dir,
        )


if __name__ == "__main__":
    main()
