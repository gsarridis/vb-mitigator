"""Harvest the imagenet9m sweep results into a tidy CSV.

Walks output/<PROJECT>/<scenario>_<model>_c<corr>_s<seed>/<method>/logs<seed>.csv,
takes the best epoch (argmax of the selection metric `test_overall`, matching how the
framework picks the "best" checkpoint), and records the worst-group + overall accuracy.

Outputs:
  <out_dir>/results.csv   -- one row per (scenario, model, correlation, seed, method)
  <out_dir>/summary.csv   -- mean/std over seeds per (scenario, model, correlation, method)
"""

import argparse
import os
import re

import numpy as np
import pandas as pd

TAG_RE = re.compile(r"^(jpeg|resize|multi)_(resnet18|resnet50)_c([0-9.]+)_s([0-9]+)$")
METHODS = ["erm", "flac", "badd", "maviasb", "bpa"]
WG = "test_worst_group_accuracy"
OV = "test_overall"


def harvest(project_dir):
    rows = []
    for tag in sorted(os.listdir(project_dir)):
        m = TAG_RE.match(tag)
        if not m:
            continue
        scenario, model, corr, seed = m.group(1), m.group(2), float(m.group(3)), int(m.group(4))
        for method in METHODS:
            csv = os.path.join(project_dir, tag, method, f"logs{seed}.csv")
            rec = dict(scenario=scenario, model=model, correlation=corr, seed=seed,
                       method=method, status="missing", epochs=0,
                       wg=np.nan, overall=np.nan, wg_max=np.nan, wg_last=np.nan)
            if os.path.isfile(csv):
                try:
                    df = pd.read_csv(csv)
                except Exception:
                    df = None
                if df is not None and len(df) and WG in df and OV in df:
                    df = df.dropna(subset=[WG])
                    if len(df):
                        best = df.loc[df[WG].idxmax()]  # select the epoch with best worst-group acc
                        rec.update(status="ok", epochs=int(df["epoch"].max()),
                                   wg=float(best[WG]), overall=float(best[OV]),
                                   wg_max=float(df[WG].max()), wg_last=float(df.iloc[-1][WG]))
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(df):
    ok = df[df.status == "ok"]
    g = ok.groupby(["scenario", "model", "correlation", "method"])
    s = g["wg"].agg(["mean", "std", "count"]).reset_index()
    s = s.rename(columns={"mean": "wg_mean", "std": "wg_std", "count": "n_seeds"})
    s["overall_mean"] = g["overall"].mean().values
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", default="output/imagenet9m_baselines")
    ap.add_argument("--out-dir", default="output/imagenet9m_report")
    args = ap.parse_args()

    df = harvest(args.project_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    summarize(df).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)

    # coverage report
    total = len(df)
    ok = (df.status == "ok").sum()
    print(f"runs found: {total} | ok: {ok} | missing: {total - ok}")
    print("\nper-method ok counts:")
    print(df[df.status == "ok"].method.value_counts().to_string())
    print("\nincomplete (epochs < 30) or missing:")
    bad = df[(df.status != "ok") | (df.epochs < 30)]
    if len(bad):
        print(bad[["scenario", "model", "correlation", "seed", "method", "status", "epochs"]].to_string(index=False))
    else:
        print("  none")


if __name__ == "__main__":
    main()
