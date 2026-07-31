"""
Explore the RAISE metadata CSV to pick a 2-class / 2-camera subset for a
camera-acquisition spurious-correlation benchmark.

The camera (`Device`) is a *fixed* acquisition attribute (you cannot change which
camera took a photo), so to get a target correlation between the semantic class
and the camera you must choose classes + cameras whose natural joint
distribution supports it -- while keeping as many samples as possible. The
processing bias (jpeg/png/...) is applied later and is fully controllable.

What it reports:
  1. Camera (Device) distribution.
  2. Atomic semantic-tag frequencies (parsed from `Keywords`).
  3. Tag x camera contingency with P(camera | tag)  -> shows the natural skew.
  4. A ranked table of (tagA vs tagB) class pairs over a camera pair, by the
     MAX balanced dataset size achievable at a target correlation rho (train
     biased at rho, val/test balanced), i.e. how many samples you can keep.

Drill into a specific choice with --class0/--class1 (and --rho/--cameras), or
explore live with --interactive.

Usage:
  python tools/raise_stats.py                                   # overview + ranked pairs
  python tools/raise_stats.py --class0 indoor --class1 outdoor --rho 0.9
  python tools/raise_stats.py --interactive
"""

import argparse
import itertools
import math
import os
from collections import Counter

import numpy as np
import pandas as pd

CAMERA_COL = "Device"
TAG_COL = "Keywords"
ATOMIC_TAGS = ["outdoor", "indoor", "landscape", "buildings", "nature", "people", "objects"]


# --------------------------------------------------------------------------- #
def parse_tags(value):
    if not isinstance(value, str):
        return set()
    return {t.strip().lower() for t in value.split(";") if t.strip()}


def load(csv):
    df = pd.read_csv(csv)
    df["_tags"] = df[TAG_COL].apply(parse_tags)
    return df


def top_cameras(df, n=2):
    return df[CAMERA_COL].value_counts().head(n).index.tolist()


# --------------------------------------------------------------------------- #
# Feasibility: max per-class samples at correlation rho with balanced val/test.
# --------------------------------------------------------------------------- #
def _frac_aligned_conflict(rho, s_train):
    """Per-class fraction of samples drawn from the aligned vs conflicting camera.

    Train is biased at rho; val+test are balanced (50/50). Returns (A, C) with
    A+C = 1, where A is the share that must come from the aligned camera.
    """
    s_eval = 1.0 - s_train
    a = rho * s_train + 0.5 * s_eval
    c = (1.0 - rho) * s_train + 0.5 * s_eval
    return a, c


def max_samples(a, b, c, d, rho, s_train):
    """2x2 cell counts:  [[a,b],[c,d]] = [[c0&cam0, c0&cam1],[c1&cam0, c1&cam1]].

    Returns (orientation, per_class_T, total, binding) for the orientation that
    maximizes the kept set. ``orientation`` says which camera each class aligns to.
    """
    A, C = _frac_aligned_conflict(rho, s_train)

    def T_for(al0, cf0, al1, cf1):
        # al*: count available in each class's ALIGNED camera; cf*: in CONFLICTING.
        return min(al0 / A, cf0 / C, al1 / A, cf1 / C)

    # orientation 1: class0 -> cam0 (aligned a, conflict b); class1 -> cam1 (aligned d, conflict c)
    t1 = T_for(a, b, d, c)
    # orientation 2: class0 -> cam1 (aligned b, conflict a); class1 -> cam0 (aligned c, conflict d)
    t2 = T_for(b, a, c, d)

    if t1 >= t2:
        orient, T, binding = "class0->cam0, class1->cam1", t1, [a / A, b / C, d / A, c / C]
    else:
        orient, T, binding = "class0->cam1, class1->cam0", t2, [b / A, a / C, c / A, d / C]
    T = int(math.floor(T))
    return orient, T, 2 * T, binding


def class_masks(df, tags0, tags1):
    """Exclusive binary split: class0 has any tag in tags0 and none in tags1 (and vice versa)."""
    t0, t1 = set(tags0), set(tags1)
    has0 = df["_tags"].apply(lambda s: bool(s & t0))
    has1 = df["_tags"].apply(lambda s: bool(s & t1))
    return (has0 & ~has1), (has1 & ~has0)


def contingency(df, mask0, mask1, cam0, cam1):
    cam = df[CAMERA_COL]
    a = int((mask0 & (cam == cam0)).sum())
    b = int((mask0 & (cam == cam1)).sum())
    c = int((mask1 & (cam == cam0)).sum())
    d = int((mask1 & (cam == cam1)).sum())
    return a, b, c, d


# --------------------------------------------------------------------------- #
# Reports.
# --------------------------------------------------------------------------- #
def overview(df):
    print(f"rows: {len(df)}\n")
    print("== cameras (Device) ==")
    print(df[CAMERA_COL].value_counts().to_string())
    print()

    print("== atomic semantic tags (images containing each) ==")
    counts = Counter()
    for s in df["_tags"]:
        counts.update(s)
    for tag, n in counts.most_common():
        print(f"  {tag:<10} {n:5d}  ({n / len(df):.1%})")
    print()

    cams = top_cameras(df, 2)
    print(f"== tag x camera  (P(camera|tag) over {cams[0]} vs {cams[1]}) ==")
    sub = df[df[CAMERA_COL].isin(cams)]
    print(f"  {'tag':<10} {'n':>6} {cams[0]:>14} {cams[1]:>14}")
    for tag in ATOMIC_TAGS:
        m = sub["_tags"].apply(lambda s: tag in s)
        n = int(m.sum())
        if n == 0:
            continue
        n0 = int((m & (sub[CAMERA_COL] == cams[0])).sum())
        print(f"  {tag:<10} {n:6d} {n0 / n:>13.1%} {1 - n0 / n:>13.1%}")
    print()


def rank_pairs(df, cameras, rho, s_train, top):
    cam0, cam1 = cameras
    print(f"== ranked class pairs | cameras: {cam0} vs {cam1} | rho={rho} | split train={s_train:.0%} ==")
    print(f"   (max = largest balanced set keepable at rho; val/test balanced)\n")
    rows = []
    for ta, tb in itertools.combinations(ATOMIC_TAGS, 2):
        m0, m1 = class_masks(df, [ta], [tb])
        a, b, c, d = contingency(df, m0, m1, cam0, cam1)
        if min(a + b, c + d) == 0:
            continue
        orient, T, total, _ = max_samples(a, b, c, d, rho, s_train)
        # natural skew: P(aligned camera | class) under the chosen orientation
        if orient.startswith("class0->cam0"):
            skew0, skew1 = a / (a + b), d / (c + d)
        else:
            skew0, skew1 = b / (a + b), c / (c + d)
        rows.append((total, T, ta, tb, (a, b, c, d), orient, skew0, skew1))

    rows.sort(reverse=True)
    print(f"  {'class0':<10} {'class1':<10} {'cells[a,b,c,d]':<22} {'orient':<26} {'skew0/1':>11} {'max_total':>9}")
    for total, T, ta, tb, cells, orient, s0, s1 in rows[:top]:
        print(f"  {ta:<10} {tb:<10} {str(list(cells)):<22} {orient:<26} {s0:>5.0%}/{s1:<4.0%} {total:>9d}")
    print()


def drill(df, tags0, tags1, cameras, rho, s_train):
    cam0, cam1 = cameras
    m0, m1 = class_masks(df, tags0, tags1)
    a, b, c, d = contingency(df, m0, m1, cam0, cam1)
    orient, T, total, binding = max_samples(a, b, c, d, rho, s_train)
    A, C = _frac_aligned_conflict(rho, s_train)
    print(f"== drill-down ==")
    print(f"  class0 = {sorted(set(tags0))}   class1 = {sorted(set(tags1))}")
    print(f"  cameras: cam0={cam0}  cam1={cam1}   rho={rho}  split train={s_train:.0%}")
    print(f"  exclusive class sizes (both cameras): class0={a + b}, class1={c + d}")
    print(f"  2x2 counts            cam0={cam0:<14} cam1={cam1}")
    print(f"    class0              {a:<18d} {b}")
    print(f"    class1              {c:<18d} {d}")
    print(f"  P(cam0|class0)={a / (a + b):.1%}  P(cam1|class1)={d / (c + d):.1%}   (natural camera skew)")
    print()
    print(f"  chosen orientation : {orient}")
    print(f"  per-class fractions: aligned={A:.3f}  conflicting={C:.3f}")
    print(f"  >> MAX per class={T}  total={total}  (at rho={rho}, balanced val/test)")
    names = ["c0 aligned", "c0 conflict", "c1 aligned", "c1 conflict"]
    binding_i = int(np.argmin(binding))
    print(f"  binding cell       : {names[binding_i]} (limits the achievable size)")
    print()
    print("  correlation vs samples trade-off (max total at each rho):")
    for r in [0.7, 0.8, 0.9, 0.95, 0.99]:
        _, _, tot, _ = max_samples(a, b, c, d, r, s_train)
        print(f"    rho={r:<4} -> total={tot}")
    print()


def interactive(df, cameras, rho, s_train):
    print("\n-- interactive mode --  (blank class0 to quit)")
    print(f"   tags: {', '.join(ATOMIC_TAGS)}")
    while True:
        try:
            c0 = input("class0 tags (comma-sep): ").strip()
            if not c0:
                break
            c1 = input("class1 tags (comma-sep): ").strip()
            cam = input(f"cameras (comma-sep) [{cameras[0]},{cameras[1]}]: ").strip()
            r = input(f"rho [{rho}]: ").strip()
        except EOFError:
            break
        tags0 = [t.strip().lower() for t in c0.split(",") if t.strip()]
        tags1 = [t.strip().lower() for t in c1.split(",") if t.strip()]
        cams = [s.strip() for s in cam.split(",")] if cam else cameras
        rr = float(r) if r else rho
        if len(cams) != 2:
            print("  need exactly 2 cameras; using defaults")
            cams = cameras
        drill(df, tags0, tags1, cams, rr, s_train)


def _split_counts(T, split_ratios, rho):
    """Per-split (n_aligned, n_conflict) for one class. Train biased at rho; val/test balanced."""
    tr, va, te = split_ratios
    T_tr = round(tr * T)
    T_va = round(va * T)
    T_te = T - T_tr - T_va  # remainder -> test
    out = {}
    for name, Ts, aligned_frac in [("train", T_tr, rho), ("val", T_va, 0.5), ("test", T_te, 0.5)]:
        na = round(aligned_frac * Ts)
        out[name] = (na, Ts - na)
    return out


def build_manifest(df, tags0, tags1, cam0, cam1, rho, split_ratios, seed, per_class, out_path):
    """Select a balanced/correlated subset (camera = fixed acquisition bias) and save it.

    Each (class, camera) cell is sampled WITHOUT replacement and split disjointly into
    train/val/test. Train is camera-biased at ``rho``; val/test are camera-balanced.
    """
    m0, m1 = class_masks(df, tags0, tags1)
    dev = df[CAMERA_COL]
    cells = {
        (0, 0): df.index[m0 & (dev == cam0)].to_numpy(),
        (0, 1): df.index[m0 & (dev == cam1)].to_numpy(),
        (1, 0): df.index[m1 & (dev == cam0)].to_numpy(),
        (1, 1): df.index[m1 & (dev == cam1)].to_numpy(),
    }
    a, b, c, d = (len(cells[k]) for k in [(0, 0), (0, 1), (1, 0), (1, 1)])
    orient, T_max, _, _ = max_samples(a, b, c, d, rho, split_ratios[0])
    aligned_cam = {0: 0, 1: 1} if orient.startswith("class0->cam0") else {0: 1, 1: 0}

    T = T_max if per_class is None else min(per_class, T_max)
    if per_class and per_class > T_max:
        print(f"WARNING: requested per-class {per_class} > feasible {T_max}; using {T_max}.")

    # shrink T until the integer per-split draws fit the available cells
    def needs(T):
        sc = _split_counts(T, split_ratios, rho)
        return sum(na for na, _ in sc.values()), sum(nc for _, nc in sc.values())
    while T > 0:
        need_al, need_cf = needs(T)
        if all(need_al <= len(cells[(cls, aligned_cam[cls])]) and
               need_cf <= len(cells[(cls, 1 - aligned_cam[cls])]) for cls in (0, 1)):
            break
        T -= 1
    if T <= 0:
        print("ERROR: no feasible subset for these settings."); return None

    rng = np.random.default_rng(seed)
    pools = {k: rng.permutation(v) for k, v in cells.items()}  # shuffle once
    cursor = {k: 0 for k in cells}
    sc = _split_counts(T, split_ratios, rho)

    cam_name = {0: cam0, 1: cam1}
    recs = []
    for cls in (0, 1):
        al_cam = aligned_cam[cls]
        for split in ("train", "val", "test"):
            na, nc = sc[split]
            for cam_idx, count in [(al_cam, na), (1 - al_cam, nc)]:
                pool, start = pools[(cls, cam_idx)], cursor[(cls, cam_idx)]
                for idx in pool[start:start + count]:
                    r = df.loc[idx]
                    recs.append({
                        "file": r["File"],
                        "nef": r.get("NEF", ""),
                        "tiff": r.get("TIFF", ""),
                        "target": cls,
                        "camera": cam_name[cam_idx],
                        "camera_idx": cam_idx,
                        "aligned": int(cam_idx == al_cam),
                        "split": split,
                    })
                cursor[(cls, cam_idx)] += count

    out = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nsaved {len(out)} rows -> {out_path}")
    print(f"  class0={sorted(set(tags0))} -> {cam_name[aligned_cam[0]]} | "
          f"class1={sorted(set(tags1))} -> {cam_name[aligned_cam[1]]}")
    print(f"  per-class T={T}  total={len(out)}  (rho={rho}, split={split_ratios})")
    print("  realized counts per split (target, camera_idx -> n):")
    print(out.groupby(["split", "target", "camera_idx"]).size().to_string())
    tr = out[out.split == "train"]
    print(f"  realized train aligned fraction = {(tr['aligned'] == 1).mean():.3f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/raise/RAISE_127.csv")
    ap.add_argument("--rho", type=float, default=0.95, help="target train correlation")
    ap.add_argument("--split", nargs=3, type=float, default=[0.7, 0.1, 0.2],
                    metavar=("TR", "VA", "TE"), help="train/val/test fractions (val+test balanced)")
    ap.add_argument("--cameras", default=None, help="comma-separated 2 cameras (default: top 2 by count)")
    ap.add_argument("--class0", default=None, help="comma-separated tags for class 0 (drill-down)")
    ap.add_argument("--class1", default=None, help="comma-separated tags for class 1 (drill-down)")
    ap.add_argument("--top", type=int, default=15, help="how many ranked pairs to show")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--out", default=None, help="write the selected balanced/correlated subset to this CSV")
    ap.add_argument("--per-class", type=int, default=None, help="cap per-class size (default: max feasible)")
    ap.add_argument("--seed", type=int, default=1, help="sampling seed for the manifest")
    args = ap.parse_args()

    df = load(args.csv)
    cameras = ([s.strip() for s in args.cameras.split(",")] if args.cameras else top_cameras(df, 2))
    s_train = args.split[0]

    if args.class0 and args.class1:
        tags0 = [t.strip().lower() for t in args.class0.split(",")]
        tags1 = [t.strip().lower() for t in args.class1.split(",")]
        drill(df, tags0, tags1, cameras, args.rho, s_train)
        if args.out:
            build_manifest(df, tags0, tags1, cameras[0], cameras[1],
                           args.rho, args.split, args.seed, args.per_class, args.out)
    else:
        overview(df)
        rank_pairs(df, cameras, args.rho, s_train, args.top)

    if args.interactive:
        interactive(df, cameras, args.rho, s_train)


if __name__ == "__main__":
    main()
