"""Metric for the SAE-steering mitigator: original vs steered vs steered+DFR-head.

The custom validation loop in ``mitigators/sae_steer.py`` builds a ``data`` dict with:
  - "targets"
  - "orig", "steered_orig", "steered_dfr"   (prediction arrays for the 3 conditions)
  - one array per bias attribute, keyed by the bias name (e.g. "resize", "jpeg")
  - "sup_pred_<bias>"                        (dedicated-neuron classifier predictions)
  - "ba_groups"                              (list of aligned (target, first-bias) tuples)

For each condition it reports overall / worst-group / bias-aligned / bias-conflict
accuracy; works for any number of bias attributes (single or multi).
"""

import numpy as np
from collections import defaultdict

# best SAE+head = highest steered+retrained worst-group accuracy
sae_steer_dict = {"best": "high", "performance": "steered_dfr_wg"}

_CONDITIONS = ["orig", "steered_orig", "steered_dfr"]
_SPECIAL = set(["targets", "ba_groups"] + _CONDITIONS)


def _bias_names(data):
    return [k for k in data if k not in _SPECIAL and not k.startswith("sup_pred_")]


def _worst_and_avg_group(targets, preds, biases):
    """Group by (target, *biases); return (worst-group acc, avg-group acc)."""
    groups = defaultdict(lambda: [0, 0])  # key -> [correct, total]
    for i in range(len(targets)):
        key = (targets[i],) + tuple(b[i] for b in biases)
        groups[key][1] += 1
        groups[key][0] += int(targets[i] == preds[i])
    accs = [c / t for c, t in groups.values() if t > 0]
    if not accs:
        return 0.0, 0.0
    return float(min(accs)), float(sum(accs) / len(accs))


def _aligned_conflict(targets, preds, first_bias, ba_groups):
    """Bias-aligned vs bias-conflict accuracy, using the first bias attribute."""
    ba_set = set(tuple(g) for g in ba_groups)
    aligned, conflict = [], []
    for i in range(len(targets)):
        correct = int(targets[i] == preds[i])
        if (targets[i], first_bias[i]) in ba_set:
            aligned.append(correct)
        else:
            conflict.append(correct)
    ba = float(np.mean(aligned)) if aligned else 0.0
    bc = float(np.mean(conflict)) if conflict else 0.0
    return ba, bc


def sae_steer(data):
    targets = np.asarray(data["targets"])
    ba_groups = data.get("ba_groups", [])
    bias_names = _bias_names(data)
    biases = [np.asarray(data[b]) for b in bias_names]
    first_bias = biases[0] if biases else np.zeros_like(targets)

    out = {}
    for cond in _CONDITIONS:
        if cond not in data:
            continue
        preds = np.asarray(data[cond])
        out[f"{cond}_acc"] = float(np.mean(preds == targets))
        wg, avg = _worst_and_avg_group(targets, preds, biases)
        out[f"{cond}_wg"] = wg
        out[f"{cond}_avg"] = avg
        ba, bc = _aligned_conflict(targets, preds, first_bias, ba_groups)
        out[f"{cond}_ba"] = ba
        out[f"{cond}_bc"] = bc

    # dedicated-neuron classifier accuracy per attribute
    for b in bias_names:
        sp = data.get(f"sup_pred_{b}")
        if sp is not None:
            out[f"sup_acc_{b}"] = float(np.mean(np.asarray(sp) == np.asarray(data[b])))

    # headline gains (steered+DFR vs original)
    if "steered_dfr_wg" in out and "orig_wg" in out:
        out["wg_gain"] = out["steered_dfr_wg"] - out["orig_wg"]
    if "steered_dfr_bc" in out and "orig_bc" in out:
        out["bc_gain"] = out["steered_dfr_bc"] - out["orig_bc"]
    return out


if __name__ == "__main__":
    # tiny sanity check (2 biases)
    n = 8
    data = {
        "targets": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        "orig": np.array([0, 0, 0, 0, 0, 0, 0, 0]),  # always predicts 0
        "steered_orig": np.array([0, 0, 1, 1, 0, 1, 1, 1]),
        "steered_dfr": np.array([0, 0, 1, 1, 0, 0, 1, 1]),  # perfect
        "resize": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        "jpeg": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        "sup_pred_resize": np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        "sup_pred_jpeg": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        "ba_groups": [(0, 0), (1, 1)],
    }
    from pprint import pprint
    pprint(sae_steer(data))
