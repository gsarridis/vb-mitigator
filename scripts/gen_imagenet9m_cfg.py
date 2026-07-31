#!/usr/bin/env python
"""Generate a temporary imagenet9m config from a base config for the sweep.

Applies the per-run overrides that tools/train.py cannot set from the CLI:
  * MODEL.TYPE
  * DATASET.IMAGENET9M.CORRELATION            (single scenarios)
    DATASET.IMAGENET9M.CORRELATION_JPEG/RESIZE (multi scenario)
  * EXPERIMENT.TAG   (so each run gets its own output dir)
  * checkpoint-dependency paths, rewritten to sweep-specific locations:
      - BCCs      : bcc_jpeg/erm   -> bcc_jpeg_<model>/erm     (and bcc_resize)
      - bpa CKPT  : imagenet9m_baselines/<scenario>/erm/best
                    -> imagenet9m_baselines/<tag>/erm/best

Usage:
  gen_imagenet9m_cfg.py BASE OUT SCENARIO MODEL CORR TAG
"""
import sys
import yaml


def main():
    base, out, scenario, model, corr, tag = sys.argv[1:7]

    with open(base) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("MODEL", {})["TYPE"] = model
    cfg.setdefault("EXPERIMENT", {})["TAG"] = tag

    im = cfg["DATASET"]["IMAGENET9M"]
    if scenario == "multi":
        im["CORRELATION_JPEG"] = float(corr)
        im["CORRELATION_RESIZE"] = float(corr)
    else:
        im["CORRELATION"] = float(corr)

    def rewrite(s):
        if not isinstance(s, str):
            return s
        s = s.replace("bcc_jpeg/erm", f"bcc_jpeg_{model}/erm")
        s = s.replace("bcc_resize/erm", f"bcc_resize_{model}/erm")
        s = s.replace(
            f"imagenet9m_baselines/{scenario}/erm/best",
            f"imagenet9m_baselines/{tag}/erm/best",
        )
        return s

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return rewrite(o)

    cfg = walk(cfg)
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


if __name__ == "__main__":
    main()
