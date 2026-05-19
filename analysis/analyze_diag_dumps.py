"""Analyze per-trial diagnostic JSON dumps from run_single_bullet_trial.

Reads all ``trial_*.json`` files in a directory and prints / saves:
  * Per-trial summary table (one row per JSON)
  * Aggregate comparison grouped by (n_modules, restored)
  * Best-guess failure-mode classification

Usage:
  py analysis/analyze_diag_dumps.py tmp/n50_diag/
  py analysis/analyze_diag_dumps.py tmp/n50_diag/ --csv analysis/n50_diag_summary.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

POS_ERR_BINS = [0.0, 0.01, 0.05, 0.2, 100.0]
POS_ERR_LABELS = ["<0.01", "0.01-0.05", "0.05-0.2", ">0.2"]


def load_dumps(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(os.path.join(path, "trial_*.json"))):
        with open(p) as f:
            d = json.load(f)
            d["__path"] = p
            out.append(d)
    return out


def summarize_trial(d: Dict[str, Any]) -> Dict[str, Any]:
    pivots = d.get("pivot_log", [])
    drift = d.get("attitude_drift_log", [])
    bonds = d.get("auto_bond_stats", {}) or {}
    comps = d.get("final_components", [])
    n_active = d.get("n_active_modules", 0)

    n_pivots = len(pivots)
    timed_out = sum(1 for p in pivots if p.get("timed_out"))
    settled_pos = sum(1 for p in pivots
                      if p.get("settle_exit") == "pos_tol")
    settled_dur = sum(1 for p in pivots
                      if p.get("settle_exit") == "duration")
    settled_none = sum(1 for p in pivots
                       if p.get("settle_exit") in (None, "none"))

    pos_errs = np.array([float(p.get("pos_error", 0.0)) for p in pivots])
    pos_hist = np.histogram(pos_errs, bins=POS_ERR_BINS)[0] if len(pos_errs) else np.zeros(4, dtype=int)
    angle_errs = np.array([float(p.get("angle_error", 0.0)) for p in pivots])

    pairs = max(int(bonds.get("pairs_in_com_range", 0)), 1)
    ok = int(bonds.get("bonded_ok", 0))
    cos_fail = int(bonds.get("cosine_fail_i", 0)) + int(bonds.get("cosine_fail_j", 0))
    dist_fail = int(bonds.get("connector_dist_fail", 0))

    # Attitude drift is sampled over ALL N bodies (faults included), so use
    # total body count as the denominator, not active-module count.
    n_total = d.get("n_modules", n_active)
    drift_arr = np.array(drift) if drift else np.zeros((0, 4))
    if drift_arr.size:
        max_below = int(drift_arr[:, 3].max())
        final_mean_cos = float(drift_arr[-1, 1])
        final_min_cos = float(drift_arr[-1, 2])
        max_below_frac = max_below / n_total if n_total else 0.0
    else:
        max_below = 0
        final_mean_cos = 1.0
        final_min_cos = 1.0
        max_below_frac = 0.0

    comp_sizes = sorted((len(c) for c in comps), reverse=True)
    n_components = len(comp_sizes)
    largest = comp_sizes[0] if comp_sizes else 0
    largest_frac = largest / n_active if n_active else 0.0

    return {
        "trial_id": d["trial_id"],
        "seed": d["seed"],
        "n": d["n_modules"],
        "f": d["n_faults"],
        "mode": d["fault_mode"],
        "restored": d["connected"],
        "p1_moves": d["total_phase1_moves"],
        "p2_moves": d["total_phase2_moves"],
        "p1_ticks": d["total_phase1_ticks"],
        "n_pivots": n_pivots,
        "%timed_out": _pct(timed_out, n_pivots),
        "%settle_pos": _pct(settled_pos, n_pivots),
        "%settle_dur": _pct(settled_dur, n_pivots),
        "%settle_none": _pct(settled_none, n_pivots),
        "pos<0.01": int(pos_hist[0]),
        "pos<0.05": int(pos_hist[1]),
        "pos<0.2": int(pos_hist[2]),
        "pos>0.2": int(pos_hist[3]),
        "mean_pos_err": float(pos_errs.mean()) if pos_errs.size else 0.0,
        "mean_ang_err": float(angle_errs.mean()) if angle_errs.size else 0.0,
        "ab_success": ok / pairs if pairs else 0.0,
        "ab_pairs": int(bonds.get("pairs_in_com_range", 0)),
        "ab_cos_fail": cos_fail,
        "ab_dist_fail": dist_fail,
        "max_below_0.985": max_below,
        "max_below_frac": max_below_frac,
        "final_mean_cos": final_mean_cos,
        "final_min_cos": final_min_cos,
        "n_components": n_components,
        "largest_frac": largest_frac,
        "rev_coag": d.get("reversal_count_coag", 0),
        "rev_restruct": d.get("reversal_count_restruct", 0),
    }


def _pct(num: int, den: int) -> float:
    return 100.0 * num / max(den, 1)


def classify_failure(s: Dict[str, Any]) -> str:
    """Bucket a failing trial. ``s`` is the row from summarize_trial."""
    if s["restored"]:
        return "OK"
    # Fragmented: more than one component, and largest component < 90% of active.
    if s["n_components"] >= 2 and s["largest_frac"] < 0.9:
        return "FRAGMENTED"
    # Settle-cap dominated: ≥50% of pivots ended via the duration cap.
    if s["%settle_dur"] >= 50.0 and s["n_pivots"] >= 5:
        return "SETTLE_CAP"
    # Cascading drift: significant share of modules drifted off-axis.
    if s["max_below_frac"] >= 0.20:
        return "DRIFT"
    # Thrashing: many pivots, low success rate.
    if s["n_pivots"] >= 30 and (s["%timed_out"] >= 40 or s["rev_coag"] >= 10):
        return "THRASHING"
    return "OTHER"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Directory of trial_*.json dumps")
    parser.add_argument("--csv", type=str, default=None,
                        help="Optional path to write per-trial summary CSV.")
    args = parser.parse_args()

    dumps = load_dumps(args.path)
    if not dumps:
        sys.exit(f"No trial_*.json in {args.path}")
    rows = [summarize_trial(d) for d in dumps]
    for r in rows:
        r["class"] = classify_failure(r)
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 50)
    print(f"\n=== {len(df)} trials ===")
    cols_main = ["trial_id", "n", "f", "mode", "restored", "class",
                 "n_pivots", "%timed_out", "%settle_pos", "%settle_dur",
                 "%settle_none", "mean_pos_err", "ab_success",
                 "max_below_frac", "final_mean_cos",
                 "n_components", "largest_frac", "rev_coag"]
    print(df[cols_main].to_string(index=False))

    print("\n=== Aggregate by (n, restored) ===")
    agg = df.groupby(["n", "restored"]).agg(
        trials=("trial_id", "count"),
        avg_pivots=("n_pivots", "mean"),
        avg_pct_timeout=("%timed_out", "mean"),
        avg_pct_settle_dur=("%settle_dur", "mean"),
        avg_mean_pos_err=("mean_pos_err", "mean"),
        avg_ab_success=("ab_success", "mean"),
        avg_max_below_frac=("max_below_frac", "mean"),
        avg_final_mean_cos=("final_mean_cos", "mean"),
        avg_n_components=("n_components", "mean"),
        avg_largest_frac=("largest_frac", "mean"),
        avg_rev_coag=("rev_coag", "mean"),
    )
    print(agg.to_string())

    print("\n=== Failure-mode tally ===")
    print(df["class"].value_counts().to_string())

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
