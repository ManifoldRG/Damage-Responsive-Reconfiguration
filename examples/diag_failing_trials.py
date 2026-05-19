"""Replay a handful of canonical bullet MC trials with diagnostics enabled.

Pulls 9 trials from the most recent simul-fault sweep:
  - 3 n=50/f=10 random (2 failing + 1 succeeding)
  - 3 n=50/f=15 localized (2 failing + 1 succeeding)
  - 3 n=10/f=2 random successful (low-N reference)

Re-runs each through ``run_single_bullet_trial`` with --dump-diagnostics
pointing at a chosen output dir. Sequential so the dumps don't interleave.

Usage:
  py examples/diag_failing_trials.py
  py examples/diag_failing_trials.py --quick     # one n=10 trial only
  py examples/diag_failing_trials.py --sweep-dir <path> --out-dir <path>
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_bullet_monte_carlo import run_single_bullet_trial


def pick_trials(trials_csv: str) -> List[Tuple[str, dict]]:
    """Return list of (label, kwargs) for the canonical 9 trials."""
    df = pd.read_csv(trials_csv)

    def take(filter_fn, n, label_prefix):
        sub = df[df.apply(filter_fn, axis=1)].head(n)
        out = []
        for _, row in sub.iterrows():
            tag = f"{label_prefix}_t{int(row.trial_id)}_s{int(row.seed)}"
            out.append((tag, row.to_dict()))
        return out

    picks: List[Tuple[str, dict]] = []
    picks += take(
        lambda r: r.n_modules == 50 and r.n_faults == 10
        and r.fault_mode == "random" and not r.restored, 2, "n50_f10_rand_FAIL")
    picks += take(
        lambda r: r.n_modules == 50 and r.n_faults == 10
        and r.fault_mode == "random" and r.restored, 1, "n50_f10_rand_OK")
    picks += take(
        lambda r: r.n_modules == 50 and r.n_faults == 15
        and r.fault_mode == "localized" and not r.restored, 2, "n50_f15_loc_FAIL")
    picks += take(
        lambda r: r.n_modules == 50 and r.n_faults == 15
        and r.fault_mode == "localized" and r.restored, 1, "n50_f15_loc_OK")
    picks += take(
        lambda r: r.n_modules == 10 and r.n_faults == 2
        and r.fault_mode == "random" and r.restored, 3, "n10_f2_rand_OK")
    return picks


def replay(label: str, row: dict, out_dir: str) -> None:
    print(f"\n=== Replay: {label} ===", flush=True)
    kwargs = dict(
        n_modules=int(row["n_modules"]),
        n_faults=int(row["n_faults"]),
        seed=int(row["seed"]),
        trial_id=int(row["trial_id"]),
        mode_2d=False,
        fully_connected=True,
        fault_mode=str(row["fault_mode"]),
        temperature=0.01,
        pivot_exclusion_radius=4,
        max_phase_time=180.0,
        stall_interval=10.0,
        stall_patience=16,
        dt=0.1,
        restructuring_method=str(row["restructuring_method"]),
        token_strategy=str(row["token_strategy"]),
        safety_radius=int(row["safety_radius"]),
        module_shape="sphere",
        max_pivot_time=None,
        use_flood_echo=False,
        token_gen_interval=0.1,
        dump_diagnostics_dir=out_dir,
    )
    result = run_single_bullet_trial(**kwargs)
    print(f"    restored={result.restored} "
          f"p1_moves={result.phase1_moves} p2_moves={result.phase2_moves} "
          f"p1_ticks={result.phase1_iterations}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir", type=str,
        default="bullet_mc_results/sweep_simul_sr4_noflood_tg01_t20",
        help="Sweep directory whose trials.csv we pick from.")
    parser.add_argument(
        "--out-dir", type=str, default="tmp/n50_diag",
        help="Where per-trial JSON dumps land.")
    parser.add_argument("--quick", action="store_true",
                        help="Run just one n=10 trial as a smoke test.")
    args = parser.parse_args()

    runs = sorted(os.listdir(args.sweep_dir))
    if not runs:
        sys.exit(f"No subdirs in {args.sweep_dir}")
    trials_csv = os.path.join(args.sweep_dir, runs[-1], "trials.csv")
    if not os.path.exists(trials_csv):
        sys.exit(f"No trials.csv at {trials_csv}")
    print(f"Reading trials from {trials_csv}", flush=True)

    picks = pick_trials(trials_csv)
    if args.quick:
        picks = [p for p in picks if p[0].startswith("n10")][:1]

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Replaying {len(picks)} trials -> {args.out_dir}", flush=True)
    for label, row in picks:
        replay(label, row, args.out_dir)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
