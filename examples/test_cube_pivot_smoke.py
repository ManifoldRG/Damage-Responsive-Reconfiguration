"""Smoke test for the cube-module variant.

Runs one tiny trial with module_shape="cube" to confirm the whole stack
(BulletSimulator cube collision shape + edge-hinge start_pivot, agent_policy
lateral-only enumeration with swept-clear check, run_single_bullet_trial CLI
plumbing) executes end-to-end without crashes.

Usage:
    py examples/test_cube_pivot_smoke.py
"""
import os
import sys

import numpy as np

# Allow running directly from the repo root: examples/ is a sibling of
# run_bullet_monte_carlo.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_bullet_monte_carlo import run_single_bullet_trial


def main():
    print("=" * 60)
    print("CUBE PIVOT SMOKE TEST")
    print("=" * 60)
    any_moves = False
    for seed in (42, 7, 19, 31, 100):
        for n in (8, 10):
            result = run_single_bullet_trial(
                n_modules=n,
                n_faults=1,
                seed=seed,
                trial_id=0,
                mode_2d=False,
                fully_connected=True,
                temperature=0.01,
                pivot_exclusion_radius=2,
                max_phase_time=60.0,
                stall_interval=10.0,
                stall_patience=4,
                dt=0.1,
                restructuring_method="rendezvous",
                token_strategy="furthest",
                safety_radius=2,
                module_shape="cube",
            )
            print(f"  seed={seed:3d} n={n:2d}  restored={result.restored}  "
                  f"p1_moves={result.phase1_moves}  p2_moves={result.phase2_moves}  "
                  f"shape_diff={result.shape_difference}")
            if result.phase1_moves > 0:
                any_moves = True
    print()
    if any_moves:
        print("  OK: cube pivots executed in at least one configuration.")
    else:
        print("  WARNING: 0 moves across all configs — eligibility likely too tight "
              "or actuation not driving the lever.")


if __name__ == "__main__":
    main()
