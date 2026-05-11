"""Re-run single failed n=50 trials, capturing per-pivot outcomes and
auto-bond miss reasons across all per-fault simulators."""
import sys
from collections import Counter
from typing import Any, Dict, List

from loguru import logger
from src.bullet_sim import BulletSimulator
from run_bullet_monte_carlo import run_single_bullet_trial

logger.remove()
logger.add(sys.stderr, level="WARNING")

# Captures one entry per BulletSimulator instance (i.e. per fault processed).
captured: List[Dict[str, Any]] = []

_real_disconnect = BulletSimulator.disconnect


def _capture_disconnect(self):
    try:
        captured.append({
            "pivot_log": self.get_pivot_diagnostic_log(),
            "auto_bond_stats": self.get_auto_bond_stats(),
            "sim_time": self.sim_time,
        })
    except Exception as e:
        captured.append({"error": repr(e)})
    return _real_disconnect(self)


BulletSimulator.disconnect = _capture_disconnect


def summarize_trial(seed: int, label: str) -> None:
    captured.clear()
    result = run_single_bullet_trial(
        n_modules=50,
        n_faults=5,
        seed=seed,
        trial_id=0,
        mode_2d=False,
        fully_connected=True,
        config_mode="random",
        fault_mode="random",
        restructuring_method="displacement",
        token_strategy="furthest",
        safety_radius=2,
        module_shape="sphere",
    )

    print(f"\n========= {label} (seed={seed}) =========")
    print(f"restored={result.restored}  p1_moves={result.phase1_moves}  "
          f"p2_moves={result.phase2_moves}  p1_iters={result.phase1_iterations}")

    pivot_total = 0
    pivot_completed_clean = 0
    pivot_timed_out = 0
    pos_err_buckets = Counter()
    bond_stats_total = Counter()
    sim_times = []

    for sim in captured:
        if "error" in sim:
            continue
        sim_times.append(sim["sim_time"])
        for ev in sim["pivot_log"]:
            pivot_total += 1
            if ev.get("timed_out"):
                pivot_timed_out += 1
            else:
                pivot_completed_clean += 1
            pe = float(ev.get("pos_error", 0.0))
            if pe < 0.01:
                pos_err_buckets["<0.01"] += 1
            elif pe < 0.05:
                pos_err_buckets["0.01-0.05"] += 1
            elif pe < 0.1:
                pos_err_buckets["0.05-0.1"] += 1
            elif pe < 0.5:
                pos_err_buckets["0.1-0.5"] += 1
            else:
                pos_err_buckets[">=0.5"] += 1
        for k, v in sim["auto_bond_stats"].items():
            bond_stats_total[k] += int(v)

    print(f"\nSimulators (one per fault processed): {len(captured)}")
    print(f"Sim-time per fault: {[f'{t:.1f}s' for t in sim_times]}")

    print(f"\nPivot outcomes (across all faults): total={pivot_total}")
    print(f"  completed_clean: {pivot_completed_clean}")
    print(f"  timed_out:       {pivot_timed_out}")
    print(f"  Final pos_error distribution:")
    for bucket, count in sorted(pos_err_buckets.items()):
        print(f"    {bucket:>10}: {count}")

    print(f"\nAuto-bond stats (across all faults):")
    total_misses = (bond_stats_total['cosine_fail_i']
                    + bond_stats_total['cosine_fail_j']
                    + bond_stats_total['connector_dist_fail'])
    for k in ("pairs_in_com_range", "bonded_ok",
             "cosine_fail_i", "cosine_fail_j", "connector_dist_fail"):
        print(f"  {k:>22}: {bond_stats_total[k]}")
    print(f"  {'(misses total)':>22}: {total_misses}")
    if bond_stats_total["pairs_in_com_range"] > 0:
        bond_rate = (
            bond_stats_total["bonded_ok"]
            / bond_stats_total["pairs_in_com_range"])
        print(f"  bond success rate: {bond_rate:.3%} "
              f"({bond_stats_total['bonded_ok']}/"
              f"{bond_stats_total['pairs_in_com_range']})")


if __name__ == "__main__":
    # seed 2526: trial 0 — the high-activity failure (p1_moves=490, p1_iters=34169)
    # seed 2529: trial 3 — typical low-move stall (p1_moves=2, p1_iters=4950)
    # seed 2532: trial 6 — a SUCCESS for comparison
    summarize_trial(2532, "trial 6: SUCCESS")
    summarize_trial(2529, "trial 3: LOW-MOVE STALL")
    summarize_trial(2526, "trial 0: HIGH-ACTIVITY FAILURE")
