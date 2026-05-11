"""Run trial 0 (high-activity failure) with both fixes individually toggled."""
import sys
from collections import Counter
from typing import Any, Dict, List

from loguru import logger
from src.bullet_sim import BulletSimulator
from run_bullet_monte_carlo import run_single_bullet_trial

logger.remove()
logger.add(sys.stderr, level="ERROR")

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


def run(label: str):
    captured.clear()
    result = run_single_bullet_trial(
        n_modules=50, n_faults=5, seed=2526, trial_id=0,
        mode_2d=False, fully_connected=True,
        config_mode="random", fault_mode="random",
        restructuring_method="displacement",
        token_strategy="furthest", safety_radius=2,
        module_shape="sphere",
    )
    pivot_total = pivot_clean = pivot_timeout = 0
    err_buckets = Counter()
    bond_total = Counter()
    for s in captured:
        if "error" in s:
            continue
        for ev in s["pivot_log"]:
            pivot_total += 1
            if ev.get("timed_out"):
                pivot_timeout += 1
            else:
                pivot_clean += 1
            pe = float(ev.get("pos_error", 0.0))
            if pe < 0.01: err_buckets["<0.01"] += 1
            elif pe < 0.05: err_buckets["0.01-0.05"] += 1
            elif pe < 0.1: err_buckets["0.05-0.1"] += 1
            elif pe < 0.5: err_buckets["0.1-0.5"] += 1
            else: err_buckets[">=0.5"] += 1
        for k, v in s["auto_bond_stats"].items():
            bond_total[k] += int(v)
    print(f"\n========= {label} =========")
    print(f"restored={result.restored}  p1_moves={result.phase1_moves}  "
          f"p1_iters={result.phase1_iterations}")
    print(f"pivots: total={pivot_total} clean={pivot_clean} "
          f"timeout={pivot_timeout}  pos_err: {dict(err_buckets)}")
    print(f"auto-bond: pairs_in_range={bond_total['pairs_in_com_range']} "
          f"bonded={bond_total['bonded_ok']} "
          f"cosine_i_fail={bond_total['cosine_fail_i']} "
          f"cosine_j_fail={bond_total['cosine_fail_j']} "
          f"conn_fail={bond_total['connector_dist_fail']}")


if __name__ == "__main__":
    # A: cosine loose, snap off
    BulletSimulator.AUTO_BOND_COSINE_TOL = 0.985
    BulletSimulator.SNAP_TO_LATTICE_ON_PIVOT_COMPLETE = False
    run("A: cosine=0.985, snap=OFF")

    # B: cosine tight (original), snap on
    BulletSimulator.AUTO_BOND_COSINE_TOL = 0.99985
    BulletSimulator.SNAP_TO_LATTICE_ON_PIVOT_COMPLETE = True
    run("B: cosine=0.99985 (orig), snap=ON")
