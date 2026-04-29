"""
Run PyBullet async damage response on an n-module Y-line (fault at chosen index).

Line along +Y at x=z=0, matching render_line_fault.py via shared scenario builder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger

from line_fault_scenario import build_y_line_fault_scenario
from src.bullet_bridge import AsyncSimRunner, print_diagnostic_summary
from src.bullet_sim import BulletSimulator


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_json_ready(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return tuple(_json_ready(v) for v in obj)
    return obj


def _parse_args():
    p = argparse.ArgumentParser(
        description="PyBullet decentralized line-fault damage response demo.",
    )
    p.add_argument("--n", type=int, default=11, help="number of modules")
    p.add_argument(
        "--fault-index",
        type=int,
        default=None,
        help="fault body index (default: n//2)",
    )
    p.add_argument("--gui", action="store_true", help="PyBullet GUI client")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="write full JSON report to this path",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="NumPy RNG seed (PyBullet remains partially nondeterministic)",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="loguru level (e.g. DEBUG, INFO, WARNING)",
    )
    p.add_argument(
        "--max-sim-time",
        type=float,
        default=None,
        help=(
            "optional per-phase simulated time cap in seconds for coagulation and "
            "restructuring (default: no cap). Each pivot still uses sim MAX_PIVOT_TIME."
        ),
    )
    p.add_argument(
        "--token-strategy",
        type=str,
        default="furthest",
        choices=("furthest", "nearest"),
        help="restructuring token selection strategy",
    )
    p.add_argument(
        "--momentum-diagnostics",
        action="store_true",
        help="record linear momentum & COM per physics substep (see --momentum-diag-out)",
    )
    p.add_argument(
        "--momentum-diag-out",
        type=str,
        default=None,
        help="write momentum/COM diagnostic JSON (implies --momentum-diagnostics if set)",
    )
    p.add_argument(
        "--momentum-diag-subsample",
        type=int,
        default=1,
        help="store every Nth substep in samples (1=all; use 10+ for long runs)",
    )
    p.add_argument(
        "--momentum-diag-max-samples",
        type=int,
        default=200_000,
        help="cap stored sample rows (summary maxes still count all substeps)",
    )
    p.add_argument(
        "--pivot-attract-scale",
        type=float,
        default=0.1,
        help="scale rigid pivot inverse-square attract gain vs 20 N·m² (default 0.1 = 10× weaker)",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())

    if args.seed is not None:
        np.random.seed(args.seed)

    scenario = build_y_line_fault_scenario(n=args.n, fault_body_idx=args.fault_index)

    kwargs = {}
    if args.max_sim_time is not None:
        kwargs["max_sim_time"] = args.max_sim_time
    # Rolling pivots may need long wall time; stall window must exceed pivot patience.
    kwargs.setdefault("stall_interval", 20.0)
    kwargs.setdefault("stall_patience", 18)

    BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
    mom_diag = args.momentum_diagnostics or bool(args.momentum_diag_out)
    sim_kw: Dict[str, Any] = {}
    if mom_diag:
        sim_kw["momentum_diagnostics"] = True
        sim_kw["momentum_diag_subsample"] = max(1, int(args.momentum_diag_subsample))
        sim_kw["momentum_diag_max_samples"] = max(0, int(args.momentum_diag_max_samples))
    sim_kw["pivot_attract_scale"] = float(args.pivot_attract_scale)
    sim = BulletSimulator(
        scenario.n,
        scenario.pos0,
        scenario.bonded0,
        gui=args.gui,
        **sim_kw,
    )
    report: Dict[str, Any] = {}
    try:
        bridge = AsyncSimRunner(sim, scenario.module_ids, scenario.body_indices, **kwargs)
        result = bridge.run_damage_response(
            fault_id=scenario.fault_id,
            fault_adjacent=scenario.fault_adjacent,
            fault_body_idx=scenario.fault_body_idx,
            pre_damage_neighbor_slots=scenario.pre_damage_neighbor_slots,
            token_strategy=args.token_strategy,
        )
        print_diagnostic_summary(result)

        report = {
            "scenario": {
                "n": scenario.n,
                "fault_id": scenario.fault_id,
                "fault_body_idx": scenario.fault_body_idx,
                "fault_adjacent": scenario.fault_adjacent,
            },
            "args": {k: getattr(args, k) for k in vars(args)},
            "result": result,
        }
        if mom_diag:
            mom_report = sim.get_momentum_diagnostic_report()
            report["momentum_diagnostics"] = mom_report
            s = mom_report["summary"]
            print("\n=== MOMENTUM / COM DIAGNOSTICS ===")
            print(f"  Substeps: {s['substeps']}, rolling substeps: {s.get('rolling_substeps', 0)}")
            print(f"  max ||P||: {s['max_norm_P']:.6g}  (start {s['norm_P_at_start']:.6g}, end {s['norm_P_at_end']:.6g})")
            print(f"  max ||dP|| across p.stepSimulation: {s['max_delta_P_bullet']:.6g}")
            print(f"  max ||dP|| across rolling post-step: {s.get('max_delta_P_rolling', 0.0):.6g}")
            print(f"  ||dP|| start->end: {s['delta_P_start_to_end']:.6g}")
            print(f"  COM displacement (mass-weighted): {s['com_displacement_mass_weighted']:.6g} m")
            print(f"  max |mean(pos)-weighted_COM| mismatch: {s['max_com_mean_weighted_mismatch']:.6g} m")
            print(f"  bridge com_drift vs sim COM displacement: "
                  f"{result.get('com_drift', 0):.6g} vs {s['com_displacement_mass_weighted']:.6g}")
    finally:
        sim.disconnect()

    if args.momentum_diag_out and report.get("momentum_diagnostics"):
        mom_path = args.momentum_diag_out
        with open(mom_path, "w", encoding="utf-8") as f:
            json.dump(_json_ready(report["momentum_diagnostics"]), f, indent=2)
        print(f"Wrote momentum diagnostic: {mom_path}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(_json_ready(report), f, indent=2)
        print(f"Wrote report: {args.out}")


if __name__ == "__main__":
    main()
