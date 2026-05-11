"""
Async decentralized simulation runner.

Drives the PyBullet world with decentralized agent policies.
The main loop: advance physics by a time slice, then let each
agent check its state and act. Repeat until done.
"""

import numpy as np
from typing import Any, Dict, List, Set
from loguru import logger

from .bullet_sim import BulletSimulator
from .agent_policy import (
    DecentralizedCoagulation,
    DecentralizedRestructuring,
)


class AsyncSimRunner:
    """
    Runs the full async decentralized damage response.

    Physics advances in small time slices. Between slices,
    each module agent checks its state and acts independently.

    By default there is **no** overall phase time limit: coagulation and
    restructuring run until success, stall, or (restructuring) no tokens.
    Optional ``max_sim_time`` caps **each** phase's simulated duration.
    Individual pivots still use ``BulletSimulator.MAX_PIVOT_TIME``.
    """

    TICK_DT = 0.05          # agent tick cadence (advances 5 physics substeps)
    TRAJ_INTERVAL = 1.0 / 15  # trajectory sample interval (15 fps real-time)
    STALL_INTERVAL = 10.0      # seconds between stall checks
    STALL_PATIENCE = 3         # consecutive stalls before abort

    def __init__(self, sim: BulletSimulator, module_ids: List[str],
                 body_indices: Dict[str, int], *,
                 max_sim_time: float | None = None,
                 stall_interval: float | None = None,
                 stall_patience: int | None = None,
                 record_trajectories: bool = True):
        self.sim = sim
        self.module_ids = module_ids
        self.body_indices = body_indices
        self._idx_to_mid = {v: k for k, v in body_indices.items()}
        # None = no wall-clock limit on phase duration (exit on success / stall only).
        # Per-maneuver timeout remains BulletSimulator.MAX_PIVOT_TIME.
        self._max_sim_time = (
            float(max_sim_time) if max_sim_time is not None else None
        )
        self._stall_interval = (
            float(stall_interval) if stall_interval is not None else self.STALL_INTERVAL
        )
        self._stall_patience = (
            int(stall_patience) if stall_patience is not None else self.STALL_PATIENCE
        )

        # Track trajectories for ALL bodies (including faulty); skipped
        # entirely when record_trajectories=False (headless MC).
        self._record_trajectories = bool(record_trajectories)
        self._trajectories: Dict[str, List[np.ndarray]] = {
            mid: [] for mid in body_indices
        }
        self._bond_snapshots: List[List[tuple]] = []
        self._next_traj_sample = 0.0

    def _sample_trajectory(self):
        """Record current positions and bond state for all bodies if it's time."""
        if not self._record_trajectories:
            return
        if self.sim.sim_time >= self._next_traj_sample - 1e-6:
            pos = self.sim.get_positions()
            for mid, idx in self.body_indices.items():
                if mid in self._trajectories:
                    self._trajectories[mid].append(pos[idx].copy())
            # Snapshot bond pairs as (mid_a, mid_b) strings
            bm = self.sim.get_bond_matrix()
            bonds = []
            for i in range(self.sim.N):
                for j in range(i + 1, self.sim.N):
                    if bm[i, j]:
                        mid_i = self._idx_to_mid.get(i)
                        mid_j = self._idx_to_mid.get(j)
                        if mid_i and mid_j:
                            bonds.append((mid_i, mid_j))
            self._bond_snapshots.append(bonds)
            self._next_traj_sample = self.sim.sim_time + self.TRAJ_INTERVAL

    def run_coagulation(self, fault_id: str,
                        fault_adjacent: List[str],
                        fault_body_idx: int,
                        ) -> Dict[str, Any]:
        """
        Run async decentralized coagulation until connected, stalled, or
        optional per-phase ``max_sim_time`` (see ``__init__``).

        Args:
            fault_id: ID of faulty module
            fault_adjacent: module IDs that were neighbors of fault
            fault_body_idx: PyBullet body index of the fault module
        """
        logger.info("Starting async coagulation (fault={})", fault_id)

        coag = DecentralizedCoagulation(
            self.sim, fault_id, self.module_ids, self.body_indices)
        coag.set_fault_adjacent(fault_adjacent, fault_body_idx)

        start_time = self.sim.sim_time
        stall_check_time = start_time
        last_move_count = 0
        stall_count = 0

        while True:
            # Advance physics
            self.sim.step(self.TICK_DT)
            self._sample_trajectory()

            # Let agents act
            any_active = coag.tick()

            # Check if connected
            if coag.is_connected():
                logger.info("Coagulation complete: connected after {:.1f}s, "
                            "{} moves", self.sim.sim_time - start_time,
                            coag.total_moves)
                break

            # Stall detection
            if self.sim.sim_time - stall_check_time > self._stall_interval:
                if coag.total_moves == last_move_count:
                    stall_count += 1
                    if stall_count >= self._stall_patience:
                        logger.warning("Coagulation stalled after {:.1f}s",
                                       self.sim.sim_time - start_time)
                        break
                else:
                    stall_count = 0
                last_move_count = coag.total_moves
                stall_check_time = self.sim.sim_time

            if not any_active and not coag.is_connected():
                # All agents idle but not connected — need to re-emit tokens
                coag.set_fault_adjacent(fault_adjacent, fault_body_idx)

            if (
                self._max_sim_time is not None
                and (self.sim.sim_time - start_time)
                >= self._max_sim_time - 1e-9
            ):
                logger.info(
                    "Coagulation: phase time limit {:.1f}s reached",
                    self._max_sim_time,
                )
                break

        # Collect moved modules
        moved_mids = set()
        for entry in coag.move_log:
            moved_mids.add(entry["module"])

        return {
            "connected": coag.is_connected(),
            "total_moves": coag.total_moves,
            "move_log": coag.move_log,
            "moved_modules": moved_mids,
            "sim_time": self.sim.sim_time - start_time,
        }

    def run_restructuring(self, coag_moved: Set[str],
                          pre_damage_neighbor_slots: Dict[str, List[np.ndarray]],
                          token_strategy: str = "furthest",
                          ) -> Dict[str, Any]:
        """
        Run async decentralized restructuring until stalled or no tokens, or
        optional per-phase ``max_sim_time`` (see ``__init__``).
        """
        logger.info("Starting async restructuring ({} movers to restore)",
                     len(coag_moved))

        restruct = DecentralizedRestructuring(
            self.sim, self.module_ids, self.body_indices,
            coag_moved, pre_damage_neighbor_slots, token_strategy)
        restruct.generate_initial_tokens()

        start_time = self.sim.sim_time
        stall_check_time = start_time
        last_move_count = 0
        stall_count = 0

        while True:
            self.sim.step(self.TICK_DT)
            self._sample_trajectory()

            any_active = restruct.tick()

            if self.sim.sim_time - stall_check_time > self._stall_interval:
                if restruct.total_moves == last_move_count:
                    stall_count += 1
                    if stall_count >= self._stall_patience:
                        logger.warning("Restructuring stalled after {:.1f}s",
                                       self.sim.sim_time - start_time)
                        break
                else:
                    stall_count = 0
                    last_move_count = restruct.total_moves
                stall_check_time = self.sim.sim_time

            if not any_active:
                # Re-emit tokens for remaining empty slots
                restruct.generate_initial_tokens()
                # Check if any tokens were actually generated
                has_tokens = any(a.incoming_tokens for a in restruct.agents.values())
                if not has_tokens:
                    logger.info("Restructuring: no more empty slots to fill")
                    break

            if (
                self._max_sim_time is not None
                and (self.sim.sim_time - start_time)
                >= self._max_sim_time - 1e-9
            ):
                logger.info(
                    "Restructuring: phase time limit {:.1f}s reached",
                    self._max_sim_time,
                )
                break

        return {
            "total_moves": restruct.total_moves,
            "move_log": restruct.move_log,
            "sim_time": self.sim.sim_time - start_time,
        }

    def run_damage_response(self, fault_id: str,
                            fault_adjacent: List[str],
                            fault_body_idx: int,
                            pre_damage_neighbor_slots: Dict[str, List[np.ndarray]],
                            token_strategy: str = "furthest",
                            ) -> Dict[str, Any]:
        """Run full damage response: coagulation then restructuring."""
        self.sim.clear_pivot_diagnostics()
        init_com = np.mean(self.sim.get_positions(), axis=0)
        self._sample_trajectory()

        # Phase 1
        phase1 = self.run_coagulation(fault_id, fault_adjacent, fault_body_idx)

        # Phase 2
        phase2 = self.run_restructuring(
            phase1["moved_modules"],
            pre_damage_neighbor_slots,
            token_strategy)

        # Final state
        final_pos = self.sim.get_positions()
        final_com = np.mean(final_pos, axis=0)
        com_drift = float(np.linalg.norm(final_com - init_com))

        total_moves = phase1["total_moves"] + phase2["total_moves"]
        logger.info("Damage response complete: {} total moves, "
                     "COM drift={:.4f}", total_moves, com_drift)

        # Build trajectory report
        traj_report = {}
        for mid, traj in self._trajectories.items():
            traj_report[mid] = [pos.tolist() for pos in traj]

        physics_metrics = self._bond_length_metrics(final_pos)

        return {
            "phase1": phase1,
            "phase2": phase2,
            "total_moves": total_moves,
            "com_drift": com_drift,
            "trajectories": traj_report,
            "bond_snapshots": [list(bs) for bs in self._bond_snapshots],
            "move_log": phase1["move_log"] + phase2["move_log"],
            "physics_metrics": physics_metrics,
            "pivot_diagnostics": self.sim.get_pivot_diagnostic_log(),
        }

    def _bond_length_metrics(self, positions: np.ndarray) -> Dict[str, float]:
        """Edge length stats for bonded pairs vs nominal 1 m spacing."""
        bm = self.sim.get_bond_matrix()
        lengths: List[float] = []
        nom = float(self.sim.NOMINAL_DIST)
        for i in range(self.sim.N):
            for j in range(i + 1, self.sim.N):
                if bm[i, j]:
                    lengths.append(float(np.linalg.norm(positions[i] - positions[j])))
        if not lengths:
            return {
                "bond_count": 0,
                "nominal_m": nom,
                "min_m": 0.0,
                "max_m": 0.0,
                "rmse_m": 0.0,
            }
        arr = np.array(lengths)
        return {
            "bond_count": float(len(lengths)),
            "nominal_m": nom,
            "min_m": float(arr.min()),
            "max_m": float(arr.max()),
            "rmse_m": float(np.sqrt(np.mean((arr - nom) ** 2))),
        }


def print_diagnostic_summary(result: Dict[str, Any]) -> None:
    """Print summary of async simulation results."""
    p1 = result.get("phase1", {})
    p2 = result.get("phase2", {})

    print("\n=== ASYNC DECENTRALIZED SIMULATION ===")
    print(f"Phase 1: connected={p1.get('connected', False)}, "
          f"moves={p1.get('total_moves', 0)}, "
          f"sim_time={p1.get('sim_time', 0):.1f}s")
    print(f"Phase 2: moves={p2.get('total_moves', 0)}, "
          f"sim_time={p2.get('sim_time', 0):.1f}s")
    print(f"Total moves: {result.get('total_moves', 0)}")
    print(f"COM drift: {result.get('com_drift', 0):.4f}")
    pm = result.get("physics_metrics") or {}
    if pm.get("bond_count", 0):
        print(f"Bond lengths: n={int(pm['bond_count'])}, "
              f"min={pm['min_m']:.4f}m, max={pm['max_m']:.4f}m, "
              f"RMSE vs {pm.get('nominal_m', 1.0):.2f}m={pm['rmse_m']:.4f}m")

    move_log = result.get("move_log", [])
    if move_log:
        print(f"\nMove log ({len(move_log)} moves):")
        for i, m in enumerate(move_log):
            print(f"  {i+1:>3}. {m['module']:>4} -> {m['to']} "
                  f"(axis={m['axis']}, t={m['sim_time']:.1f}s)")
    print()
