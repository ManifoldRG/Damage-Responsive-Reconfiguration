"""
Bridge between the graph-based UDQDG reconfiguration algorithm and the
OpenModelica physical simulation.

DEPRECATED: This bridge calls ``UDQDGSystem.full_damage_response`` with
``record_steps=True`` to harvest a ``PivotStep`` / ``RestorationStep``
sequence, which is then replayed in Modelica. Both
``full_damage_response`` and the step-log format were removed in the
Graph Parity refactor — all simulation now flows through
``GraphSimulator`` / ``BulletSimulator`` directly. If a Modelica path is
still desired, port the policy callbacks from ``src/agent_policy.py``
into a Modelica-driving simulator wrapper instead of replaying a
pre-recorded step list.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from .udqdg_system import UDQDGSystem, PivotStep, RestorationStep
from .modelica_sim import ModelicaSimulator, SimResult, SegmentDiagnostics


@dataclass
class StepDiagnostics:
    """Diagnostics for one bridge-level step (may have 1-2 sim segments)."""
    step_index: int
    step_type: str                              # "corner" or "lateral"
    module_id: str
    segment_diagnostics: List[SegmentDiagnostics] = field(default_factory=list)
    position_errors: Dict[str, float] = field(default_factory=dict)
    velocity_residual_before_rigidify: float = 0.0
    reconnection_distances: Dict[str, float] = field(default_factory=dict)
    com_drift_this_step: float = 0.0


class ModelicaBridge:
    """
    Synchronizes a UDQDGSystem with a ModelicaSimulator and replays
    algorithm-planned pivot sequences as physically-driven motions.

    The key output is the position error between the graph algorithm's
    idealized positions and the physics simulation's actual positions.
    """

    def __init__(self, system: UDQDGSystem, simulator: ModelicaSimulator):
        self.system = system
        self.sim = simulator

        # Module ID <-> index mapping (Modelica uses 0-based internally)
        self._mid_to_idx: Dict[str, int] = {}
        self._idx_to_mid: Dict[int, str] = {}

        # Current physical state
        self._N: int = 0
        self._pos: Optional[np.ndarray] = None
        self._vel: Optional[np.ndarray] = None
        self._bonded: Optional[np.ndarray] = None

        # Trajectory accumulation across all steps
        self._trajectories: Dict[str, List[np.ndarray]] = {}

        # Diagnostics accumulation
        self._step_diagnostics: List[StepDiagnostics] = []
        self._step_counter: int = 0

    def sync_from_graph(self):
        """Mirror the current UDQDGSystem state into the simulator."""
        # Build module list (active, non-faulty only)
        active_modules = [
            (mid, mod) for mid, mod in self.system.modules.items()
            if mod.is_active and not mod.is_faulty
        ]
        self._N = len(active_modules)
        self._mid_to_idx = {}
        self._idx_to_mid = {}
        self._pos = np.zeros((self._N, 3))
        self._vel = np.zeros((self._N, 3))
        self._bonded = np.zeros((self._N, self._N), dtype=bool)

        for idx, (mid, mod) in enumerate(active_modules):
            self._mid_to_idx[mid] = idx
            self._idx_to_mid[idx] = mid
            self._pos[idx] = mod.position

        # Build bond matrix from graph edges
        for (a, b) in self.system.edges:
            if a in self._mid_to_idx and b in self._mid_to_idx:
                ia, ib = self._mid_to_idx[a], self._mid_to_idx[b]
                self._bonded[ia, ib] = True
                self._bonded[ib, ia] = True

        # Initialize trajectories
        self._trajectories = {
            mid: [self._pos[idx].copy()]
            for mid, idx in self._mid_to_idx.items()
        }

        logger.info("Synced {} modules from graph to Modelica bridge",
                     self._N)

    def _append_trajectories(self, result: SimResult):
        """Append simulation trajectories to accumulated data."""
        for idx, mid in self._idx_to_mid.items():
            if idx in result.trajectories:
                # Skip the first point (same as previous final state)
                traj = result.trajectories[idx]
                if len(traj) > 1:
                    self._trajectories[mid].extend(traj[1:])

    def _disconnect_pivot_module(self, pivot_idx: int, keep_idx: Optional[int] = None):
        """Disconnect pivot module from all neighbors except keep_idx."""
        for j in range(self._N):
            if j == pivot_idx:
                continue
            if keep_idx is not None and j == keep_idx:
                continue
            self._bonded[pivot_idx, j] = False
            self._bonded[j, pivot_idx] = False

    def _reconnect_by_proximity(self, module_idx: int, threshold: float = 1.15
                                ) -> Dict[str, float]:
        """Form bonds between module_idx and any module within threshold distance.

        Returns dict of {neighbor_module_id: distance} for all neighbors checked.
        """
        distances: Dict[str, float] = {}
        for j in range(self._N):
            if j == module_idx:
                continue
            dist = float(np.linalg.norm(self._pos[module_idx] - self._pos[j]))
            if dist < threshold:
                self._bonded[module_idx, j] = True
                self._bonded[j, module_idx] = True
            # Record distance to immediate neighbors (within 2x threshold)
            if dist < threshold * 2:
                mid_j = self._idx_to_mid.get(j, f"idx_{j}")
                distances[mid_j] = dist
        return distances

    def _execute_step(self, step: Union[PivotStep, RestorationStep],
                      duration: float = 8.0) -> Optional[SimResult]:
        """Execute a single pivot step via Modelica."""
        module_id = step.module_id
        target_pos = np.asarray(step.to_pos, dtype=float)

        if module_id not in self._mid_to_idx:
            logger.warning("Module {} not in simulator, skipping", module_id)
            return None

        pivot_idx = self._mid_to_idx[module_id]
        com_before = np.mean(self._pos, axis=0).copy()
        seg_diags: List[SegmentDiagnostics] = []

        if step.pivot_type == "corner":
            axis_id = step.param1
            if axis_id not in self._mid_to_idx:
                logger.warning("Axis module {} not in simulator", axis_id)
                return None
            axis_idx = self._mid_to_idx[axis_id]

            # Algorithm rule: corner pivot disconnects from all except axis
            self._disconnect_pivot_module(pivot_idx, keep_idx=axis_idx)

            result = self.sim.simulate_pivot(
                N=self._N, pos0=self._pos, vel0=self._vel,
                bonded=self._bonded, pivot_module=pivot_idx,
                axis_module=axis_idx, target_pos=target_pos,
                duration=duration,
            )
            if result.diagnostics is not None:
                seg_diags.append(result.diagnostics)

        elif step.pivot_type == "lateral":
            old_neighbor = step.param1
            new_neighbor = step.param2
            if (old_neighbor not in self._mid_to_idx or
                    new_neighbor not in self._mid_to_idx):
                logger.warning("Neighbor {} or {} not in simulator",
                               old_neighbor, new_neighbor)
                return None

            old_idx = self._mid_to_idx[old_neighbor]
            new_idx = self._mid_to_idx[new_neighbor]

            # Disconnect from all EXCEPT old_neighbor (stay tethered for arc 1)
            self._disconnect_pivot_module(pivot_idx, keep_idx=old_idx)

            # Compute midpoint and arc angles
            old_pos = self._pos[old_idx]
            new_pos = self._pos[new_idx]
            midpoint = self.sim._compute_lateral_midpoint(
                self._pos[pivot_idx], old_pos, new_pos, target_pos)

            r_vec1 = self._pos[pivot_idx] - old_pos
            r_mid = midpoint - old_pos
            angle1 = np.arccos(np.clip(
                np.dot(r_vec1, r_mid) / (np.linalg.norm(r_vec1) * np.linalg.norm(r_mid) + 1e-12),
                -1, 1))

            # Arc 1: pivot around old_neighbor toward midpoint
            half_dur = duration / 2.0
            result = self.sim.simulate_pivot(
                N=self._N, pos0=self._pos, vel0=self._vel,
                bonded=self._bonded, pivot_module=pivot_idx,
                axis_module=old_idx, target_pos=midpoint,
                duration=half_dur, angle=angle1,
            )
            if result.diagnostics is not None:
                seg_diags.append(result.diagnostics)
            self._pos = result.final_pos
            self._vel = result.final_vel
            self._append_trajectories(result)

            # Bond swap: connect to new_neighbor, then disconnect from old_neighbor
            self._bonded[pivot_idx, new_idx] = True
            self._bonded[new_idx, pivot_idx] = True
            self._bonded[pivot_idx, old_idx] = False
            self._bonded[old_idx, pivot_idx] = False

            # Compute arc 2 angle
            r_vec2 = self._pos[pivot_idx] - new_pos
            r_tgt = target_pos - new_pos
            angle2 = np.arccos(np.clip(
                np.dot(r_vec2, r_tgt) / (np.linalg.norm(r_vec2) * np.linalg.norm(r_tgt) + 1e-12),
                -1, 1))

            # Arc 2: pivot around new_neighbor toward target
            result = self.sim.simulate_pivot(
                N=self._N, pos0=self._pos, vel0=self._vel,
                bonded=self._bonded, pivot_module=pivot_idx,
                axis_module=new_idx, target_pos=target_pos,
                duration=half_dur, angle=angle2,
            )
            if result.diagnostics is not None:
                seg_diags.append(result.diagnostics)
        else:
            logger.warning("Unknown pivot type: {}", step.pivot_type)
            return None

        # Update state from simulation
        self._pos = result.final_pos
        self._vel = result.final_vel

        # Clean up solver residual velocities
        vel_residual = self._rigidify_velocities()

        # After pivot, reconnect pivot module to any adjacent modules
        reconn_dists = self._reconnect_by_proximity(pivot_idx)
        self._append_trajectories(result)

        # Compute per-module position error against graph-expected positions
        graph_positions = self.system.get_all_positions()
        step_pos_errors: Dict[str, float] = {}
        for mid in graph_positions:
            if mid in self._mid_to_idx:
                idx = self._mid_to_idx[mid]
                step_pos_errors[mid] = float(np.linalg.norm(
                    graph_positions[mid] - self._pos[idx]))

        com_after = np.mean(self._pos, axis=0)
        com_drift = float(np.linalg.norm(com_after - com_before))

        # Record step diagnostics
        self._step_counter += 1
        self._step_diagnostics.append(StepDiagnostics(
            step_index=self._step_counter,
            step_type=step.pivot_type,
            module_id=module_id,
            segment_diagnostics=seg_diags,
            position_errors=step_pos_errors,
            velocity_residual_before_rigidify=vel_residual,
            reconnection_distances=reconn_dists,
            com_drift_this_step=com_drift,
        ))

        return result

    def _rigidify_velocities(self) -> float:
        """Project velocities onto rigid-body translation (COM velocity).

        After a pivot simulation the rigid cluster and pivot module may have
        slightly different velocities due to solver residuals.  Projecting
        everything onto the COM velocity ensures clean hand-off to the next
        segment (or analytical settle).

        Returns the max velocity residual (before projection).
        """
        vel_com = np.mean(self._vel, axis=0)
        max_residual = float(max(
            np.linalg.norm(self._vel[i] - vel_com) for i in range(self._N)
        )) if self._N > 0 else 0.0
        for i in range(self._N):
            self._vel[i] = vel_com
        return max_residual

    def _execute_settle(self, duration: float = 1.0):
        """Analytical coast — rigid structure drifts at COM velocity.

        With rigid MultiBody joints there are no spring oscillations to damp,
        so we propagate positions analytically instead of running Modelica.
        """
        self._rigidify_velocities()
        vel_com = self._vel[0].copy()  # all equal after rigidify
        self._pos += vel_com * duration
        # Append a single trajectory point for each module
        for idx, mid in self._idx_to_mid.items():
            self._trajectories[mid].append(self._pos[idx].copy())

    def replay_steps(
        self,
        steps: List[Union[PivotStep, RestorationStep]],
        parallel_steps: Optional[List[List[Union[PivotStep, RestorationStep]]]] = None,
        pivot_duration: float = 8.0,
        settle_duration: float = 1.0,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Physically execute recorded algorithm steps.

        For now, parallel steps are executed sequentially (each pivot gets
        its own Modelica simulation). Concurrent execution can be added later
        by extending the parameter arrays.
        """
        if parallel_steps is not None:
            total_groups = len(parallel_steps)
            for group_idx, group in enumerate(parallel_steps):
                if not group:
                    continue
                logger.info("Executing group {}/{} ({} pivots)",
                            group_idx + 1, total_groups, len(group))
                for step in group:
                    self._execute_step(step, duration=pivot_duration)
                self._execute_settle(settle_duration)
        else:
            for i, step in enumerate(steps):
                logger.info("Executing step {}/{}: {} {} via {}",
                            i + 1, len(steps), step.pivot_type,
                            step.module_id, step.param1)
                self._execute_step(step, duration=pivot_duration)
                self._execute_settle(settle_duration)

        return dict(self._trajectories)

    def run_damage_response(
        self,
        fault_id: str,
        pivot_duration: float = 8.0,
        settle_duration: float = 1.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run the full graph algorithm, then replay its steps physically.

        Returns dict with:
            - "algorithm_result": full result from full_damage_response()
            - "trajectories": {module_id: [position history]}
            - "position_errors": {module_id: distance graph vs physics}
            - "com_drift": displacement of system center of mass
            - "structure_rotation_deg": approximate rotation of the structure
        """
        # Reset diagnostics for this run
        self._step_diagnostics = []
        self._step_counter = 0

        # Snapshot pre-damage state (BEFORE algorithm mutates edges)
        pre_positions = self.system.get_all_positions()
        pre_edges = list(self.system.edges.keys())

        # Run the graph algorithm (this mutates system.edges!)
        kwargs.setdefault("record_steps", True)
        result = self.system.full_damage_response(fault_id, **kwargs)

        # Sync pre-damage state to physics (exclude faulty module)
        self._mid_to_idx = {}
        self._idx_to_mid = {}
        active_mids = [mid for mid in pre_positions
                       if mid != fault_id]
        self._N = len(active_mids)
        self._pos = np.zeros((self._N, 3))
        self._vel = np.zeros((self._N, 3))
        self._bonded = np.zeros((self._N, self._N), dtype=bool)

        for idx, mid in enumerate(active_mids):
            self._mid_to_idx[mid] = idx
            self._idx_to_mid[idx] = mid
            self._pos[idx] = pre_positions[mid]

        # Build pre-damage bonds from saved edges (exclude faulty module)
        for (a, b) in pre_edges:
            if a in self._mid_to_idx and b in self._mid_to_idx:
                ia, ib = self._mid_to_idx[a], self._mid_to_idx[b]
                self._bonded[ia, ib] = True
                self._bonded[ib, ia] = True

        # Record initial COM
        init_com = np.mean(self._pos, axis=0)

        # Initialize trajectories
        self._trajectories = {
            mid: [self._pos[self._mid_to_idx[mid]].copy()]
            for mid in self._mid_to_idx
        }

        # Collect steps from both phases
        all_steps = []
        all_parallel = []

        for phase_key in ("phase1", "phase2"):
            phase = result.get(phase_key, {})
            if phase:
                all_steps.extend(phase.get("steps", []))
                all_parallel.extend(phase.get("parallel_steps", []))

        # Replay physically
        if all_parallel:
            self.replay_steps(
                steps=all_steps, parallel_steps=all_parallel,
                pivot_duration=pivot_duration,
                settle_duration=settle_duration,
            )
        else:
            self.replay_steps(
                steps=all_steps,
                pivot_duration=pivot_duration,
                settle_duration=settle_duration,
            )

        # Compare final positions
        graph_positions = self.system.get_all_positions()
        position_errors = {}
        for mid in graph_positions:
            if mid in self._mid_to_idx and mid != fault_id:
                mod = self.system.modules[mid]
                if mod.is_active and not mod.is_faulty:
                    idx = self._mid_to_idx[mid]
                    position_errors[mid] = np.linalg.norm(
                        graph_positions[mid] - self._pos[idx])

        # COM drift
        final_com = np.mean(self._pos, axis=0)
        com_drift = np.linalg.norm(final_com - init_com)

        # Structure rotation estimate
        structure_rotation = 0.0
        if self._N >= 2:
            ref_mid = active_mids[0]
            ref_idx = self._mid_to_idx[ref_mid]
            v_init = pre_positions[ref_mid] - init_com
            v_final = self._pos[ref_idx] - final_com
            n_init = np.linalg.norm(v_init)
            n_final = np.linalg.norm(v_final)
            if n_init > 1e-6 and n_final > 1e-6:
                cos_a = np.clip(
                    np.dot(v_init, v_final) / (n_init * n_final), -1, 1)
                structure_rotation = np.degrees(np.arccos(cos_a))

        mean_err = (np.mean(list(position_errors.values()))
                    if position_errors else 0.0)
        logger.info(
            "Damage response complete. Mean error: {:.4f}, COM drift: {:.4f}, "
            "structure rotation: {:.2f} deg",
            mean_err, com_drift, structure_rotation,
        )

        diagnostic_report = self._build_report()

        return {
            "algorithm_result": result,
            "trajectories": dict(self._trajectories),
            "position_errors": position_errors,
            "com_drift": com_drift,
            "structure_rotation_deg": structure_rotation,
            "step_diagnostics": self._step_diagnostics,
            "diagnostic_report": diagnostic_report,
        }

    def _build_report(self) -> Dict[str, Any]:
        """Build aggregate diagnostic report from accumulated step diagnostics."""
        report: Dict[str, Any] = {
            "n_steps": len(self._step_diagnostics),
            "per_step": [],
            "summary": {},
            "trajectories": {},
        }

        all_tracking = []
        all_saturation = []
        all_pos_err = []
        all_vel_residual = []
        all_com_drift = []

        for sd in self._step_diagnostics:
            step_row: Dict[str, Any] = {
                "step": sd.step_index,
                "type": sd.step_type,
                "module": sd.module_id,
                "n_segments": len(sd.segment_diagnostics),
                "final_tracking_errors": [
                    seg.final_tracking_error for seg in sd.segment_diagnostics
                ],
                "torque_saturation_pcts": [
                    seg.torque_saturation_pct for seg in sd.segment_diagnostics
                ],
                "final_omega_residuals": [
                    seg.final_omega_residual for seg in sd.segment_diagnostics
                ],
                "final_kinetic_energies": [
                    seg.final_kinetic_energy for seg in sd.segment_diagnostics
                ],
                "max_position_error": (max(sd.position_errors.values())
                                       if sd.position_errors else 0.0),
                "mean_position_error": (float(np.mean(list(sd.position_errors.values())))
                                        if sd.position_errors else 0.0),
                "velocity_residual": sd.velocity_residual_before_rigidify,
                "com_drift": sd.com_drift_this_step,
                "reconnection_distances": sd.reconnection_distances,
            }

            # Include per-segment controller time series
            step_row["segments"] = []
            for seg in sd.segment_diagnostics:
                step_row["segments"].append({
                    "time_points": seg.time_points.tolist(),
                    "theta_d": seg.theta_d_series.tolist(),
                    "phi": seg.phi_series.tolist(),
                    "omega_d": seg.omega_d_series.tolist(),
                    "omega": seg.omega_series.tolist(),
                    "tau_raw": seg.tau_raw_series.tolist(),
                    "target_angle": seg.target_angle,
                    "kp": seg.kp,
                    "kd": seg.kd,
                    "duration": seg.duration,
                })

            report["per_step"].append(step_row)

            for seg in sd.segment_diagnostics:
                all_tracking.append(seg.final_tracking_error)
                all_saturation.append(seg.torque_saturation_pct)
            if sd.position_errors:
                all_pos_err.extend(sd.position_errors.values())
            all_vel_residual.append(sd.velocity_residual_before_rigidify)
            all_com_drift.append(sd.com_drift_this_step)

        report["summary"] = {
            "max_tracking_error_rad": max(all_tracking) if all_tracking else 0.0,
            "mean_tracking_error_rad": float(np.mean(all_tracking)) if all_tracking else 0.0,
            "max_torque_saturation_pct": max(all_saturation) if all_saturation else 0.0,
            "mean_torque_saturation_pct": float(np.mean(all_saturation)) if all_saturation else 0.0,
            "max_position_error": max(all_pos_err) if all_pos_err else 0.0,
            "mean_position_error": float(np.mean(all_pos_err)) if all_pos_err else 0.0,
            "max_velocity_residual": max(all_vel_residual) if all_vel_residual else 0.0,
            "max_com_drift_per_step": max(all_com_drift) if all_com_drift else 0.0,
        }

        # Include full module coordinate time series
        for mid, traj in self._trajectories.items():
            report["trajectories"][mid] = [pos.tolist() for pos in traj]

        return report

    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get current physical positions of all modules."""
        return {mid: self._pos[idx].copy()
                for mid, idx in self._mid_to_idx.items()}


def print_diagnostic_summary(report: Dict[str, Any]) -> None:
    """Print a formatted diagnostic summary to console."""
    s = report.get("summary", {})
    print("\n=== MODELICA DIAGNOSTICS SUMMARY ===")
    print(f"Steps executed: {report.get('n_steps', 0)}")
    print(f"Tracking error (rad):  mean={s.get('mean_tracking_error_rad', 0):.6f}"
          f"  max={s.get('max_tracking_error_rad', 0):.6f}")
    print(f"Torque saturation (%): mean={s.get('mean_torque_saturation_pct', 0):.1f}"
          f"  max={s.get('max_torque_saturation_pct', 0):.1f}")
    print(f"Position error:        mean={s.get('mean_position_error', 0):.4f}"
          f"  max={s.get('max_position_error', 0):.4f}")
    print(f"Velocity residual max: {s.get('max_velocity_residual', 0):.6f}")
    print(f"COM drift max/step:    {s.get('max_com_drift_per_step', 0):.4f}")

    per_step = report.get("per_step", [])
    if per_step:
        print()
        print(f"{'Step':>4} {'Type':>7} {'Module':>8} {'TrackErr':>10} "
              f"{'Sat%':>6} {'PosErr':>8} {'VelRes':>8} {'COMdr':>7}")
        for row in per_step:
            te = (max(row["final_tracking_errors"])
                  if row.get("final_tracking_errors") else 0)
            sat = (max(row["torque_saturation_pcts"])
                   if row.get("torque_saturation_pcts") else 0)
            print(f"{row['step']:>4} {row['type']:>7} {row['module']:>8} "
                  f"{te:>10.6f} {sat:>6.1f} {row['max_position_error']:>8.4f} "
                  f"{row['velocity_residual']:>8.6f} {row['com_drift']:>7.4f}")

    # Flag red flags
    flags = []
    if s.get('max_tracking_error_rad', 0) > 0.01:
        flags.append(f"HIGH tracking error: {s['max_tracking_error_rad']:.4f} rad (> 0.01)")
    if s.get('max_torque_saturation_pct', 0) > 50:
        flags.append(f"HIGH torque saturation: {s['max_torque_saturation_pct']:.1f}% (> 50%)")
    if s.get('max_position_error', 0) > 0.05:
        flags.append(f"HIGH position error: {s['max_position_error']:.4f} (> 0.05)")
    if s.get('max_com_drift_per_step', 0) > 0.1:
        flags.append(f"HIGH COM drift: {s['max_com_drift_per_step']:.4f} (> 0.1/step)")
    if flags:
        print("\n*** RED FLAGS ***")
        for f in flags:
            print(f"  - {f}")
    else:
        print("\nAll metrics within normal range.")
    print()
