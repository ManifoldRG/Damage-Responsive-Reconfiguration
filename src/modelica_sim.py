"""
OpenModelica-based physical simulation of modular spacecraft reconfiguration.

Dynamically generates Modelica.Mechanics.MultiBody models for each simulation
segment. Bonded modules are connected by rigid FixedTranslation rods along a
BFS spanning tree. The active pivot uses a Revolute joint with PD-controlled
torque. The whole structure floats freely via a FreeMotion joint (zero gravity).
"""

import os
import hashlib
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

# OpenModelica paths
_OPENMODELICAHOME = os.environ.get(
    "OPENMODELICAHOME",
    r"C:\Program Files\OpenModelica1.26.3-64bit"
)


@dataclass
class SegmentDiagnostics:
    """Diagnostics from a single Modelica pivot simulation segment."""
    time_points: np.ndarray          # (K,) sampled time values
    theta_d_series: np.ndarray       # (K,) desired angle
    phi_series: np.ndarray           # (K,) actual joint angle
    omega_d_series: np.ndarray       # (K,) desired angular velocity
    omega_series: np.ndarray         # (K,) actual angular velocity
    tau_raw_series: np.ndarray       # (K,) raw torque command
    final_tracking_error: float      # |theta_d(end) - phi(end)|
    final_omega_residual: float      # |w(end)|
    max_tracking_error: float        # max |theta_d(t) - phi(t)|
    torque_saturation_pct: float     # % of samples where |tau| >= max_torque
    final_kinetic_energy: float      # 0.5 * sum(m * ||v_i||^2)
    target_angle: float
    kp: float
    kd: float
    duration: float


@dataclass
class SimResult:
    """Result of a Modelica simulation run."""
    final_pos: np.ndarray       # (N, 3) final positions
    final_vel: np.ndarray       # (N, 3) final velocities
    final_bonded: np.ndarray    # (N, N) bool final bond matrix
    trajectories: Dict[int, List[np.ndarray]]  # module_idx -> list of positions
    duration: float
    diagnostics: Optional[SegmentDiagnostics] = None


class ModelicaSimulator:
    """
    Wrapper around OpenModelica for simulating modular spacecraft physics.

    Generates Modelica.Mechanics.MultiBody models with rigid joints on the fly.
    Modules are 1 kg spheres (radius 0.5 m) in zero gravity. Bonds are rigid
    FixedTranslation rods. Pivots use a Revolute joint with PD-controlled torque
    (saturated at 0.2 Nm). The whole assembly floats freely (FreeMotion joint).
    """

    MODULE_RADIUS = 0.5
    MODULE_MASS = 1.0
    MAX_MOTOR_TORQUE = 0.2
    # Sphere inertia: (2/5) * m * R^2
    SPHERE_INERTIA = (2.0 / 5.0) * MODULE_MASS * MODULE_RADIUS ** 2  # 0.1

    def __init__(self, work_dir: Optional[str] = None):
        os.environ["OPENMODELICAHOME"] = _OPENMODELICAHOME

        if work_dir is None:
            work_dir = str(Path(__file__).parent.parent / "modelica_work")
        self._work_dir = Path(work_dir)
        self._work_dir.mkdir(parents=True, exist_ok=True)

        self._omc = None
        self._loaded_topos: set = set()  # topology hashes already loaded

    def _ensure_omc(self):
        """Lazily connect to OpenModelica."""
        if self._omc is not None:
            return
        try:
            from OMPython import OMCSessionZMQ
        except ImportError:
            raise ImportError(
                "OMPython is required. Install with: pip install OMPython"
            )
        self._omc = OMCSessionZMQ()
        self._omc.sendExpression(f'cd("{self._work_dir.as_posix()}")')
        # Load the Modelica Standard Library (needed for MultiBody)
        self._omc.sendExpression('loadModel(Modelica)')
        logger.info("OpenModelica session started (work_dir={})", self._work_dir)

    # ── Geometry helpers ──────────────────────────────────────────────

    def _compute_pd_gains(self, r_vec: np.ndarray, duration: float
                          ) -> Tuple[float, float]:
        """
        PD gains for a revolute pivot.

        I_eff = 3 * (m*r^2 + (2/5)*m*R^2) -- conservative 3x multiplier
        omega_n = 3 * 2*pi / duration -- 3 cycles per duration
        kp = I_eff * omega_n^2, kd = 2 * I_eff * omega_n (critical damping)
        """
        r = np.linalg.norm(r_vec)
        I_orbit = self.MODULE_MASS * r ** 2
        I_spin = self.SPHERE_INERTIA
        I_eff = 3.0 * (I_orbit + I_spin)
        omega_n = 3.0 * 2.0 * np.pi / duration
        kp = I_eff * omega_n ** 2
        kd = 2.0 * I_eff * omega_n
        return kp, kd

    def _get_rotation_axis(self, pivot_pos: np.ndarray, axis_pos: np.ndarray,
                           target_pos: np.ndarray) -> np.ndarray:
        """Rotation axis for a pivot (perpendicular to plane of motion).

        Falls back through multiple strategies to always return a valid
        unit vector, even when inputs are nearly degenerate.
        """
        v1 = pivot_pos - axis_pos
        v2 = target_pos - axis_pos

        # Primary: cross product of the two radii
        axis = np.cross(v1, v2)
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            return axis / norm

        # Fallback 1: v1 × arbitrary perpendicular
        ref = v1 if np.linalg.norm(v1) > 1e-8 else v2
        if np.linalg.norm(ref) > 1e-8:
            perp = np.array([1, 0, 0]) if abs(ref[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(ref, perp)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                return axis / norm

        # Fallback 2: default to Z axis
        return np.array([0.0, 0.0, 1.0])

    def _compute_lateral_midpoint(self, pivot_pos, old_pos, new_pos, target_pos):
        """Compute midpoint for a lateral pivot (equidistant from both neighbors)."""
        mid_dir = (pivot_pos - old_pos) + (target_pos - new_pos)
        mid_dir_norm = np.linalg.norm(mid_dir)
        if mid_dir_norm < 1e-8:
            v = new_pos - old_pos
            perp = np.array([1, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1, 0])
            mid_dir = np.cross(v, perp)
            mid_dir_norm = np.linalg.norm(mid_dir)
        mid_dir_unit = mid_dir / mid_dir_norm
        center_nb = (old_pos + new_pos) / 2.0
        half_dist = np.linalg.norm(new_pos - old_pos) / 2.0
        h = np.sqrt(max(1.0 - half_dist ** 2, 0.0))
        return center_nb + mid_dir_unit * h

    # ── Spanning tree ─────────────────────────────────────────────────

    @staticmethod
    def _build_spanning_forest(
        N: int, bonded: np.ndarray, root: int
    ) -> Tuple[Dict[int, int], List[Tuple[int, int]], List[int]]:
        """
        BFS spanning forest of the bonded graph.

        Starts from *root*, then covers any remaining connected components
        so that ALL bonded edges are represented by rigid connections.

        Returns:
            parent_map: {child_idx: parent_idx}  (component roots have no entry)
            tree_edges: [(parent, child), ...] in BFS order
            component_roots: list of root indices for each connected component
                             (first element is always *root*)
        """
        visited: set = set()
        parent_map: Dict[int, int] = {}
        tree_edges: List[Tuple[int, int]] = []
        component_roots: List[int] = []

        def _bfs(start: int):
            visited.add(start)
            component_roots.append(start)
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nbr in range(N):
                    if nbr not in visited and bonded[node, nbr]:
                        visited.add(nbr)
                        parent_map[nbr] = node
                        tree_edges.append((node, nbr))
                        queue.append(nbr)

        # Primary component (contains the axis module)
        _bfs(root)

        # Cover remaining connected components
        for i in range(N):
            if i not in visited:
                _bfs(i)

        return parent_map, tree_edges, component_roots

    # ── Dynamic model generation ──────────────────────────────────────

    @staticmethod
    def _topology_hash(
        N: int, bonded: np.ndarray,
        pivot_module: Optional[int], axis_module: Optional[int],
    ) -> str:
        """Stable hash of the topology (ignores positions — those are overrides)."""
        blob = bonded.tobytes() + bytes([
            N,
            pivot_module if pivot_module is not None else 255,
            axis_module if axis_module is not None else 255,
        ])
        return hashlib.sha256(blob).hexdigest()[:12]

    def _generate_multibody_model(
        self,
        N: int,
        pos0: np.ndarray,
        vel0: np.ndarray,
        bonded: np.ndarray,
        pivot_module: Optional[int],
        axis_module: Optional[int],
        rot_axis: Optional[np.ndarray],
        target_angle: Optional[float],
        kp: Optional[float],
        kd: Optional[float],
        duration: float,
    ) -> Tuple[str, str]:
        """
        Generate a complete Modelica model string using MultiBody components.

        Returns:
            (model_name, model_string)
        """
        has_pivot = pivot_module is not None and axis_module is not None
        root = axis_module if has_pivot else 0

        parent_map, tree_edges, component_roots = self._build_spanning_forest(
            N, bonded, root)

        topo_hash = self._topology_hash(N, bonded, pivot_module, axis_module)
        model_name = f"Seg_{topo_hash}"

        I = self.SPHERE_INERTIA  # 0.1

        # ── Format helpers ────────────────────────────────────────────
        def v3(arr):
            """Format a 3-vector as Modelica literal."""
            return f"{{{arr[0]:.10g}, {arr[1]:.10g}, {arr[2]:.10g}}}"

        # ── Component declarations ────────────────────────────────────
        decls = []
        decls.append("  inner Modelica.Mechanics.MultiBody.World world("
                      "gravityType=Modelica.Mechanics.MultiBody.Types."
                      "GravityTypes.NoGravity);")
        decls.append("")

        # FreeMotion joint for each connected component root
        for cr in component_roots:
            cr_pos = pos0[cr]
            cr_vel = vel0[cr]
            suffix = "" if cr == root else f"_{cr}"
            decls.append(
                f"  Modelica.Mechanics.MultiBody.Joints.FreeMotion "
                f"freeJoint{suffix}("
                f"r_rel_a(start={v3(cr_pos)}, each fixed=true), "
                f"v_rel_a(start={v3(cr_vel)}, each fixed=true), "
                f"angles_fixed=true, angles_start={{0,0,0}}, "
                f"w_rel_a_fixed=true, w_rel_a_start={{0,0,0}});")
        decls.append("")

        # Body for each module
        for i in range(N):
            decls.append(
                f"  Modelica.Mechanics.MultiBody.Parts.Body body_{i}("
                f"m={self.MODULE_MASS}, "
                f"I_11={I}, I_22={I}, I_33={I}, "
                f"I_21=0, I_31=0, I_32=0, "
                f"r_CM={{0,0,0}}, "
                f"sphereDiameter={2*self.MODULE_RADIUS});")
        decls.append("")

        # FixedTranslation for each spanning-tree edge (except pivot edge)
        for (parent, child) in tree_edges:
            if has_pivot and child == pivot_module and parent == axis_module:
                continue  # this edge is the Revolute joint
            if has_pivot and child == axis_module and parent == pivot_module:
                continue  # shouldn't happen (root=axis), but guard
            r = pos0[child] - pos0[parent]
            decls.append(
                f"  Modelica.Mechanics.MultiBody.Parts.FixedTranslation "
                f"rod_{parent}_{child}(r={v3(r)});")
        decls.append("")

        # Revolute + torque for pivot
        if has_pivot:
            decls.append(
                f"  Modelica.Mechanics.MultiBody.Joints.Revolute pivotJoint("
                f"n={v3(rot_axis)}, "
                f"useAxisFlange=true, "
                f"phi(fixed=true, start=0), "
                f"w(fixed=true, start=0));")
            decls.append(
                f"  Modelica.Mechanics.Rotational.Sources.Torque torqueSrc;")
            decls.append("")
            decls.append(f"  parameter Real kp = {kp};")
            decls.append(f"  parameter Real kd = {kd};")
            decls.append(f"  parameter Real target_angle = {target_angle};")
            decls.append(f"  parameter Real pivot_duration = {duration};")
            decls.append(f"  parameter Real max_torque = {self.MAX_MOTOR_TORQUE};")
            decls.append(f"  Real s_frac;")
            decls.append(f"  Real s_val;")
            decls.append(f"  Real s_deriv;")
            decls.append(f"  Real theta_d;")
            decls.append(f"  Real omega_d;")
            decls.append(f"  Real tau_raw;")
        decls.append("")

        # ── Equations (connections) ───────────────────────────────────
        eqns = []

        # Connect each component root to World via its FreeMotion joint
        for cr in component_roots:
            suffix = "" if cr == root else f"_{cr}"
            eqns.append(
                f"  connect(world.frame_b, freeJoint{suffix}.frame_a);")
            eqns.append(
                f"  connect(freeJoint{suffix}.frame_b, "
                f"body_{cr}.frame_a);")
        eqns.append("")

        # Spanning-forest rigid connections (+ revolute for pivot edge)
        for (parent, child) in tree_edges:
            if has_pivot and child == pivot_module and parent == axis_module:
                # Revolute joint instead of rigid rod
                eqns.append(
                    f"  connect(body_{axis_module}.frame_a, "
                    f"pivotJoint.frame_a);")
                eqns.append(
                    f"  connect(pivotJoint.frame_b, "
                    f"body_{pivot_module}.frame_a);")
                continue
            eqns.append(
                f"  connect(body_{parent}.frame_a, "
                f"rod_{parent}_{child}.frame_a);")
            eqns.append(
                f"  connect(rod_{parent}_{child}.frame_b, "
                f"body_{child}.frame_a);")
        eqns.append("")

        # Torque source connections + PD controller
        if has_pivot:
            eqns.append("  connect(torqueSrc.flange, pivotJoint.axis);")
            eqns.append("")
            # Smoothstep PD controller
            eqns.append("  // Smoothstep reference trajectory")
            eqns.append("  s_frac = min(max(time / pivot_duration, 0.0), 1.0);")
            eqns.append("  s_val = s_frac * s_frac * (3.0 - 2.0 * s_frac);")
            eqns.append("  s_deriv = 6.0 * s_frac * (1.0 - s_frac);")
            eqns.append("  theta_d = target_angle * s_val;")
            eqns.append("  omega_d = target_angle * s_deriv "
                         "/ max(pivot_duration, 1e-6);")
            eqns.append("")
            eqns.append("  // PD torque with saturation")
            eqns.append("  tau_raw = kp * (theta_d - pivotJoint.phi) "
                         "+ kd * (omega_d - pivotJoint.w);")
            eqns.append("  torqueSrc.tau = tau_raw;")

        # ── Assemble model string ─────────────────────────────────────
        lines = [f"model {model_name}"]
        lines.extend(decls)
        lines.append("equation")
        lines.extend(eqns)
        lines.append(f"  annotation(experiment("
                      f"StartTime=0, StopTime={duration}, "
                      f"Tolerance=1e-6));")
        lines.append(f"end {model_name};")

        model_string = "\n".join(lines)
        return model_name, model_string

    # ── Simulation ────────────────────────────────────────────────────

    def simulate(
        self,
        N: int,
        pos0: np.ndarray,
        vel0: np.ndarray,
        bonded: np.ndarray,
        duration: float,
        pivot_module: Optional[int] = None,
        axis_module: Optional[int] = None,
        rot_axis: Optional[np.ndarray] = None,
        target_angle: Optional[float] = None,
        kp: Optional[float] = None,
        kd: Optional[float] = None,
        n_intervals: int = 500,
    ) -> SimResult:
        """
        Run a single simulation segment using a dynamically generated
        MultiBody model.

        Args:
            N: Number of modules
            pos0: (N, 3) initial positions
            vel0: (N, 3) initial velocities
            bonded: (N, N) bool bond matrix
            duration: Simulation duration in seconds
            pivot_module: Index of pivoting module (None for settle)
            axis_module: Index of axis module (None for settle)
            rot_axis: Rotation axis unit vector
            target_angle: Target angle in radians
            kp, kd: PD gains
            n_intervals: Number of output time points

        Returns:
            SimResult with final state and trajectories
        """
        self._ensure_omc()

        model_name, model_string = self._generate_multibody_model(
            N, pos0, vel0, bonded,
            pivot_module, axis_module, rot_axis,
            target_angle, kp, kd, duration,
        )

        # Write model to file for debugging and load it
        model_file = self._work_dir / f"{model_name}.mo"
        model_file.write_text(model_string)

        # Load (or reload) the model
        topo_hash = self._topology_hash(N, bonded, pivot_module, axis_module)
        load_ok = self._omc.sendExpression(
            f'loadFile("{model_file.as_posix()}")')
        if not load_ok:
            err = self._omc.sendExpression("getErrorString()")
            raise RuntimeError(
                f"Failed to load generated model {model_name}: {err}")

        # Run simulation
        expr = (
            f'simulate({model_name}, '
            f'startTime=0, stopTime={duration}, '
            f'numberOfIntervals={n_intervals})'
        )

        logger.debug("Running MultiBody simulation (N={}, pivot={}, "
                      "duration={:.1f}s)", N, pivot_module, duration)
        result = self._omc.sendExpression(expr)

        if not isinstance(result, dict) or not result.get("resultFile"):
            err = self._omc.sendExpression("getErrorString()")
            raise RuntimeError(f"Simulation failed: {err}")

        logger.debug("Simulation complete: {}", result["resultFile"])

        # ── Extract results ───────────────────────────────────────────
        def _val(expr: str) -> float:
            """Read a scalar from the result file, defaulting to 0.0."""
            v = self._omc.sendExpression(expr)
            return float(v) if v is not None else 0.0

        final_pos = np.zeros((N, 3))
        final_vel = np.zeros((N, 3))
        for i in range(N):
            for k in range(3):
                final_pos[i, k] = _val(
                    f"val(body_{i}.frame_a.r_0[{k+1}], {duration})")
                final_vel[i, k] = _val(
                    f"val(body_{i}.v_0[{k+1}], {duration})")

        # Trajectories (sampled at fewer points — val() calls are slow)
        trajectories: Dict[int, List[np.ndarray]] = {}
        n_samples = min(n_intervals, 60)
        sample_times = np.linspace(0, duration, n_samples)

        for i in range(N):
            traj = []
            for t in sample_times:
                pt = np.array([
                    _val(f"val(body_{i}.frame_a.r_0[1], {t})"),
                    _val(f"val(body_{i}.frame_a.r_0[2], {t})"),
                    _val(f"val(body_{i}.frame_a.r_0[3], {t})"),
                ])
                traj.append(pt)
            trajectories[i] = traj
        logger.debug("Read {} trajectory samples for {} modules",
                     n_samples, N)

        # Bond state is unchanged (rigid joints never break)
        final_bonded = bonded.copy()

        # ── Controller diagnostics (pivot segments only) ──────────
        diagnostics = None
        has_pivot = pivot_module is not None and axis_module is not None
        if has_pivot and target_angle is not None:
            n_diag = 30
            diag_times = np.linspace(0, duration, n_diag)

            theta_d_arr = np.array([_val(f"val(theta_d, {t})") for t in diag_times])
            phi_arr = np.array([_val(f"val(pivotJoint.phi, {t})") for t in diag_times])
            omega_d_arr = np.array([_val(f"val(omega_d, {t})") for t in diag_times])
            omega_arr = np.array([_val(f"val(pivotJoint.w, {t})") for t in diag_times])
            tau_arr = np.array([_val(f"val(tau_raw, {t})") for t in diag_times])

            tracking_err = np.abs(theta_d_arr - phi_arr)
            ke = 0.5 * self.MODULE_MASS * np.sum(final_vel ** 2)

            diagnostics = SegmentDiagnostics(
                time_points=diag_times,
                theta_d_series=theta_d_arr,
                phi_series=phi_arr,
                omega_d_series=omega_d_arr,
                omega_series=omega_arr,
                tau_raw_series=tau_arr,
                final_tracking_error=float(tracking_err[-1]),
                final_omega_residual=float(np.abs(omega_arr[-1])),
                max_tracking_error=float(np.max(tracking_err)),
                torque_saturation_pct=float(
                    100.0 * np.mean(np.abs(tau_arr) >= self.MAX_MOTOR_TORQUE * 0.99)
                ),
                final_kinetic_energy=ke,
                target_angle=target_angle,
                kp=kp,
                kd=kd,
                duration=duration,
            )
            logger.debug("Diagnostics: tracking_err={:.6f}, saturation={:.1f}%, KE={:.6f}",
                         diagnostics.final_tracking_error,
                         diagnostics.torque_saturation_pct,
                         diagnostics.final_kinetic_energy)

        return SimResult(
            final_pos=final_pos,
            final_vel=final_vel,
            final_bonded=final_bonded,
            trajectories=trajectories,
            duration=duration,
            diagnostics=diagnostics,
        )

    def simulate_pivot(
        self,
        N: int,
        pos0: np.ndarray,
        vel0: np.ndarray,
        bonded: np.ndarray,
        pivot_module: int,
        axis_module: int,
        target_pos: np.ndarray,
        duration: float = 8.0,
        angle: Optional[float] = None,
        n_intervals: int = 500,
    ) -> SimResult:
        """
        Simulate a single arc pivot (corner or one arc of a lateral).

        The bridge handles lateral pivots by calling this twice (arc 1 + arc 2)
        with bond updates in between.

        Args:
            pivot_module: Index of the pivoting module (0-based)
            axis_module: Index of the axis module (0-based)
            target_pos: Target position (used to compute rotation axis)
            duration: Pivot duration in seconds
            angle: Rotation angle in radians (default: pi/2 for corner)
        """
        pivot_pos = pos0[pivot_module]
        axis_pos = pos0[axis_module]
        r_vec = pivot_pos - axis_pos
        rot_axis = self._get_rotation_axis(pivot_pos, axis_pos, target_pos)
        kp, kd = self._compute_pd_gains(r_vec, duration)

        if angle is None:
            angle = np.pi / 2.0

        return self.simulate(
            N, pos0, vel0, bonded, duration,
            pivot_module=pivot_module,
            axis_module=axis_module,
            rot_axis=rot_axis,
            target_angle=angle,
            kp=kp, kd=kd,
            n_intervals=n_intervals,
        )

    # ── Lifecycle ─────────────────────────────────────────────────────

    def disconnect(self):
        """Clean up OpenModelica session."""
        if self._omc is not None:
            try:
                self._omc.__del__()
            except Exception:
                pass
            self._omc = None
            self._loaded_topos.clear()

    def __del__(self):
        self.disconnect()
