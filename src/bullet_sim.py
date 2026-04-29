"""
Persistent PyBullet physics world for modular spacecraft reconfiguration.

Modules are 1 kg spheres (0.5 m radius) in zero gravity.

**Default (``USE_SPRING_BONDS`` False):** ``JOINT_FIXED`` per bond,
``JOINT_POINT2POINT`` between pivot and axis while pivoting (unless
``USE_ROLLING_SPHERE_PIVOT`` is True, which uses sphere–sphere distance + no-slip
coupling instead; see ``rolling_sphere_coupling.py``), and **attachment
springs** (surface anchor pull + reaction) for pivot actuation—not tangential PD,
unless ``USE_PIVOT_PD`` is enabled. COM–COM bond forces are inactive while
``RIGID_BONDS`` is True. Optional COM-relative damping applies.

**Spring mode (``USE_SPRING_BONDS`` True):** COM–COM springs plus full-graph
attachment springs; no fixed/point-to-point joints; pivot completion uses
relative-angle criteria; bonded pairs disable mutual collision.

Set ``USE_PIVOT_PD`` True only to restore legacy tangential PD torque on pivots
(rigid mode).
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from loguru import logger

try:
    import pybullet as p
    import pybullet_data
except ImportError:
    raise ImportError("pybullet is required. Install with: pip install pybullet")

from .rolling_sphere_coupling import (
    apply_prescribed_rolling_pair,
    apply_rolling_sphere_pair,
)


@dataclass
class PivotState:
    """Active pivot tracking for one module."""
    axis_idx: int
    rot_axis: np.ndarray
    target_angle: float
    kp: float
    kd: float
    duration: float
    start_time: float
    r0: np.ndarray
    lattice_ref_body_idx: int              # assembly lattice frame; frozen for maneuver
    target_pos_local: np.ndarray           # pivot COM goal in ref body frame (offset from ref COM)
    attract_body_idx: int                  # module whose connector is the fixed attract site
    attract_connector: int
    completed: bool = False
    timed_out: bool = False
    collided: bool = False
    prev_rel_v: float = 0.0
    pivot_type: str = "corner"
    repel_idx: Optional[int] = None    # module whose surface repels; defaults to axis_idx
    repel_connector: Optional[int] = None    # connector index on repel module


class BulletSimulator:
    """PyBullet: default rigid bonds + point-to-point pivot + attachment actuation."""

    MODULE_MASS = 1.0
    MODULE_RADIUS = 0.5
    MAX_MOTOR_TORQUE = 2.0
    SPHERE_INERTIA = (2.0 / 5.0) * MODULE_MASS * MODULE_RADIUS ** 2

    # False: fixed bonds + point-to-point pivot + attachment (or PD if flag on).
    USE_SPRING_BONDS = False

    # Fixed constraints when USE_SPRING_BONDS is False.
    RIGID_BONDS = True

    # Tangential PD on pivot module+axis (rigid mode only). Default: attachment + angle attraction.
    USE_PIVOT_PD = False

    # Rigid pivot: use sphere–sphere rolling coupling instead of POINT2POINT (experimental).
    USE_ROLLING_SPHERE_PIVOT = False
    # Rolling coupling: more iterations + higher lambda cap = tighter no-slip per
    # substep (less residual slip when actuation injects tangential motion).
    ROLLING_DISTANCE_BETA = 0.35
    ROLLING_MAX_POS_CORRECTION = 0.02
    ROLLING_MAX_LAMBDA_NORM = 20000.0
    ROLLING_NO_SLIP_ITERATIONS = 16

    # Optional: kinematic arc instead of iterative rolling (experiments only).
    USE_PRESCRIBED_ROLLING_PIVOT = False

    # After iterative rolling pivot: snap pivot body quaternion to crystal frame.
    USE_CRYSTAL_ATTITUDE_SNAP_AFTER_ROLLING_PIVOT = True

    BOND_CONSTRAINT_MAX_FORCE = 1e7
    # Lower than bond constraints so pivot can yield slightly under attachment/attraction.
    PIVOT_CONSTRAINT_MAX_FORCE = 1e6

    # COM–COM spring (dominant stiffness when USE_SPRING_BONDS)
    K_BOND = 90.0
    C_BOND = 50.0
    NOMINAL_DIST = 1.0
    BOND_FORCE_CAP = 140.0        # N per body from COM-COM pair

    # Attachment spring (weaker than bond): receiver COM toward surface on source
    K_ATTACHMENT = 50.0
    C_ATTACHMENT = 5.0
    ATTACHMENT_FORCE_CAP = 60.0   # N per directed attachment

    # Cartesian PD pivot actuation: F = Kp*(target - pos) - Kd*v_rel
    PIVOT_PD_KP = 20.0    # N/m   (proportional: position error -> force)
    PIVOT_PD_KD = 10.0    # N·s/m (derivative: damps relative velocity to zero)
    PIVOT_PD_F_MAX = 500.0  # N  (force cap per pivot pair)

    # Legacy soft bonds when RIGID_BONDS False and USE_SPRING_BONDS False
    K_BOND_SOFT_LEGACY = 1000.0
    C_BOND_SOFT_LEGACY = 45.0

    DAMPING_COEFF = 2.4
    PHYSICS_DT = 0.01

    # Spring pivot settling (relative geometry — not world target_pos; cluster may drift)
    SPRING_PIVOT_ANGLE_TOL = 0.04
    SPRING_PIVOT_OMEGA_TOL = 0.14
    SPRING_PIVOT_LOOSE_TIME = 14.0
    SPRING_PIVOT_LOOSE_ANGLE = 0.06
    SPRING_PIVOT_LOOSE_OMEGA = 0.22

    # Rigid pivot completion: for rolling, also require low pivot–axis relative motion.
    PIVOT_POS_TOL = 1e-2         # m
    PIVOT_ANGLE_TOL = 0.012      # unused for rigid completion; kept for instance overrides
    PIVOT_OMEGA_TOL = 1e-3       # rad/s scale: |v_tan|/|r| for pivot–axis
    PIVOT_REL_V_TOL = 1e-2       # m/s: ||v_pivot − v_axis||
    MAX_PIVOT_TIME = 60.0        # seconds simulation time per pivot

    # 6 body-frame connector directions (cardinal axes).
    CONNECTOR_DIRS = np.array([
        [1, 0, 0], [-1, 0, 0],   # +X (0), -X (1)
        [0, 1, 0], [0, -1, 0],   # +Y (2), -Y (3)
        [0, 0, 1], [0, 0, -1],   # +Z (4), -Z (5)
    ], dtype=float)

    # Auto-bond: if any two mating connectors are within this distance and
    # neither module is actively pivoting, create a rigid bond automatically.
    AUTO_BOND_CONNECTOR_DIST = 0.1  # m

    def __init__(self, N: int, pos0: np.ndarray, bonded0: np.ndarray,
                 vel0: Optional[np.ndarray] = None, gui: bool = False,
                 *,
                 mute_collision_for_bonded_after_pivot: bool = True,
                 momentum_diagnostics: bool = False,
                 momentum_diag_subsample: int = 1,
                 momentum_diag_max_samples: int = 200_000,
                 pivot_attract_scale: float = 1.0):
        self.N = N
        self._sim_time = 0.0
        self._mute_collision_for_bonded_after_pivot = (
            mute_collision_for_bonded_after_pivot)
        ps = float(pivot_attract_scale)
        if ps <= 0.0:
            raise ValueError("pivot_attract_scale must be positive")
        self._pivot_attract_scale = ps
        self._momentum_diagnostics = bool(momentum_diagnostics)
        self._momentum_diag_subsample = max(1, int(momentum_diag_subsample))
        self._momentum_diag_max_samples = max(0, int(momentum_diag_max_samples))
        self._momentum_diag_substep_idx = 0
        self._momentum_diag_samples: List[Dict[str, Any]] = []
        self._momentum_diag_summary: Dict[str, Any] = {}

        mode = p.GUI if gui else p.DIRECT
        self._physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0, physicsClientId=self._physics_client)
        p.setTimeStep(self.PHYSICS_DT, physicsClientId=self._physics_client)

        self._sphere_shape = p.createCollisionShape(
            p.GEOM_SPHERE, radius=self.MODULE_RADIUS,
            physicsClientId=self._physics_client)

        self._body_ids: List[int] = []
        for i in range(N):
            body_id = p.createMultiBody(
                baseMass=self.MODULE_MASS,
                baseCollisionShapeIndex=self._sphere_shape,
                basePosition=pos0[i].tolist(),
                physicsClientId=self._physics_client,
            )
            if vel0 is not None:
                p.resetBaseVelocity(
                    body_id,
                    linearVelocity=vel0[i].tolist(),
                    angularVelocity=[0, 0, 0],
                    physicsClientId=self._physics_client,
                )
            p.changeDynamics(
                body_id, -1,
                restitution=0.05,
                lateralFriction=0.35,
                linearDamping=0.1,
                angularDamping=0.1,
                physicsClientId=self._physics_client,
            )
            self._body_ids.append(body_id)

        self._bonds: Set[Tuple[int, int]] = set()
        for i in range(N):
            for j in range(i + 1, N):
                if bonded0[i, j]:
                    self._bonds.add((i, j))

        self._bond_constraints: Dict[Tuple[int, int], int] = {}
        self._active_pivots: Dict[int, PivotState] = {}
        self._pivot_constraints: Dict[int, int] = {}
        self._pivot_diagnostic_log: List[Dict[str, Any]] = []
        self._attachment_suppress: Set[Tuple[int, int]] = set()
        # Indices for which ``set_body_collision_with_all(i, False)`` was used;
        # ``stop_pivot`` must not re-enable those pairs when restoring filters.
        self._collision_off_with_all: Set[int] = set()

        nit = 150 if self.USE_SPRING_BONDS else 120
        p.setPhysicsEngineParameter(
            numSolverIterations=nit,
            physicsClientId=self._physics_client,
        )

        if self.USE_SPRING_BONDS:
            for (lo, hi) in list(self._bonds):
                self._set_pair_collision(lo, hi, enable=False)
            logger.info(
                "PyBullet world: {} modules, {} bonds (spring COM+attachment)",
                N, len(self._bonds))
        elif self.RIGID_BONDS:
            for (lo, hi) in list(self._bonds):
                self._add_rigid_bond_constraint(lo, hi)
            logger.info("PyBullet world created: {} modules, {} bonds (rigid fixed)",
                        N, len(self._bonds))
        else:
            logger.info("PyBullet world created: {} modules, {} bonds (spring-damper legacy)",
                        N, len(self._bonds))

        if self._momentum_diagnostics:
            self.clear_momentum_diagnostics()

    def handoff_contact_distance(self) -> float:
        """Lateral handoff proximity: one radius in spring mode."""
        return self.MODULE_RADIUS if self.USE_SPRING_BONDS else 1.05

    def _set_pair_collision(self, i: int, j: int, enable: bool):
        p.setCollisionFilterPair(
            self._body_ids[i], self._body_ids[j], -1, -1,
            int(enable), physicsClientId=self._physics_client,
        )

    def set_body_collision_with_all(self, body_idx: int, enable: bool) -> None:
        """Enable or disable collision between *body_idx* and every other module.

        When disabled, the filter stays off after pivot completion (see
        ``stop_pivot``), not only at init time.
        """
        if not (0 <= body_idx < self.N):
            raise IndexError(f"body_idx {body_idx} out of range for N={self.N}")
        if enable:
            self._collision_off_with_all.discard(body_idx)
        else:
            self._collision_off_with_all.add(body_idx)
        bid_a = self._body_ids[body_idx]
        en = int(enable)
        for j in range(self.N):
            if j == body_idx:
                continue
            p.setCollisionFilterPair(
                bid_a, self._body_ids[j], -1, -1, en,
                physicsClientId=self._physics_client,
            )

    def set_body_mass(self, body_idx: int, mass: float) -> None:
        """Set base link mass (kg). Use a small value for low-impact ghost bodies."""
        if not (0 <= body_idx < self.N):
            raise IndexError(f"body_idx {body_idx} out of range for N={self.N}")
        p.changeDynamics(
            self._body_ids[body_idx], -1, mass=float(mass),
            physicsClientId=self._physics_client,
        )

    def reset_module_world_pose(
        self,
        body_idx: int,
        position: np.ndarray,
        orientation_xyzw: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Teleport a module: set world pose and zero velocities."""
        if not (0 <= body_idx < self.N):
            raise IndexError(f"body_idx {body_idx} out of range for N={self.N}")
        pos = np.asarray(position, dtype=float).reshape(3)
        if orientation_xyzw is None:
            orn = (0.0, 0.0, 0.0, 1.0)
        else:
            orn = tuple(float(x) for x in orientation_xyzw)
        bid = self._body_ids[body_idx]
        p.resetBasePositionAndOrientation(
            bid, pos.tolist(), list(orn),
            physicsClientId=self._physics_client,
        )
        p.resetBaseVelocity(
            bid, linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self._physics_client,
        )

    # ── Bond management ────────────────────────────────────────────────

    def create_bond(self, i: int, j: int):
        lo, hi = min(i, j), max(i, j)
        if (lo, hi) in self._bonds:
            return
        self._bonds.add((lo, hi))
        if self.USE_SPRING_BONDS:
            self._set_pair_collision(lo, hi, enable=False)
        elif self.RIGID_BONDS:
            self._add_rigid_bond_constraint(lo, hi)

    def remove_bond(self, i: int, j: int):
        lo, hi = min(i, j), max(i, j)
        self._remove_rigid_bond_constraint(lo, hi)
        if self.USE_SPRING_BONDS and (lo, hi) in self._bonds:
            pair_on = (
                lo not in self._collision_off_with_all
                and hi not in self._collision_off_with_all)
            self._set_pair_collision(lo, hi, enable=pair_on)
        self._bonds.discard((lo, hi))

    def _add_rigid_bond_constraint(self, lo: int, hi: int):
        if not self.RIGID_BONDS or self.USE_SPRING_BONDS:
            return
        if (lo, hi) in self._bond_constraints:
            logger.warning(
                "Skipping duplicate rigid bond constraint for pair ({}, {})", lo, hi)
            return
        pos_lo, orn_lo = p.getBasePositionAndOrientation(
            self._body_ids[lo], physicsClientId=self._physics_client)
        pos_hi, orn_hi = p.getBasePositionAndOrientation(
            self._body_ids[hi], physicsClientId=self._physics_client)
        inv_cpos, inv_corn = p.invertTransform(pos_hi, orn_hi)
        rel_pos, rel_orn = p.multiplyTransforms(
            inv_cpos, inv_corn, pos_lo, orn_lo)
        cid = p.createConstraint(
            self._body_ids[lo], -1,
            self._body_ids[hi], -1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0.0, 0.0, 0.0],
            parentFramePosition=[0.0, 0.0, 0.0],
            parentFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            childFramePosition=rel_pos,
            childFrameOrientation=rel_orn,
            physicsClientId=self._physics_client,
        )
        p.changeConstraint(
            cid, maxForce=self.BOND_CONSTRAINT_MAX_FORCE,
            physicsClientId=self._physics_client)
        self._bond_constraints[(lo, hi)] = cid

    def _remove_rigid_bond_constraint(self, lo: int, hi: int):
        if self.USE_SPRING_BONDS:
            return
        key = (min(lo, hi), max(lo, hi))
        cid = self._bond_constraints.pop(key, None)
        if cid is not None:
            try:
                p.removeConstraint(cid, physicsClientId=self._physics_client)
            except Exception:
                pass

    def is_bonded(self, i: int, j: int) -> bool:
        return (min(i, j), max(i, j)) in self._bonds

    def get_bond_matrix(self) -> np.ndarray:
        bonded = np.zeros((self.N, self.N), dtype=bool)
        for (i, j) in self._bonds:
            bonded[i, j] = True
            bonded[j, i] = True
        return bonded

    def _auto_proximity_bond(self):
        """Bond any two non-pivoting modules whose mating connectors are within threshold."""
        pivoting: set = set(self._active_pivots.keys())
        for ps in self._active_pivots.values():
            pivoting.add(ps.axis_idx)

        pos = self.get_positions()
        com_thresh = self.NOMINAL_DIST + self.AUTO_BOND_CONNECTOR_DIST
        for i in range(self.N):
            if i in pivoting:
                continue
            for j in range(i + 1, self.N):
                if j in pivoting:
                    continue
                if (i, j) in self._bonds:
                    continue
                if float(np.linalg.norm(pos[i] - pos[j])) > com_thresh:
                    continue
                d = pos[j] - pos[i]
                d_hat = d / (np.linalg.norm(d) + 1e-12)
                rot_i = self.body_rotation_matrix(i)
                if float(np.max(np.abs(self.CONNECTOR_DIRS @ (rot_i.T @ d_hat)))) < 0.99985:
                    continue
                rot_j = self.body_rotation_matrix(j)
                if float(np.max(np.abs(self.CONNECTOR_DIRS @ (rot_j.T @ d_hat)))) < 0.99985:
                    continue
                bonded = False
                for ci in range(6):
                    p_ci = self.get_connector_world_pos(i, ci)
                    for cj in range(6):
                        if float(np.linalg.norm(
                                p_ci - self.get_connector_world_pos(j, cj)
                        )) < self.AUTO_BOND_CONNECTOR_DIST:
                            self.create_bond(i, j)
                            bonded = True
                            break
                    if bonded:
                        break

    # ── Pivot control ──────────────────────────────────────────────────

    def start_pivot(self, pivot_idx: int, axis_idx: int,
                    rot_axis: np.ndarray, target_angle: float,
                    kp: float, kd: float, duration: float,
                    lattice_ref_body_idx: int,
                    target_pos_local: np.ndarray,
                    attract_body_idx: int,
                    attract_connector: int,
                    pivot_type: str = "corner",
                    repel_idx: Optional[int] = None):
        """Begin pivot: spring-asymmetric attachments or legacy PD + point2point.

        *target_pos_local* is the goal pivot COM offset in *lattice_ref_body_idx*'s
        current body frame. *attract_body_idx* / *attract_connector* are fixed for
        the whole maneuver (both lateral legs).
        """
        pos = self.get_positions()
        r0 = pos[pivot_idx] - pos[axis_idx]
        target_pos_local = np.asarray(target_pos_local, dtype=float).reshape(3)

        rep_mod = repel_idx if repel_idx is not None else axis_idx
        if rep_mod is not None and rep_mod >= 0:
            repel_connector = self.nearest_connector(
                rep_mod, pos[pivot_idx] - pos[rep_mod])
        else:
            repel_connector = None

        if self.USE_SPRING_BONDS:
            self._attachment_suppress.add((axis_idx, pivot_idx))
            self._active_pivots[pivot_idx] = PivotState(
                axis_idx=axis_idx,
                rot_axis=rot_axis / (np.linalg.norm(rot_axis) + 1e-12),
                target_angle=target_angle,
                kp=kp, kd=kd,
                duration=duration,
                start_time=self._sim_time,
                r0=r0.copy(),
                lattice_ref_body_idx=lattice_ref_body_idx,
                target_pos_local=target_pos_local,
                attract_body_idx=attract_body_idx,
                attract_connector=int(attract_connector),
                pivot_type=pivot_type,
                repel_idx=repel_idx,
                repel_connector=repel_connector,
            )
            logger.debug(
                "Spring pivot started: module {} axis {} type={} ref={} local={}",
                pivot_idx, axis_idx, pivot_type, lattice_ref_body_idx, target_pos_local)
            return

        # Rigid: drop fixed bond pivot–axis so point-to-point allows hinge motion
        # (covers lateral after handoff, where only sim.start_pivot is called).
        if self.RIGID_BONDS:
            lo, hi = min(pivot_idx, axis_idx), max(pivot_idx, axis_idx)
            if (lo, hi) in self._bonds:
                self.remove_bond(pivot_idx, axis_idx)

        use_rolling = (
            self.USE_ROLLING_SPHERE_PIVOT
            and self.RIGID_BONDS
            and not self.USE_SPRING_BONDS
        )
        if use_rolling:
            logger.debug(
                "Pivot {} uses rolling sphere coupling (no POINT2POINT)",
                pivot_idx,
            )
        else:
            child_frame = (-r0).tolist()
            cid = p.createConstraint(
                parentBodyUniqueId=self._body_ids[axis_idx],
                parentLinkIndex=-1,
                childBodyUniqueId=self._body_ids[pivot_idx],
                childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=child_frame,
                physicsClientId=self._physics_client,
            )
            p.changeConstraint(
                cid, maxForce=self.PIVOT_CONSTRAINT_MAX_FORCE,
                physicsClientId=self._physics_client)
            self._pivot_constraints[pivot_idx] = cid

        self._attachment_suppress.add((axis_idx, pivot_idx))

        # Keep collision enabled vs all bodies during pivot (no broad filter-off).

        self._active_pivots[pivot_idx] = PivotState(
            axis_idx=axis_idx,
            rot_axis=rot_axis / (np.linalg.norm(rot_axis) + 1e-12),
            target_angle=target_angle,
            kp=kp, kd=kd,
            duration=duration,
            start_time=self._sim_time,
            r0=r0.copy(),
            lattice_ref_body_idx=lattice_ref_body_idx,
            target_pos_local=target_pos_local,
            attract_body_idx=attract_body_idx,
            attract_connector=int(attract_connector),
            pivot_type=pivot_type,
            repel_idx=repel_idx,
            repel_connector=repel_connector,
        )
        logger.debug("Legacy pivot started: module {} around {} (angle={:.2f})",
                     pivot_idx, axis_idx, target_angle)

    def _crystal_target_quaternion_for_pivot(
            self, pivot_idx: int, ps: PivotState) -> Tuple[float, float, float, float]:
        """World orientation (xyzw) for crystalline connector alignment.

        Match ``lattice_ref_body_idx``; if that is the pivot, use the axis body.
        """
        ref = int(ps.lattice_ref_body_idx)
        if ref == pivot_idx:
            ref = int(ps.axis_idx)
        if not (0 <= ref < self.N):
            return (0.0, 0.0, 0.0, 1.0)
        _, orn = p.getBasePositionAndOrientation(
            self._body_ids[ref], physicsClientId=self._physics_client)
        return tuple(float(x) for x in orn)

    def stop_pivot(self, pivot_idx: int):
        if pivot_idx in self._active_pivots:
            ps = self._active_pivots[pivot_idx]
            if (
                    self.USE_CRYSTAL_ATTITUDE_SNAP_AFTER_ROLLING_PIVOT
                    and self.USE_ROLLING_SPHERE_PIVOT
                    and self.RIGID_BONDS
                    and not self.USE_SPRING_BONDS):
                orn_tgt = self._crystal_target_quaternion_for_pivot(
                    pivot_idx, ps)
                pos_cur, _ = p.getBasePositionAndOrientation(
                    self._body_ids[pivot_idx],
                    physicsClientId=self._physics_client)
                p.resetBasePositionAndOrientation(
                    self._body_ids[pivot_idx],
                    list(pos_cur),
                    list(orn_tgt),
                    physicsClientId=self._physics_client,
                )

            self._pivot_diagnostic_log.append(
                self._snapshot_pivot_metrics(pivot_idx, timed_out=ps.timed_out))
            self._attachment_suppress.discard((ps.axis_idx, pivot_idx))
            del self._active_pivots[pivot_idx]

            for j in range(self.N):
                if j != pivot_idx:
                    if (
                        pivot_idx in self._collision_off_with_all
                        or j in self._collision_off_with_all):
                        enable = False
                    elif self._mute_collision_for_bonded_after_pivot:
                        lo, hi = min(pivot_idx, j), max(pivot_idx, j)
                        bonded = (lo, hi) in self._bonds
                        enable = not bonded
                    else:
                        enable = True
                    self._set_pair_collision(pivot_idx, j, enable=enable)

        if pivot_idx in self._pivot_constraints:
            p.removeConstraint(self._pivot_constraints[pivot_idx],
                               physicsClientId=self._physics_client)
            del self._pivot_constraints[pivot_idx]

    def clear_pivot_diagnostics(self):
        self._pivot_diagnostic_log.clear()

    def get_pivot_diagnostic_log(self) -> List[Dict[str, Any]]:
        return list(self._pivot_diagnostic_log)

    def pivot_timed_out(self, pivot_idx: int) -> bool:
        """True if the active pivot for *pivot_idx* has timed out."""
        ps = self._active_pivots.get(pivot_idx)
        if ps is None:
            return False
        return ps.timed_out

    def pivot_collided(self, pivot_idx: int) -> bool:
        """True if the active pivot for *pivot_idx* detected a collision."""
        ps = self._active_pivots.get(pivot_idx)
        if ps is None:
            return False
        return ps.collided

    def is_pivot_complete(self, pivot_idx: int) -> bool:
        if pivot_idx not in self._active_pivots:
            return True
        ps = self._active_pivots[pivot_idx]
        pos = self.get_positions()
        vel = self.get_velocities()

        if self.USE_SPRING_BONDS:
            rp = pos[pivot_idx] - pos[ps.axis_idx]
            theta_c = self._measure_angle(rp, ps.r0, ps.rot_axis)
            angle_error = abs(theta_c - ps.target_angle)
            rp_mag = np.linalg.norm(rp) + 1e-12
            rp_hat = rp / rp_mag
            vp_rel = vel[pivot_idx] - vel[ps.axis_idx]
            tan_dir = np.cross(ps.rot_axis, rp_hat)
            tan_norm = np.linalg.norm(tan_dir)
            if tan_norm > 1e-8:
                tan_dir /= tan_norm
            omega_c = abs(np.dot(vp_rel, tan_dir) / rp_mag)
            t_elapsed = self._sim_time - ps.start_time
            settled = (angle_error < self.SPRING_PIVOT_ANGLE_TOL
                       and omega_c < self.SPRING_PIVOT_OMEGA_TOL)
            loose = (
                t_elapsed >= self.SPRING_PIVOT_LOOSE_TIME
                and angle_error < self.SPRING_PIVOT_LOOSE_ANGLE
                and omega_c < self.SPRING_PIVOT_LOOSE_OMEGA)
            dwell = t_elapsed >= 22.0 and angle_error < 0.09
            if settled or loose or dwell:
                ps.completed = True
                return True
            if self._sim_time - ps.start_time > self.MAX_PIVOT_TIME:
                logger.warning(
                    "Spring pivot {} timed out ({:.1f}s) "
                    "angle_err={:.4f} omega_tan={:.4f}",
                    pivot_idx, self.MAX_PIVOT_TIME, angle_error, omega_c)
                ps.timed_out = True
                ps.completed = True
                return True
            return False

        tgt_w = self.resolve_pivot_target_world(ps)
        pos_err = float(np.linalg.norm(pos[pivot_idx] - tgt_w))

        rp = pos[pivot_idx] - pos[ps.axis_idx]
        theta_c = self._measure_angle(rp, ps.r0, ps.rot_axis)
        angle_error = abs(theta_c - ps.target_angle)
        rp_mag = np.linalg.norm(rp) + 1e-12
        rp_hat = rp / rp_mag
        vp_rel = vel[pivot_idx] - vel[ps.axis_idx]
        tan_dir = np.cross(ps.rot_axis, rp_hat)
        tan_norm = np.linalg.norm(tan_dir)
        if tan_norm > 1e-8:
            tan_dir /= tan_norm
        omega_c = abs(np.dot(vp_rel, tan_dir) / rp_mag)
        rel_v = float(np.linalg.norm(vp_rel))

        rolling_rigid = (
            self.USE_ROLLING_SPHERE_PIVOT
            and self.RIGID_BONDS
            and not self.USE_SPRING_BONDS)
        if rolling_rigid:
            if (pos_err < float(self.PIVOT_POS_TOL)
                    and omega_c < float(self.PIVOT_OMEGA_TOL)
                    and rel_v < float(self.PIVOT_REL_V_TOL)):
                ps.completed = True
                return True
        else:
            if pos_err < float(self.PIVOT_POS_TOL):
                ps.completed = True
                return True

        if self._sim_time - ps.start_time > self.MAX_PIVOT_TIME:
            logger.warning(
                "Pivot {} timed out after {:.1f}s "
                "(angle_err={:.4f}, omega={:.4f}, pos_err={:.4f})",
                pivot_idx, self.MAX_PIVOT_TIME, angle_error, omega_c, pos_err)
            ps.timed_out = True
            ps.completed = True
            return True
        return False

    def _snapshot_pivot_metrics(self, pivot_idx: int, timed_out: bool) -> Dict[str, Any]:
        ps = self._active_pivots[pivot_idx]
        pos = self.get_positions()
        vel = self.get_velocities()
        tgt_w = self.resolve_pivot_target_world(ps)
        pos_err = float(np.linalg.norm(pos[pivot_idx] - tgt_w))
        vp_rel = vel[pivot_idx] - vel[ps.axis_idx]
        rp = pos[pivot_idx] - pos[ps.axis_idx]
        theta_c = self._measure_angle(rp, ps.r0, ps.rot_axis)
        rp_mag = np.linalg.norm(rp) + 1e-12
        rp_hat = rp / rp_mag
        tan_dir = np.cross(ps.rot_axis, rp_hat)
        tan_norm = np.linalg.norm(tan_dir)
        if tan_norm > 1e-8:
            tan_dir /= tan_norm
        omega_c = float(abs(np.dot(vp_rel, tan_dir) / rp_mag))
        out: Dict[str, Any] = {
            "pivot_idx": pivot_idx,
            "axis_idx": ps.axis_idx,
            "pos_error": pos_err,
            "v_rel": float(np.linalg.norm(vp_rel)),
            "elapsed": float(self._sim_time - ps.start_time),
            "timed_out": timed_out,
            "spring_mode": self.USE_SPRING_BONDS,
            "angle_error": float(abs(theta_c - ps.target_angle)),
            "omega_tan": omega_c,
        }
        return out

    def has_active_pivots(self) -> bool:
        return len(self._active_pivots) > 0

    def all_pivots_complete(self) -> bool:
        for pivot_idx in list(self._active_pivots.keys()):
            if not self.is_pivot_complete(pivot_idx):
                return False
        return True

    def step(self, dt: Optional[float] = None):
        if dt is None:
            dt = self.PHYSICS_DT
        n_substeps = max(1, int(round(dt / self.PHYSICS_DT)))
        for _ in range(n_substeps):
            self._apply_bond_forces()
            if self.USE_SPRING_BONDS:
                self._apply_attachment_springs()
            else:
                if self._active_pivots:
                    if self.USE_PIVOT_PD:
                        self._apply_pivot_forces()
                    else:
                        self._apply_pivot_attachment_actuation()
            diag = self._momentum_diagnostics
            if diag:
                P_act = self.total_linear_momentum()

            p.stepSimulation(physicsClientId=self._physics_client)
            self._sim_time += self.PHYSICS_DT

            if diag:
                P_bul = self.total_linear_momentum()
                com_bul = self.mass_weighted_com()
                dP_bullet = P_bul - P_act
                n_db = float(np.linalg.norm(dP_bullet))

            had_roll = (
                self._active_pivots
                and self.USE_ROLLING_SPHERE_PIVOT
                and self.RIGID_BONDS
                and not self.USE_SPRING_BONDS
            )
            if had_roll:
                for pidx, pst in list(self._active_pivots.items()):
                    ax = pst.axis_idx
                    if self.USE_PRESCRIBED_ROLLING_PIVOT:
                        apply_prescribed_rolling_pair(
                            self._physics_client,
                            self._body_ids[pidx],
                            self._body_ids[ax],
                            self.MODULE_RADIUS,
                            pst.r0,
                            pst.rot_axis,
                            pst.target_angle,
                            pst.start_time,
                            self._sim_time,
                            pst.duration,
                        )
                    else:
                        apply_rolling_sphere_pair(
                            self._physics_client,
                            self._body_ids[pidx],
                            self._body_ids[ax],
                            self.MODULE_RADIUS,
                            self.MODULE_MASS,
                            self.SPHERE_INERTIA,
                            distance_beta=self.ROLLING_DISTANCE_BETA,
                            max_position_correction=self.ROLLING_MAX_POS_CORRECTION,
                            max_lambda_norm=self.ROLLING_MAX_LAMBDA_NORM,
                            no_slip_iterations=self.ROLLING_NO_SLIP_ITERATIONS,
                        )

            self._auto_proximity_bond()

            if diag:
                P_roll = self.total_linear_momentum()
                com_roll = self.mass_weighted_com()
                dP_roll = (P_roll - P_bul) if had_roll else np.zeros(3)
                n_dr = float(np.linalg.norm(dP_roll)) if had_roll else 0.0
                pos = self.get_positions()
                com_mean = np.mean(pos, axis=0)
                com_err = float(np.linalg.norm(com_roll - com_mean))
                self._momentum_diag_substep_idx += 1
                s = self._momentum_diag_summary
                s["substeps"] = s.get("substeps", 0) + 1
                s["max_norm_P"] = max(
                    s.get("max_norm_P", 0.0),
                    float(np.linalg.norm(P_act)),
                    float(np.linalg.norm(P_bul)),
                    float(np.linalg.norm(P_roll)),
                )
                s["max_delta_P_bullet"] = max(
                    s.get("max_delta_P_bullet", 0.0), n_db)
                if had_roll:
                    s["rolling_substeps"] = s.get("rolling_substeps", 0) + 1
                    s["max_delta_P_rolling"] = max(
                        s.get("max_delta_P_rolling", 0.0), n_dr)
                s["max_com_mean_weighted_mismatch"] = max(
                    s.get("max_com_mean_weighted_mismatch", 0.0), com_err)
                if (
                    self._momentum_diag_max_samples > 0
                    and len(self._momentum_diag_samples)
                    < self._momentum_diag_max_samples
                    and (self._momentum_diag_substep_idx
                         % self._momentum_diag_subsample == 0)
                ):
                    self._momentum_diag_samples.append({
                        "t": float(self._sim_time),
                        "norm_P_act": float(np.linalg.norm(P_act)),
                        "norm_P_bullet": float(np.linalg.norm(P_bul)),
                        "norm_P_rolling": float(np.linalg.norm(P_roll)),
                        "delta_P_bullet": n_db,
                        "delta_P_rolling": n_dr,
                        "had_rolling": bool(had_roll),
                        "com_bullet": com_bul.tolist(),
                        "com_rolling": com_roll.tolist(),
                        "com_mean": com_mean.tolist(),
                        "com_mean_minus_weighted": com_err,
                    })

    def step_for(self, duration: float, trajectory_interval: float = 0.0
                 ) -> Optional[Dict[int, List[np.ndarray]]]:
        trajectories = None
        if trajectory_interval > 0:
            trajectories = {i: [] for i in range(self.N)}
            next_sample = self._sim_time
        end_time = self._sim_time + duration
        while self._sim_time < end_time - 1e-9:
            if trajectories is not None and self._sim_time >= next_sample - 1e-9:
                pos = self.get_positions()
                for i in range(self.N):
                    trajectories[i].append(pos[i].copy())
                next_sample += trajectory_interval
            self.step()
        if trajectories is not None:
            pos = self.get_positions()
            for i in range(self.N):
                trajectories[i].append(pos[i].copy())
        return trajectories

    def get_positions(self) -> np.ndarray:
        pos = np.zeros((self.N, 3))
        for i in range(self.N):
            pos[i] = p.getBasePositionAndOrientation(
                self._body_ids[i], physicsClientId=self._physics_client)[0]
        return pos

    def get_velocities(self) -> np.ndarray:
        vel = np.zeros((self.N, 3))
        for i in range(self.N):
            vel[i] = p.getBaseVelocity(
                self._body_ids[i], physicsClientId=self._physics_client)[0]
        return vel

    def total_linear_momentum(self) -> np.ndarray:
        """Total linear momentum (kg·m/s); sum_i m_i v_i."""
        v = self.get_velocities()
        return float(self.MODULE_MASS) * np.sum(v, axis=0)

    def mass_weighted_com(self) -> np.ndarray:
        """COM with uniform sphere masses; equals mean position when all m_i equal."""
        pos = self.get_positions()
        m = float(self.MODULE_MASS)
        return m * np.sum(pos, axis=0) / (self.N * m)

    def clear_momentum_diagnostics(self):
        """Reset per-run accumulators (call after world is fully built)."""
        self._momentum_diag_substep_idx = 0
        self._momentum_diag_samples = []
        pos = self.get_positions()
        P0 = self.total_linear_momentum()
        com0 = self.mass_weighted_com()
        com_mean0 = np.mean(pos, axis=0)
        self._momentum_diag_summary = {
            "substeps": 0,
            "rolling_substeps": 0,
            "max_norm_P": 0.0,
            "max_delta_P_bullet": 0.0,
            "max_delta_P_rolling": 0.0,
            "max_com_mean_weighted_mismatch": 0.0,
            "com_at_start": com0.tolist(),
            "com_mean_at_start": com_mean0.tolist(),
            "mean_minus_weighted_at_start": float(
                np.linalg.norm(com_mean0 - com0)),
            "P_at_start": P0.tolist(),
            "norm_P_at_start": float(np.linalg.norm(P0)),
        }

    def get_momentum_diagnostic_report(self) -> Dict[str, Any]:
        """Snapshot summary + samples for JSON export after a run."""
        pos = self.get_positions()
        Pf = self.total_linear_momentum()
        com_f = self.mass_weighted_com()
        com_mean_f = np.mean(pos, axis=0)
        s = dict(self._momentum_diag_summary)
        s["com_at_end"] = com_f.tolist()
        s["com_mean_at_end"] = com_mean_f.tolist()
        s["com_displacement_mass_weighted"] = (
            float(np.linalg.norm(com_f - np.array(s["com_at_start"]))))
        s["P_at_end"] = Pf.tolist()
        s["norm_P_at_end"] = float(np.linalg.norm(Pf))
        s["delta_P_start_to_end"] = float(
            np.linalg.norm(Pf - np.array(s["P_at_start"])))
        return {
            "summary": s,
            "module_mass_kg": float(self.MODULE_MASS),
            "N": int(self.N),
            "sample_interval_substeps": int(self._momentum_diag_subsample),
            "sample_count": len(self._momentum_diag_samples),
            "samples": list(self._momentum_diag_samples),
        }

    @property
    def sim_time(self) -> float:
        return self._sim_time

    # ── Body-frame / lattice helpers ──────────────────────────────────

    def body_rotation_matrix(self, body_idx: int) -> np.ndarray:
        _, orn = p.getBasePositionAndOrientation(
            self._body_ids[body_idx], physicsClientId=self._physics_client)
        return np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)

    def target_world_from_local(self, ref_idx: int,
                                local_offset: np.ndarray) -> np.ndarray:
        """World-space COM position from offset in *ref_idx* body frame."""
        pos = self.get_positions()
        R = self.body_rotation_matrix(ref_idx)
        return pos[ref_idx] + R @ np.asarray(local_offset, dtype=float).reshape(3)

    def resolve_pivot_target_world(self, ps: PivotState) -> np.ndarray:
        return self.target_world_from_local(
            ps.lattice_ref_body_idx, ps.target_pos_local)

    # ── Body-frame connector helpers ────────────────────────────────────

    def get_connector_world_pos(self, body_idx: int, connector_idx: int) -> np.ndarray:
        """World-frame position of a body-frame connector on module *body_idx*."""
        pos, orn = p.getBasePositionAndOrientation(
            self._body_ids[body_idx], physicsClientId=self._physics_client)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        world_dir = rot @ self.CONNECTOR_DIRS[connector_idx]
        return np.array(pos) + self.MODULE_RADIUS * world_dir

    def nearest_connector(self, body_idx: int, world_direction: np.ndarray) -> int:
        """Index of the connector on *body_idx* best aligned with *world_direction*."""
        _, orn = p.getBasePositionAndOrientation(
            self._body_ids[body_idx], physicsClientId=self._physics_client)
        rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        d_hat = world_direction / (np.linalg.norm(world_direction) + 1e-12)
        dots = (rot @ self.CONNECTOR_DIRS.T).T @ d_hat  # shape (6,)
        return int(np.argmax(dots))

    def pivot_axis_geometric_contact(
            self, pivot_idx: int, axis_idx: int, pos: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """World-frame tangency points on pivot and axis along their COM line."""
        d = pos[axis_idx] - pos[pivot_idx]
        dist = float(np.linalg.norm(d))
        if dist < 1e-8:
            return None
        n_hat = d / dist
        p_piv = pos[pivot_idx] + self.MODULE_RADIUS * n_hat
        p_ax = pos[axis_idx] - self.MODULE_RADIUS * n_hat
        return p_piv, p_ax

    def pivot_axis_actuation_lever_points(
            self, pivot_idx: int, axis_idx: int, pos: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """World points for rolling-mode actuation: opposite torques on pivot vs axis.

        Using the **near** contact on both bodies gives r_B = −r_A and F_B = −F_A,
        hence **identical** τ = r×F — wrong for rolling (pivot appeared to spin
        backward).  Axis stays at the near tangency; pivot uses the **far**
        hemisphere on the line of centers so τ_pivot ≈ −τ_axis (equal-and-opposite
        forces, zero net force).
        """
        d = pos[axis_idx] - pos[pivot_idx]
        dist = float(np.linalg.norm(d))
        if dist < 1e-8:
            return None
        n_hat = d / dist
        p_ax = pos[axis_idx] - self.MODULE_RADIUS * n_hat
        p_piv = pos[pivot_idx] - self.MODULE_RADIUS * n_hat
        return p_piv, p_ax

    def lateral_pivot_target_world(
            self, axis_idx: int, pivot_idx: int, handoff_idx: int) -> np.ndarray:
        """World-space pivot COM goal for a lateral handoff.

        Final COM is one nominal spacing from the handoff module (same offset
        as pivot–axis), so mating connectors align. Equivalently:
        ``pos[handoff] + (pos[pivot] - pos[axis])``.
        """
        pos = self.get_positions()
        return pos[handoff_idx] + (pos[pivot_idx] - pos[axis_idx])

    def lateral_handoff_attract_connector(
            self, axis_idx: int, pivot_idx: int, handoff_idx: int) -> int:
        """Connector on *handoff_idx* whose world point is ~``NOMINAL_DIST`` from
        the connector on *axis_idx* that faces *pivot_idx* (axis–pivot bond)."""
        pos = self.get_positions()
        d_bond = pos[pivot_idx] - pos[axis_idx]
        c_axis = self.nearest_connector(axis_idx, d_bond)
        p_attach = self.get_connector_world_pos(axis_idx, c_axis)
        nom = float(self.NOMINAL_DIST)
        best_j = 0
        best_err = float("inf")
        for j in range(6):
            err = abs(
                float(np.linalg.norm(
                    self.get_connector_world_pos(handoff_idx, j) - p_attach))
                - nom)
            if err < best_err:
                best_err = err
                best_j = j
        return best_j

    # ── Internal force computation ─────────────────────────────────────

    def _bond_K_C(self) -> Tuple[float, float]:
        if self.USE_SPRING_BONDS:
            return self.K_BOND, self.C_BOND
        return self.K_BOND_SOFT_LEGACY, self.C_BOND_SOFT_LEGACY

    def _apply_bond_forces(self):
        if not self._bonds:
            return
        if self.RIGID_BONDS and not self.USE_SPRING_BONDS:
            return
        k_b, c_b = self._bond_K_C()
        pos = self.get_positions()
        vel = self.get_velocities()
        for (i, j) in self._bonds:
            rr = pos[i] - pos[j]
            dd = np.linalg.norm(rr) + 1e-12
            rh = rr / dd
            vr = vel[i] - vel[j]
            vrad = np.dot(vr, rh)
            ff = -k_b * (dd - self.NOMINAL_DIST) - c_b * vrad
            force = ff * rh
            if self.USE_SPRING_BONDS:
                fn = float(np.linalg.norm(force))
                if fn > self.BOND_FORCE_CAP:
                    force = force * (self.BOND_FORCE_CAP / fn)
            p.applyExternalForce(
                self._body_ids[i], -1, force.tolist(),
                pos[i].tolist(), p.WORLD_FRAME,
                physicsClientId=self._physics_client)
            p.applyExternalForce(
                self._body_ids[j], -1, (-force).tolist(),
                pos[j].tolist(), p.WORLD_FRAME,
                physicsClientId=self._physics_client)

    def _goal_com_for_attachment(self, recv: int, src: int,
                                  pos: np.ndarray) -> np.ndarray:
        ps = self._active_pivots.get(recv)
        if ps is not None and ps.axis_idx == src:
            return self.resolve_pivot_target_world(ps)
        return pos[recv]

    def _surface_point_toward(
            self, pos_src: np.ndarray, pos_recv: np.ndarray,
            goal_com_recv: np.ndarray) -> np.ndarray:
        u = goal_com_recv - pos_src
        nu = np.linalg.norm(u)
        if nu < 1e-9:
            u = pos_recv - pos_src
            nu = np.linalg.norm(u)
        if nu < 1e-9:
            u = np.array([1.0, 0.0, 0.0])
            nu = 1.0
        u = u / nu
        return pos_src + self.MODULE_RADIUS * u

    def _apply_directed_attachment(
            self, recv: int, src: int, pos: np.ndarray, vel: np.ndarray,
            *, k_scale: float = 1.0, c_scale: float = 1.0,
            cap_scale: float = 1.0) -> None:
        """Damped pull: recv COM toward surface anchor on src; equal opposite on src."""
        k_att = self.K_ATTACHMENT * k_scale
        c_att = self.C_ATTACHMENT * c_scale
        f_cap = self.ATTACHMENT_FORCE_CAP * cap_scale
        goal = self._goal_com_for_attachment(recv, src, pos)
        p_tgt = self._surface_point_toward(pos[src], pos[recv], goal)
        delta = p_tgt - pos[recv]
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            return
        h = delta / dist
        vr = vel[recv] - vel[src]
        vrad = float(np.dot(vr, h))
        dist_eff = min(dist, 0.42)
        f_mag = k_att * dist_eff - c_att * vrad
        f_vec = f_mag * h
        fn = float(np.linalg.norm(f_vec))
        if fn > f_cap:
            f_vec *= f_cap / fn
        p.applyExternalForce(
            self._body_ids[recv], -1,
            f_vec.tolist(), pos[recv].tolist(),
            p.WORLD_FRAME, physicsClientId=self._physics_client)
        p.applyExternalForce(
            self._body_ids[src], -1,
            (-f_vec).tolist(), p_tgt.tolist(),
            p.WORLD_FRAME, physicsClientId=self._physics_client)

    def _apply_attachment_springs(self):
        if not self.USE_SPRING_BONDS or not self._bonds:
            return
        pos = self.get_positions()
        vel = self.get_velocities()

        for lo, hi in self._bonds:
            for recv, src in ((lo, hi), (hi, lo)):
                if (recv, src) in self._attachment_suppress:
                    continue
                self._apply_directed_attachment(recv, src, pos, vel)

    def _apply_pivot_attachment_actuation(self):
        """Cartesian PD actuation for active pivots (rigid mode).

        F = Kp * (target_world - pos_pivot) - Kd * v_rel

        Equal-and-opposite forces on pivot and axis.  With
        ``USE_ROLLING_SPHERE_PIVOT`` the forces are applied at lever points
        (axis near contact, pivot far hemisphere) so the pair gets opposing
        spin; otherwise forces act at the COMs.
        """
        if self.USE_SPRING_BONDS or not self._active_pivots:
            return
        kp = self.PIVOT_PD_KP * self._pivot_attract_scale
        kd = self.PIVOT_PD_KD
        f_cap = self.PIVOT_PD_F_MAX
        pos = self.get_positions()
        vel = self.get_velocities()
        for pivot_idx, ps in self._active_pivots.items():
            ax = ps.axis_idx

            tgt_w = self.resolve_pivot_target_world(ps)
            pos_err = tgt_w - pos[pivot_idx]
            v_rel = vel[pivot_idx] - vel[ax]

            f_vec = kp * pos_err - kd * v_rel

            fn = float(np.linalg.norm(f_vec))
            if fn > f_cap:
                f_vec *= f_cap / fn

            use_contact_forces = (
                self.USE_ROLLING_SPHERE_PIVOT
                and self.RIGID_BONDS
                and not self.USE_SPRING_BONDS
            )
            if use_contact_forces:
                pts = self.pivot_axis_actuation_lever_points(pivot_idx, ax, pos)
                if pts is not None:
                    p_piv, p_ax = pts
                    p.applyExternalForce(
                        self._body_ids[pivot_idx], -1,
                        f_vec.tolist(), p_piv.tolist(),
                        p.WORLD_FRAME, physicsClientId=self._physics_client)
                    p.applyExternalForce(
                        self._body_ids[ax], -1,
                        (-f_vec).tolist(), p_ax.tolist(),
                        p.WORLD_FRAME, physicsClientId=self._physics_client)
                    continue
            p.applyExternalForce(
                self._body_ids[pivot_idx], -1,
                f_vec.tolist(), pos[pivot_idx].tolist(),
                p.WORLD_FRAME, physicsClientId=self._physics_client)
            p.applyExternalForce(
                self._body_ids[ax], -1,
                (-f_vec).tolist(), pos[ax].tolist(),
                p.WORLD_FRAME, physicsClientId=self._physics_client)

    def _apply_pivot_forces(self):
        if not self._active_pivots:
            return
        pos = self.get_positions()
        vel = self.get_velocities()

        for pivot_idx, ps in self._active_pivots.items():
            ax = ps.axis_idx
            rp = pos[pivot_idx] - pos[ax]
            rp_mag = np.linalg.norm(rp) + 1e-12
            rp_hat = rp / rp_mag
            vp_rel = vel[pivot_idx] - vel[ax]
            t_elapsed = self._sim_time - ps.start_time
            s_frac = np.clip(t_elapsed / max(ps.duration, 1e-6), 0.0, 1.0)
            s_val = s_frac * s_frac * (3.0 - 2.0 * s_frac)
            s_deriv = 6.0 * s_frac * (1.0 - s_frac)
            theta_d = ps.target_angle * s_val
            omega_d = ps.target_angle * s_deriv / max(ps.duration, 1e-6)
            theta_c = self._measure_angle(rp, ps.r0, ps.rot_axis)
            tan_dir = np.cross(ps.rot_axis, rp_hat)
            tan_norm = np.linalg.norm(tan_dir)
            if tan_norm > 1e-8:
                tan_dir /= tan_norm
            omega_c = np.dot(vp_rel, tan_dir) / rp_mag
            tau_raw = ps.kp * (theta_d - theta_c) + ps.kd * (omega_d - omega_c)
            tau_sat = np.clip(tau_raw, -self.MAX_MOTOR_TORQUE, self.MAX_MOTOR_TORQUE)
            f_total = (tau_sat / rp_mag) * tan_dir
            use_contact_forces = (
                self.USE_ROLLING_SPHERE_PIVOT
                and self.RIGID_BONDS
                and not self.USE_SPRING_BONDS
            )
            if use_contact_forces:
                pts = self.pivot_axis_actuation_lever_points(pivot_idx, ax, pos)
                if pts is not None:
                    p_piv, p_ax = pts
                    p.applyExternalForce(
                        self._body_ids[pivot_idx], -1,
                        f_total.tolist(), p_piv.tolist(),
                        p.WORLD_FRAME, physicsClientId=self._physics_client)
                    p.applyExternalForce(
                        self._body_ids[ax], -1,
                        (-f_total).tolist(), p_ax.tolist(),
                        p.WORLD_FRAME, physicsClientId=self._physics_client)
                    continue
            p.applyExternalForce(
                self._body_ids[pivot_idx], -1,
                f_total.tolist(), pos[pivot_idx].tolist(),
                p.WORLD_FRAME, physicsClientId=self._physics_client)
            p.applyExternalForce(
                self._body_ids[ax], -1,
                (-f_total).tolist(), pos[ax].tolist(),
                p.WORLD_FRAME, physicsClientId=self._physics_client)

    def _apply_damping(self):
        if self.DAMPING_COEFF <= 0:
            return
        vel = self.get_velocities()
        vel_com = np.mean(vel, axis=0)
        pos = self.get_positions()
        for i in range(self.N):
            f_damp = -self.DAMPING_COEFF * (vel[i] - vel_com)
            p.applyExternalForce(
                self._body_ids[i], -1, f_damp.tolist(),
                pos[i].tolist(), p.WORLD_FRAME,
                physicsClientId=self._physics_client)

    @staticmethod
    def _measure_angle(rp, r0, rot_axis):
        ref_p = r0 - np.dot(r0, rot_axis) * rot_axis
        cur_p = rp - np.dot(rp, rot_axis) * rot_axis
        ref_n = np.linalg.norm(ref_p) + 1e-12
        cur_n = np.linalg.norm(cur_p) + 1e-12
        cos_a = np.clip(np.dot(ref_p, cur_p) / (ref_n * cur_n), -1.0, 1.0)
        cross_v = np.cross(ref_p, cur_p)
        sgn = 1.0 if np.dot(cross_v, rot_axis) >= 0 else -1.0
        return sgn * np.arccos(cos_a)

    @classmethod
    def compute_pd_gains(cls, r_vec, duration):
        r = np.linalg.norm(r_vec)
        I_orbit = cls.MODULE_MASS * r ** 2
        I_eff = 3.0 * (I_orbit + cls.SPHERE_INERTIA)
        omega_n = 3.0 * 2.0 * np.pi / duration
        return I_eff * omega_n ** 2, 2.0 * I_eff * omega_n

    @staticmethod
    def get_rotation_axis(pivot_pos, axis_pos, target_pos):
        v1 = pivot_pos - axis_pos
        v2 = target_pos - axis_pos
        axis = np.cross(v1, v2)
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            return axis / norm
        ref = v1 if np.linalg.norm(v1) > 1e-8 else v2
        if np.linalg.norm(ref) > 1e-8:
            perp = np.array([1, 0, 0]) if abs(ref[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(ref, perp)
            norm = np.linalg.norm(axis)
            if norm > 1e-8:
                return axis / norm
        return np.array([0.0, 0.0, 1.0])

    @staticmethod
    def compute_lateral_midpoint(pivot_pos, old_pos, new_pos, target_pos):
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

    def disconnect(self):
        if self._physics_client is not None:
            try:
                p.disconnect(self._physics_client)
            except Exception:
                pass
            self._physics_client = None

    def __del__(self):
        self.disconnect()
