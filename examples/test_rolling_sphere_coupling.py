"""
Verify sphere–sphere rolling coupling: 2R separation stabilization and slip decay.

Run from repo root: py -3 examples/test_rolling_sphere_coupling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pybullet as p

from src.bullet_sim import BulletSimulator
from src.rolling_sphere_coupling import (
    apply_no_slip_velocity_impulse,
    apply_prescribed_rolling_pair,
    apply_rolling_sphere_pair,
    prescribed_theta_and_rate,
    tangential_slip_norm,
)


def _slread_pair(
    sim: BulletSimulator, i: int, j: int, R: float
) -> tuple[float, float]:
    """Return (COM separation, tangential slip magnitude at contact)."""
    pos = sim.get_positions()
    d = pos[j] - pos[i]
    dist = float(np.linalg.norm(d))
    v_a, w_a = p.getBaseVelocity(
        sim._body_ids[i], physicsClientId=sim._physics_client)
    v_b, w_b = p.getBaseVelocity(
        sim._body_ids[j], physicsClientId=sim._physics_client)
    v_a = np.array(v_a, dtype=float)
    w_a = np.array(w_a, dtype=float)
    v_b = np.array(v_b, dtype=float)
    w_b = np.array(w_b, dtype=float)
    st = tangential_slip_norm(pos[i], pos[j], v_a, w_a, v_b, w_b, R)
    return dist, st


def main() -> None:
    R = float(BulletSimulator.MODULE_RADIUS)
    target_sep = 2.0 * R
    m = float(BulletSimulator.MODULE_MASS)
    I = float(BulletSimulator.SPHERE_INERTIA)

    pos0 = np.array([[-R, 0.0, 0.0], [R, 0.0, 0.0]], dtype=float)
    bonded = np.zeros((2, 2), dtype=bool)

    sim = BulletSimulator(2, pos0, bonded, gui=False)
    try:
        cid0, cid1 = sim._body_ids[0], sim._body_ids[1]
        client = sim._physics_client

        # --- Test A: single no-slip impulse removes synthetic slip ---
        p.resetBaseVelocity(cid0, [0.15, -0.08, 0.02], [0.0, 0.0, 0.0],
                            physicsClientId=client)
        p.resetBaseVelocity(cid1, [0.0, 0.0, 0.0], [0.1, 0.0, 0.0],
                            physicsClientId=client)
        _, slip0 = _slread_pair(sim, 0, 1, R)
        apply_no_slip_velocity_impulse(client, cid0, cid1, R, m, I)
        _, slip1 = _slread_pair(sim, 0, 1, R)
        assert slip1 < slip0 * 0.05 + 1e-4, (
            f"no-slip impulse should crush slip: {slip0} -> {slip1}"
        )

        # --- Test B: perturbed separation + many substeps converge ---
        p.resetBasePositionAndOrientation(
            cid0, [-R - 0.04, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client)
        p.resetBasePositionAndOrientation(
            cid1, [R + 0.04, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client)
        p.resetBaseVelocity(cid0, [0.05, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=client)
        p.resetBaseVelocity(cid1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=client)

        last_slip = 1e9
        for _ in range(800):
            p.stepSimulation(physicsClientId=client)
            apply_rolling_sphere_pair(client, cid0, cid1, R, m, I)
            dist, slip = _slread_pair(sim, 0, 1, R)
            if slip < last_slip:
                last_slip = slip

        dist_f, slip_f = _slread_pair(sim, 0, 1, R)
        assert abs(dist_f - target_sep) < 0.02, (
            f"separation should stay ~2R: got {dist_f}, want {target_sep}"
        )
        assert slip_f < 0.02, f"slip should decay: final {slip_f}"

        # --- Test D: prescribed rolling holds2R and zero slip at end of smoothstep ---
        td, dur = 1.0, 0.5
        th0, thd0 = prescribed_theta_and_rate(0.0, dur, np.pi / 3)
        th1, thd1 = prescribed_theta_and_rate(dur, dur, np.pi / 3)
        assert abs(th0) < 1e-9 and abs(thd0) < 1e-9
        assert abs(th1 - np.pi / 3) < 1e-9 and abs(thd1) < 1e-9

        p.resetBasePositionAndOrientation(
            cid0, [-R, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client)
        p.resetBasePositionAndOrientation(
            cid1, [R, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0],
            physicsClientId=client)
        p.resetBaseVelocity(cid0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=client)
        p.resetBaseVelocity(cid1, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=client)
        r0 = np.array([-2.0 * R, 0.0, 0.0], dtype=float)
        k_axis = np.array([0.0, 0.0, 1.0], dtype=float)
        apply_prescribed_rolling_pair(
            client, cid0, cid1, R, r0, k_axis, np.pi / 2,
            start_time=0.0, sim_time=td, duration=td,
        )
        dist_p, slip_p = _slread_pair(sim, 0, 1, R)
        assert abs(dist_p - target_sep) < 1e-6, (
            f"prescribed separation: {dist_p} vs {target_sep}"
        )
        assert slip_p < 1e-5, f"prescribed end slip {slip_p}"

        # --- Test C: BulletSimulator.step invokes rolling when pivot active ---
        saved = BulletSimulator.USE_ROLLING_SPHERE_PIVOT
        BulletSimulator.USE_ROLLING_SPHERE_PIVOT = True
        sim2 = None
        try:
            nom = float(BulletSimulator.NOMINAL_DIST)
            pos_c = np.array([[-nom, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
            bond_c = np.zeros((2, 2), dtype=bool)
            bond_c[0, 1] = bond_c[1, 0] = True
            sim2 = BulletSimulator(2, pos_c, bond_c, gui=False)
            lattice_ref = 1
            ppos = sim2.get_positions()
            tgt = ppos[lattice_ref] + np.array([0.0, nom, 0.0], dtype=float)
            Rb = sim2.body_rotation_matrix(lattice_ref)
            tloc = Rb.T @ (tgt - ppos[lattice_ref])
            conn = sim2.nearest_connector(lattice_ref, tgt - ppos[lattice_ref])
            r_vec = ppos[0] - ppos[1]
            rot_axis = sim2.get_rotation_axis(ppos[0], ppos[1], tgt)
            r_target = tgt - ppos[1]
            cos_a = np.clip(
                np.dot(r_vec, r_target)
                / (np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1.0,
                1.0,
            )
            angle = float(np.arccos(cos_a))
            kp, kd = sim2.compute_pd_gains(r_vec, duration=12.0)
            sim2.start_pivot(
                0,
                1,
                rot_axis,
                angle,
                kp,
                kd,
                duration=12.0,
                lattice_ref_body_idx=lattice_ref,
                target_pos_local=tloc,
                attract_body_idx=lattice_ref,
                attract_connector=conn,
                pivot_type="corner",
            )
            assert 0 not in sim2._pivot_constraints, "rolling mode skips P2P"
            sim2.step(0.02)
            assert 0 in sim2._active_pivots
        finally:
            BulletSimulator.USE_ROLLING_SPHERE_PIVOT = saved
            if sim2 is not None:
                sim2.disconnect()

        print("ROLLING_SPHERE_COUPLING: PASS")
    finally:
        sim.disconnect()


if __name__ == "__main__":
    main()
