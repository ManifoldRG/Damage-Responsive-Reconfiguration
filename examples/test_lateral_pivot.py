"""
Rigid lateral-style pivot smoke test (no full decentralized policy).

Layout: axis M0 at origin, handoff neighbor M1 at +X, pivot M2 at +Y (L-shape).
Bonds M0–M1 and M0–M2. ``start_pivot`` (lateral) strips M0–M2 fixed bond and adds
point-to-point; attachment / angle attraction drives the hinge.

Pass criteria: hinge angle advances materially before timeout or ``is_pivot_complete``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bullet_sim import BulletSimulator


def main():
    N = 3
    pos0 = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    bonded = np.zeros((N, N), dtype=bool)
    bonded[0, 1] = bonded[1, 0] = True
    bonded[0, 2] = bonded[2, 0] = True

    pivot_idx, axis_idx = 2, 0
    handoff_idx = 1
    # Target on +X side (toward handoff direction), not colinear with M0–M2
    target_pos = np.array([0.8, 0.4, 0.0], dtype=float)

    sim = BulletSimulator(N, pos0, bonded, gui=False)
    try:
        pos = sim.get_positions()
        lattice_ref = axis_idx
        R0 = sim.body_rotation_matrix(lattice_ref)
        target_pos_local = R0.T @ (target_pos - pos[lattice_ref])
        target_world = sim.target_world_from_local(lattice_ref, target_pos_local)
        attract_body = handoff_idx
        attract_conn = sim.nearest_connector(
            handoff_idx, target_pos - pos[handoff_idx])
        r_vec = pos[pivot_idx] - pos[axis_idx]
        rot_axis = sim.get_rotation_axis(
            pos[pivot_idx], pos[axis_idx], target_world
        )
        r_target = target_world - pos[axis_idx]
        cos_angle = np.clip(
            np.dot(r_vec, r_target)
            / (
                np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12
            ),
            -1.0,
            1.0,
        )
        angle = float(np.arccos(cos_angle))
        kp, kd = sim.compute_pd_gains(r_vec, duration=12.0)

        print(
            f"Spring bonds: {sim.USE_SPRING_BONDS}, "
            f"USE_PIVOT_PD: {sim.USE_PIVOT_PD}, "
            f"target_angle={angle:.4f} rad"
        )

        sim.start_pivot(
            pivot_idx,
            axis_idx,
            rot_axis,
            angle,
            kp,
            kd,
            duration=12.0,
            lattice_ref_body_idx=lattice_ref,
            target_pos_local=target_pos_local,
            attract_body_idx=attract_body,
            attract_connector=attract_conn,
            pivot_type="lateral",
        )

        assert sim.has_active_pivots(), "pivot should be active"
        # No fixed M0–M2 after start_pivot in rigid mode
        key = (min(0, 2), max(0, 2))
        assert key not in sim._bonds, "pivot–axis fixed bond should be removed"

        pivot_pos_start = sim.get_positions()[pivot_idx].copy()
        tick_dt = 0.01
        max_sim = 45.0
        last_print = -1.0
        theta_at = {}

        while sim.sim_time < max_sim - 1e-9:
            sim.step(tick_dt)
            if sim.is_pivot_complete(pivot_idx):
                print(f"Pivot complete at sim_time={sim.sim_time:.2f}s")
                break
            if sim.sim_time - last_print >= 2.0:
                ps = sim._active_pivots[pivot_idx]
                pos = sim.get_positions()
                rp = pos[pivot_idx] - pos[axis_idx]
                theta_c = sim._measure_angle(rp, ps.r0, ps.rot_axis)
                err = abs(theta_c - ps.target_angle)
                print(
                    f"t={sim.sim_time:.1f}s  theta={theta_c:.4f}  "
                    f"target={ps.target_angle:.4f}  |err|={err:.4f}"
                )
                last_print = sim.sim_time
            for mark in (0.1, 1.0, 5.0):
                if mark not in theta_at and sim.sim_time >= mark - 1e-6:
                    ps = sim._active_pivots[pivot_idx]
                    pos = sim.get_positions()
                    rp = pos[pivot_idx] - pos[axis_idx]
                    theta_at[mark] = sim._measure_angle(rp, ps.r0, ps.rot_axis)
        else:
            print(f"No completion within {max_sim}s sim time")

        pos = sim.get_positions()
        ps = sim._active_pivots.get(pivot_idx)
        pivot_displacement = float(
            np.linalg.norm(pos[pivot_idx] - pivot_pos_start))
        if ps is not None:
            rp = pos[pivot_idx] - pos[axis_idx]
            theta_fin = sim._measure_angle(rp, ps.r0, ps.rot_axis)
            print(f"Final theta={theta_fin:.4f}  target={ps.target_angle:.4f}")
        print("Final positions:")
        for i in range(N):
            print(f"  body {i}: {pos[i]}")
        d_theta = None
        if 0.1 in theta_at and 5.0 in theta_at:
            d_theta = abs(theta_at[5.0] - theta_at[0.1])
        hinge_ok = (
            pivot_displacement > 0.05 or (d_theta is not None and d_theta > 0.05)
        )
        print(
            f"\n--- Lateral pivot motion check ---\n"
            f"pivot COM displacement: {pivot_displacement:.4f} m\n"
            f"|d_theta(0.1s to 5s)|: {d_theta}\n"
            f"LATERAL_PIVOT_PHYSICS_POSSIBLE: {hinge_ok}"
            + ("  (PASS)" if hinge_ok else "  (FAIL)")
        )
    finally:
        sim.disconnect()


if __name__ == "__main__":
    main()
