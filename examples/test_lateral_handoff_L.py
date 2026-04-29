"""
Simplest lateral handoff: L-shaped chain module1--module2--module3 (right angle).

Initial bonds: 1-2 and 2-3. Goal: module 1 transfers from 2 to 3 so that
finally 1-3 and 3-2 are bonded (and 1-2 is gone).

Uses rigid BulletSimulator + the same pivot / handoff sequence as agent_policy
(create_bond to handoff, remove_bond from old axis, stop_pivot, start_pivot).

During the second p2p leg the 1-3 fixed bond is stripped again by ``start_pivot``;
the decentralized agent runs ``_reconnect_bonds`` after pivot complete. This test
calls ``create_bond(0, 2)`` when spacing allows, to mirror that final graph.

Bodies are indexed 0,1,2 = modules 1,2,3. Corner module 2 = body index 1.

**Corner:** four sites on the axis perpendicular to the pivot–axis arm.
**Lateral:** pivot COM goal is ``pos[handoff] + (pos[pivot] − pos[axis])`` —
one nominal spacing from the handoff COM; pivot COM moves by one diameter along
a pivot body-frame cardinal (the axis→handoff leg). Attract the handoff
connector one diameter from the axis connector that faces the pivot
(``lateral_handoff_attract_connector``).

Here (labels M0,M1,M2 = bodies 0,1,2): axis M1 at origin, pivot M0 at (−nom,0),
handoff M2 at (0,nom) → goal COM (−nom, nom, 0).

Run: py -3 examples/test_lateral_handoff_L.py
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.bullet_sim import BulletSimulator


def _start_lateral_pivot(
    sim: BulletSimulator,
    pivot_idx: int,
    axis_idx: int,
    lattice_ref_body_idx: int,
    target_pos_local: np.ndarray,
    attract_body_idx: int,
    attract_connector: int,
    pivot_type: str = "lateral",
    repel_idx: Optional[int] = None,
) -> None:
    pos = sim.get_positions()
    target_world = sim.target_world_from_local(
        lattice_ref_body_idx, target_pos_local)
    my_pos = pos[pivot_idx]
    axis_pos = pos[axis_idx]
    r_vec = my_pos - axis_pos
    rot_axis = sim.get_rotation_axis(my_pos, axis_pos, target_world)
    r_target = target_world - axis_pos
    cos_angle = np.clip(
        np.dot(r_vec, r_target)
        / (np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
        -1.0,
        1.0,
    )
    angle = float(np.arccos(cos_angle))
    kp, kd = sim.compute_pd_gains(r_vec, duration=12.0)
    sim.start_pivot(
        pivot_idx,
        axis_idx,
        rot_axis,
        angle,
        kp,
        kd,
        duration=12.0,
        lattice_ref_body_idx=lattice_ref_body_idx,
        target_pos_local=target_pos_local,
        attract_body_idx=attract_body_idx,
        attract_connector=attract_connector,
        pivot_type=pivot_type,
        repel_idx=repel_idx,
    )


def main():
    # Module 1 at -X, module 2 (corner) at origin, module 3 at +Y
    N = 3
    nom = float(BulletSimulator.NOMINAL_DIST)
    pos0 = np.array(
        [
            [-nom, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, nom, 0.0],
        ],
        dtype=float,
    )
    bonded = np.zeros((N, N), dtype=bool)
    bonded[0, 1] = bonded[1, 0] = True
    bonded[1, 2] = bonded[2, 1] = True

    pivot_idx, axis_idx, handoff_idx = 0, 1, 2

    sim = BulletSimulator(N, pos0, bonded, gui=False)
    handoff_done = False
    max_sim = 90.0
    tick_dt = 0.01

    pos_init = sim.get_positions()
    lattice_ref = axis_idx
    R0 = sim.body_rotation_matrix(lattice_ref)
    target_pos = sim.lateral_pivot_target_world(
        axis_idx, pivot_idx, handoff_idx)
    target_pos_local = R0.T @ (target_pos - pos_init[lattice_ref])
    attract_body = handoff_idx
    attract_conn = sim.lateral_handoff_attract_connector(
        axis_idx, pivot_idx, handoff_idx)

    try:
        _start_lateral_pivot(
            sim, pivot_idx, axis_idx, lattice_ref, target_pos_local,
            attract_body, attract_conn)

        print(
            f"USE_SPRING_BONDS={sim.USE_SPRING_BONDS} USE_PIVOT_PD={sim.USE_PIVOT_PD}"
        )
        print(f"target_pos (module 1 goal)={target_pos}")
        print(f"handoff_contact_distance={sim.handoff_contact_distance():.3f}")

        while sim.sim_time < max_sim:
            sim.step(tick_dt)

            if (
                sim.has_active_pivots()
                and not handoff_done
                and pivot_idx in sim._active_pivots
            ):
                ps = sim._active_pivots[pivot_idx]
                if ps.pivot_type == "lateral":
                    pos = sim.get_positions()
                    d = float(np.linalg.norm(pos[pivot_idx] - pos[handoff_idx]))
                    if d < sim.handoff_contact_distance():
                        print(
                            f"t={sim.sim_time:.2f}s HANDOFF: d(pivot,mod3)={d:.4f}"
                        )
                        sim.create_bond(pivot_idx, handoff_idx)
                        sim.remove_bond(pivot_idx, axis_idx)
                        sim.stop_pivot(pivot_idx)
                        pos = sim.get_positions()
                        new_ax = handoff_idx
                        my_pos = pos[pivot_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = sim.target_world_from_local(
                            lattice_ref, target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w
                        )
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target)
                            / (
                                np.linalg.norm(r_vec)
                                * np.linalg.norm(r_target)
                                + 1e-12
                            ),
                            -1.0,
                            1.0,
                        )
                        angle = float(np.arccos(cos_a))
                        kp, kd = sim.compute_pd_gains(r_vec, duration=12.0)
                        sim.start_pivot(
                            pivot_idx,
                            new_ax,
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
                            repel_idx=-1,
                        )
                        handoff_done = True

            if sim.has_active_pivots() and pivot_idx in sim._active_pivots:
                ps = sim._active_pivots[pivot_idx]
                pos = sim.get_positions()
                vel = sim.get_velocities()
                rp = pos[pivot_idx] - pos[ps.axis_idx]
                rp_mag = float(np.linalg.norm(rp)) + 1e-12
                rp_hat = rp / rp_mag
                theta_c = sim._measure_angle(rp, ps.r0, ps.rot_axis)
                vp_rel = vel[pivot_idx] - vel[ps.axis_idx]
                tan_dir = np.cross(ps.rot_axis, rp_hat)
                tn = np.linalg.norm(tan_dir)
                if tn > 1e-8:
                    tan_dir /= tn
                omega_c = float(np.dot(vp_rel, tan_dir) / rp_mag)
                angle_err = abs(theta_c - ps.target_angle)
                t_el = sim.sim_time - ps.start_time
                if int(t_el * 10) % 50 == 0 and t_el > 0.05:
                    p_tgt = sim.get_connector_world_pos(
                        ps.attract_body_idx, ps.attract_connector)
                    d_tgt = float(np.linalg.norm(p_tgt - pos[pivot_idx]))
                    rep = ps.repel_idx if ps.repel_idx is not None else ps.axis_idx
                    if rep >= 0 and ps.repel_connector is not None:
                        p_old = sim.get_connector_world_pos(rep, ps.repel_connector)
                        d_old = float(np.linalg.norm(pos[pivot_idx] - p_old))
                    else:
                        d_old = float("nan")
                    kp = sim.PIVOT_PD_KP
                    kd = sim.PIVOT_PD_KD
                    tgt_w = sim.resolve_pivot_target_world(ps)
                    pos_err = float(np.linalg.norm(tgt_w - pos[pivot_idx]))
                    v_rel = vel[pivot_idx] - vel[ps.axis_idx]
                    v_rel_mag = float(np.linalg.norm(v_rel))
                    print(
                        f"  t={sim.sim_time:.2f}s  angle_err={angle_err:.6f}  "
                        f"omega={omega_c:.6f}  rp_mag={rp_mag:.4f}  "
                        f"Kp*pos_err={kp*pos_err:.3f}  Kd*v_rel={kd*v_rel_mag:.3f}  "
                        f"d_tgt={d_tgt:.4f}  d_old={d_old:.4f}")

            if sim.has_active_pivots() and sim.is_pivot_complete(pivot_idx):
                sim.stop_pivot(pivot_idx)
                print(f"t={sim.sim_time:.2f}s pivot complete")
                break
        else:
            print(f"No completion by sim_time={max_sim}s")

        # Like agent _reconnect_bonds: restore fixed 1-3 if physically adjacent (p2p
        # phase removes the bond created at handoff).
        pos = sim.get_positions()
        d_02 = float(np.linalg.norm(pos[0] - pos[2]))
        if d_02 < 1.12:
            sim.create_bond(0, 2)

        bm = sim.get_bond_matrix()
        has_01 = bool(bm[0, 1])
        has_02 = bool(bm[0, 2])
        has_12 = bool(bm[1, 2])
        pos = sim.get_positions()
        d_02 = float(np.linalg.norm(pos[0] - pos[2]))
        d_12 = float(np.linalg.norm(pos[1] - pos[2]))

        print("\nFinal bond adjacency (0=mod1, 1=mod2, 2=mod3):")
        print(f"  1-2 (0-1): {has_01}  |  1-3 (0-2): {has_02}  |  2-3 (1-2): {has_12}")
        print(f"  dist 1-3: {d_02:.4f}  dist 2-3: {d_12:.4f}")
        print("positions:")
        for i in range(N):
            print(f"    body {i}: {pos[i]}")

        topology_ok = has_02 and has_12 and not has_01
        metric_ok = d_02 < 1.15 and d_12 < 1.15
        ok = topology_ok and metric_ok
        print(
            f"\nLATERAL_HANDOFF_L_SHAPE: {'PASS' if ok else 'FAIL'} "
            f"(topology_ok={topology_ok}, metric_ok={metric_ok})"
        )
    finally:
        sim.disconnect()


if __name__ == "__main__":
    main()
