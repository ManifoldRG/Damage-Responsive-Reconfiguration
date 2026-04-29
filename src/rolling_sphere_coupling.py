"""
Sphere–sphere rolling coupling for PyBullet (no built-in rolling joint).

Enforces nominal center separation 2R and **zero tangential slip** at the
contact point (normal relative motion is left to the distance stabilizer).
Several Gauss–Seidel passes per substep approximate a hard no-slip condition.
Used in place of POINT2POINT during rigid pivots when enabled on
BulletSimulator.

**Prescribed rolling** (``apply_prescribed_rolling_pair``): exact COM separation
``2R``, pivot COM on the rotation arc of ``r0`` about ``rot_axis``, with a
time smoothstep so tangential speed is zero at start and end; velocities match
sphere–sphere rolling without slip at the contact.
"""

from __future__ import annotations

import numpy as np

try:
    import pybullet as p
except ImportError:
    p = None  # type: ignore


def rodrigues_rotate_vector(
    v: np.ndarray, k: np.ndarray, angle: float
) -> np.ndarray:
    """Rotate vector ``v`` by ``angle`` (rad) about unit axis ``k`` (Rodrigues)."""
    vv = np.asarray(v, dtype=float).reshape(3)
    kk = np.asarray(k, dtype=float).reshape(3)
    kn = float(np.linalg.norm(kk))
    if kn < 1e-12:
        return vv.copy()
    kk = kk / kn
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return (
        vv * c
        + np.cross(kk, vv) * s
        + kk * float(np.dot(kk, vv)) * (1.0 - c)
    )


def prescribed_theta_and_rate(
    t_elapsed: float,
    duration: float,
    target_angle: float,
) -> tuple[float, float]:
    """Smoothstep-in-time: θ = s(t/T) θ_tgt, zero dθ/dt at t=0 and t=T."""
    T = max(float(duration), 1e-9)
    u = float(np.clip(t_elapsed / T, 0.0, 1.0))
    s = u * u * (3.0 - 2.0 * u)
    s_dot = (6.0 * u * (1.0 - u)) / T
    theta = s * float(target_angle)
    theta_dot = s_dot * float(target_angle)
    return theta, theta_dot


def _skew(r: np.ndarray) -> np.ndarray:
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _J_body(r: np.ndarray) -> np.ndarray:
    """Jacobian 3x6: contact_velocity = J @ [v; omega] = v + omega cross r."""
    return np.hstack([np.eye(3), -_skew(r)])


def _Minv_sphere(m: float, I: float) -> np.ndarray:
    inv_m = 1.0 / m
    inv_I = 1.0 / I
    out = np.zeros((6, 6), dtype=float)
    out[0:3, 0:3] = inv_m * np.eye(3)
    out[3:6, 3:6] = inv_I * np.eye(3)
    return out


def contact_velocities(
    v_a: np.ndarray,
    w_a: np.ndarray,
    v_b: np.ndarray,
    w_b: np.ndarray,
    r_a: np.ndarray,
    r_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return L_a, L_b, L_a - L_b (world frame)."""
    L_a = v_a + np.cross(w_a, r_a)
    L_b = v_b + np.cross(w_b, r_b)
    return L_a, L_b, L_a - L_b


def tangential_slip_norm(
    x_a: np.ndarray,
    x_b: np.ndarray,
    v_a: np.ndarray,
    w_a: np.ndarray,
    v_b: np.ndarray,
    w_b: np.ndarray,
    sphere_radius: float,
) -> float:
    """|| (v_slip) - n (n·v_slip) || with contact normal n from COM line."""
    d = x_b - x_a
    dist = float(np.linalg.norm(d))
    if dist < 1e-12:
        return 0.0
    n = d / dist
    r_a = sphere_radius * n
    r_b = -sphere_radius * n
    _, _, v_slip = contact_velocities(v_a, w_a, v_b, w_b, r_a, r_b)
    vt = v_slip - np.dot(v_slip, n) * n
    return float(np.linalg.norm(vt))


def stabilize_com_distance(
    physics_client_id: int,
    body_a: int,
    body_b: int,
    sphere_radius: float,
    *,
    beta: float = 0.25,
    max_correction_per_body: float = 0.02,
) -> None:
    """Baumgarte-style correction so COM separation tends to 2R."""
    if p is None:
        raise ImportError("pybullet is required")

    pos_a, orn_a = p.getBasePositionAndOrientation(body_a, physicsClientId=physics_client_id)
    pos_b, orn_b = p.getBasePositionAndOrientation(body_b, physicsClientId=physics_client_id)
    x_a = np.array(pos_a, dtype=float)
    x_b = np.array(pos_b, dtype=float)
    d = x_b - x_a
    dist = float(np.linalg.norm(d))
    target = 2.0 * sphere_radius
    if dist < 1e-9:
        return
    n = d / dist
    err = dist - target
    corr = beta * err
    if corr > max_correction_per_body:
        corr = max_correction_per_body
    if corr < -max_correction_per_body:
        corr = -max_correction_per_body
    half = 0.5 * corr
    # Move A along +n and B along -n so separation moves toward 2R
    x_a_new = (x_a + half * n).tolist()
    x_b_new = (x_b - half * n).tolist()
    p.resetBasePositionAndOrientation(
        body_a, x_a_new, orn_a, physicsClientId=physics_client_id
    )
    p.resetBasePositionAndOrientation(
        body_b, x_b_new, orn_b, physicsClientId=physics_client_id
    )


def apply_no_slip_velocity_impulse(
    physics_client_id: int,
    body_a: int,
    body_b: int,
    sphere_radius: float,
    mass: float,
    inertia_scalar: float,
    *,
    max_lambda_norm: float = 20000.0,
    rcond: float = 1e-10,
    ridge: float = 5e-5,
) -> float:
    """
    Remove relative **tangential** slip at the contact point (one Gauss–Seidel pass).

    Only the component of slip perpendicular to the contact normal is targeted;
    center separation along ``n`` is handled by ``stabilize_com_distance``. Returns
    ||v_slip_t|| before correction (tangential magnitude).
    """
    if p is None:
        raise ImportError("pybullet is required")

    pos_a, _ = p.getBasePositionAndOrientation(body_a, physicsClientId=physics_client_id)
    pos_b, _ = p.getBasePositionAndOrientation(body_b, physicsClientId=physics_client_id)
    x_a = np.array(pos_a, dtype=float)
    x_b = np.array(pos_b, dtype=float)
    d = x_b - x_a
    dist = float(np.linalg.norm(d))
    if dist < 1e-9:
        return 0.0
    n = d / dist
    r_a = sphere_radius * n
    r_b = -sphere_radius * n

    v_a, w_a = p.getBaseVelocity(body_a, physicsClientId=physics_client_id)
    v_b, w_b = p.getBaseVelocity(body_b, physicsClientId=physics_client_id)
    v_a = np.array(v_a, dtype=float)
    w_a = np.array(w_a, dtype=float)
    v_b = np.array(v_b, dtype=float)
    w_b = np.array(w_b, dtype=float)

    _, _, v_slip = contact_velocities(v_a, w_a, v_b, w_b, r_a, r_b)
    # Rolling: enforce no slip in the tangent plane only
    v_slip_t = v_slip - np.dot(v_slip, n) * n
    slip_before = float(np.linalg.norm(v_slip_t))
    if slip_before < 1e-10:
        return slip_before

    J_a = _J_body(r_a)
    J_b = _J_body(r_b)
    Minv = _Minv_sphere(mass, inertia_scalar)
    K = J_a @ Minv @ J_a.T + J_b @ Minv @ J_b.T + ridge * np.eye(3)
    rhs = -v_slip_t
    try:
        lam = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        lam = np.linalg.lstsq(K, rhs, rcond=rcond)[0]

    ln = float(np.linalg.norm(lam))
    if ln > max_lambda_norm and ln > 1e-12:
        lam *= max_lambda_norm / ln

    dq_a = Minv @ J_a.T @ lam
    dq_b = -Minv @ J_b.T @ lam
    dv_a = dq_a[0:3]
    dw_a = dq_a[3:6]
    dv_b = dq_b[0:3]
    dw_b = dq_b[3:6]

    v_a_n = (v_a + dv_a).tolist()
    w_a_n = (w_a + dw_a).tolist()
    v_b_n = (v_b + dv_b).tolist()
    w_b_n = (w_b + dw_b).tolist()

    p.resetBaseVelocity(
        body_a, v_a_n, w_a_n, physicsClientId=physics_client_id
    )
    p.resetBaseVelocity(
        body_b, v_b_n, w_b_n, physicsClientId=physics_client_id
    )

    return slip_before


def apply_rolling_sphere_pair(
    physics_client_id: int,
    body_a: int,
    body_b: int,
    sphere_radius: float,
    mass: float,
    inertia_scalar: float,
    *,
    distance_beta: float = 0.35,
    max_position_correction: float = 0.02,
    max_lambda_norm: float = 20000.0,
    no_slip_iterations: int = 16,
    conserve_pair_linear_momentum: bool = True,
) -> None:
    """Stabilize 2R separation then several no-slip passes (one physics substep).

    No-slip Gauss–Seidel impulses do not conserve total linear momentum of the
    pair; by default we restore the pair's pre-pass linear momentum by splitting
    the spurious delta equally on both bodies (equal masses).
    """
    v_a0, _ = p.getBaseVelocity(body_a, physicsClientId=physics_client_id)
    v_b0, _ = p.getBaseVelocity(body_b, physicsClientId=physics_client_id)
    v_a0 = np.array(v_a0, dtype=float)
    v_b0 = np.array(v_b0, dtype=float)

    stabilize_com_distance(
        physics_client_id,
        body_a,
        body_b,
        sphere_radius,
        beta=distance_beta,
        max_correction_per_body=max_position_correction,
    )
    n_it = max(1, int(no_slip_iterations))
    for _ in range(n_it):
        apply_no_slip_velocity_impulse(
            physics_client_id,
            body_a,
            body_b,
            sphere_radius,
            mass,
            inertia_scalar,
            max_lambda_norm=max_lambda_norm,
        )

    if conserve_pair_linear_momentum and mass > 1e-12:
        v_a1, w_a1 = p.getBaseVelocity(body_a, physicsClientId=physics_client_id)
        v_b1, w_b1 = p.getBaseVelocity(body_b, physicsClientId=physics_client_id)
        v_a1 = np.array(v_a1, dtype=float)
        v_b1 = np.array(v_b1, dtype=float)
        d_p = mass * (v_a1 - v_a0 + v_b1 - v_b0)
        corr = d_p / (2.0 * mass)
        v_a2 = (v_a1 - corr).tolist()
        v_b2 = (v_b1 - corr).tolist()
        p.resetBaseVelocity(
            body_a, v_a2, w_a1, physicsClientId=physics_client_id
        )
        p.resetBaseVelocity(
            body_b, v_b2, w_b1, physicsClientId=physics_client_id
        )


def apply_prescribed_rolling_pair(
    physics_client_id: int,
    body_pivot: int,
    body_axis: int,
    sphere_radius: float,
    r0: np.ndarray,
    rot_axis_unit: np.ndarray,
    target_angle: float,
    start_time: float,
    sim_time: float,
    duration: float,
) -> None:
    """Place pivot on the rolling arc and set velocities for ideal no-slip.

    ``r0`` is pivot minus axis at pivot start (world). Pivot COM follows
    ``x_p = x_a + 2R hat(u)`` where ``u`` is ``r0`` rotated about ``rot_axis_unit``
    by ``theta(t)``; ``theta`` uses a cubic smoothstep in time so ``theta_dot``
    is zero at start and end. Rolling angular velocity satisfies no slip at the
    contact with the axis sphere.
    """
    if p is None:
        raise ImportError("pybullet is required")

    t_elapsed = float(sim_time - start_time)
    theta, theta_dot = prescribed_theta_and_rate(
        t_elapsed, duration, target_angle)

    r0v = np.asarray(r0, dtype=float).reshape(3)
    omega_hat = np.asarray(rot_axis_unit, dtype=float).reshape(3)
    on = float(np.linalg.norm(omega_hat))
    if on > 1e-12:
        omega_hat = omega_hat / on

    pos_a, _ = p.getBasePositionAndOrientation(
        body_axis, physicsClientId=physics_client_id)
    orn_p = p.getBasePositionAndOrientation(
        body_pivot, physicsClientId=physics_client_id)[1]

    x_a = np.array(pos_a, dtype=float)
    v_a, w_a = p.getBaseVelocity(body_axis, physicsClientId=physics_client_id)
    v_a = np.array(v_a, dtype=float)
    w_a = np.array(w_a, dtype=float)

    r_rot = rodrigues_rotate_vector(r0v, omega_hat, theta)
    rn = float(np.linalg.norm(r_rot))
    if rn < 1e-12:
        return
    u_hat = r_rot / rn
    target_sep = 2.0 * float(sphere_radius)
    x_p = x_a + target_sep * u_hat

    v_p = v_a + target_sep * np.cross(omega_hat, u_hat) * theta_dot

    w = (v_p - v_a - float(sphere_radius) * np.cross(w_a, u_hat)) / float(
        sphere_radius
    )
    w_p = np.cross(u_hat, w)

    p.resetBasePositionAndOrientation(
        body_pivot, x_p.tolist(), orn_p, physicsClientId=physics_client_id
    )
    p.resetBaseVelocity(
        body_pivot,
        v_p.tolist(),
        w_p.tolist(),
        physicsClientId=physics_client_id,
    )
