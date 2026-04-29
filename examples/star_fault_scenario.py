"""
Star-fault layout for PyBullet examples.

A central module (the fault) at the origin with 6 branches radiating along
the ±X, ±Y, ±Z axes.  Each branch has ``arm_len`` modules (default 3),
giving  6 * arm_len + 1  total bodies.  The fault is body 0; branch modules
are numbered 1..N-1 in arm-major order (+X, -X, +Y, -Y, +Z, -Z).

Layout (arm_len=3, N=19)::

    Body indices per arm:
      +X: 1, 2, 3        positions (1,0,0), (2,0,0), (3,0,0)
      -X: 4, 5, 6        positions (-1,0,0), (-2,0,0), (-3,0,0)
      +Y: 7, 8, 9        positions (0,1,0), (0,2,0), (0,3,0)
      -Y: 10, 11, 12     positions (0,-1,0), (0,-2,0), (0,-3,0)
      +Z: 13, 14, 15     positions (0,0,1), (0,0,2), (0,0,3)
      -Z: 16, 17, 18     positions (0,0,-1), (0,0,-2), (0,0,-3)

Bonds are consecutive within each arm, and the first module of each arm is
bonded to the center (fault body 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


ARM_DIRECTIONS = [
    np.array([1.0, 0.0, 0.0]),   # +X
    np.array([-1.0, 0.0, 0.0]),  # -X
    np.array([0.0, 1.0, 0.0]),   # +Y
    np.array([0.0, -1.0, 0.0]),  # -Y
    np.array([0.0, 0.0, 1.0]),   # +Z
    np.array([0.0, 0.0, -1.0]),  # -Z
]


@dataclass(frozen=True)
class StarFaultScenario:
    n: int
    arm_len: int
    pos0: np.ndarray
    bonded0: np.ndarray
    fault_id: str
    fault_body_idx: int
    fault_adjacent: List[str]
    module_ids: List[str]
    body_indices: Dict[str, int]
    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]]
    ghost_pos: Dict[str, np.ndarray]
    ghost_bonds: List[Tuple[str, str]]


def build_star_fault_scenario(
    arm_len: int = 3,
) -> StarFaultScenario:
    """Build a 6-arm star with a fault at the center.

    Args:
        arm_len: number of modules per arm (default 3 → 19 total).
    """
    if arm_len < 1:
        raise ValueError("arm_len must be >= 1")

    n = 6 * arm_len + 1
    fault_body_idx = 0

    positions: List[np.ndarray] = [np.zeros(3)]  # center
    bonds: List[Tuple[int, int]] = []

    for arm_idx, direction in enumerate(ARM_DIRECTIONS):
        arm_start = 1 + arm_idx * arm_len
        for k in range(arm_len):
            body_idx = arm_start + k
            positions.append(direction * (k + 1))
            if k == 0:
                bonds.append((fault_body_idx, body_idx))
            else:
                bonds.append((body_idx - 1, body_idx))

    pos0 = np.array(positions)
    bonded0 = np.zeros((n, n), dtype=bool)
    for i, j in bonds:
        bonded0[i, j] = True
        bonded0[j, i] = True

    fault_id = "M0"
    module_ids = [f"M{i}" for i in range(n) if i != fault_body_idx]
    body_indices = {f"M{i}": i for i in range(n) if i != fault_body_idx}

    fault_adjacent = [f"M{1 + arm_idx * arm_len}" for arm_idx in range(6)]

    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]] = {}
    for i in range(n):
        if i == fault_body_idx:
            continue
        mid = f"M{i}"
        dirs: List[np.ndarray] = []
        for j in range(n):
            if bonded0[i, j]:
                diff = pos0[j] - pos0[i]
                norm = np.linalg.norm(diff)
                if norm > 1e-9:
                    dirs.append(diff / norm)
        pre_damage_neighbor_slots[mid] = dirs

    ghost_pos = {f"M{i}": pos0[i].copy() for i in range(n)}
    ghost_bonds = [(f"M{a}", f"M{b}") for a, b in bonds]

    return StarFaultScenario(
        n=n,
        arm_len=arm_len,
        pos0=pos0,
        bonded0=bonded0,
        fault_id=fault_id,
        fault_body_idx=fault_body_idx,
        fault_adjacent=fault_adjacent,
        module_ids=module_ids,
        body_indices=body_indices,
        pre_damage_neighbor_slots=pre_damage_neighbor_slots,
        ghost_pos=ghost_pos,
        ghost_bonds=ghost_bonds,
    )
