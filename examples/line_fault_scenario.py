"""
Shared Y-axis line layout for PyBullet line-fault examples.

Modules M0..M(n-1) at (0, i, 0); consecutive bonds. One body is the passive
fault (still in the world for topology); agents exclude that module ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class LineFaultScenario:
    n: int
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


def build_y_line_fault_scenario(
    n: int = 11,
    fault_body_idx: int | None = None,
) -> LineFaultScenario:
    """
    Build an ``n``-module line along +Y with a fault at ``fault_body_idx``.

    If ``fault_body_idx`` is None, uses ``n // 2`` (center for odd ``n``).
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if fault_body_idx is None:
        fault_body_idx = n // 2
    if fault_body_idx < 0 or fault_body_idx >= n:
        raise ValueError(f"fault_body_idx must be in [0, {n - 1}]")

    pos0 = np.array([[0.0, float(i), 0.0] for i in range(n)])
    bonded0 = np.zeros((n, n), dtype=bool)
    for i in range(n - 1):
        bonded0[i, i + 1] = True
        bonded0[i + 1, i] = True

    fault_id = f"M{fault_body_idx}"
    module_ids = [f"M{i}" for i in range(n) if i != fault_body_idx]
    body_indices = {f"M{i}": i for i in range(n) if i != fault_body_idx}

    fault_adjacent: List[str] = []
    if fault_body_idx > 0:
        fault_adjacent.append(f"M{fault_body_idx - 1}")
    if fault_body_idx < n - 1:
        fault_adjacent.append(f"M{fault_body_idx + 1}")

    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]] = {}
    for i in range(n):
        if i == fault_body_idx:
            continue
        mid = f"M{i}"
        dirs: List[np.ndarray] = []
        if i > 0:
            dirs.append(np.array([0.0, -1.0, 0.0]))
        if i < n - 1:
            dirs.append(np.array([0.0, 1.0, 0.0]))
        pre_damage_neighbor_slots[mid] = dirs

    ghost_pos = {f"M{i}": np.array([0.0, float(i), 0.0]) for i in range(n)}
    ghost_bonds = [(f"M{i}", f"M{i + 1}") for i in range(n - 1)]

    return LineFaultScenario(
        n=n,
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
