"""
Level-2 Vicsek fractal layout for PyBullet examples.

A 2D 4-arm Vicsek fractal in the XY plane (Z = 0).  Built recursively:

    Level 1 (+ shape):  center + 4 arms (±X, ±Y), arm length 1 → 5 modules
    Level 2:            5 copies of the + shape at scale-3 positions → 25 modules

Sub-star centers (the 5 motif centres) are all designated as **faults**,
giving a 5-fault / 20-active-module multi-fault scenario.

Layout (25 modules, 0-indexed)::

    Center sub-star  (centre 0):  0=(0,0)  1=(1,0)  2=(-1,0)  3=(0,1)  4=(0,-1)
    +X sub-star      (centre 5):  5=(3,0)  6=(4,0)  7=(2,0)   8=(3,1)  9=(3,-1)
    -X sub-star      (centre 10): 10=(-3,0) 11=(-2,0) 12=(-4,0) 13=(-3,1) 14=(-3,-1)
    +Y sub-star      (centre 15): 15=(0,3)  16=(1,3)  17=(-1,3) 18=(0,4)  19=(0,2)
    -Y sub-star      (centre 20): 20=(0,-3) 21=(1,-3) 22=(-1,-3) 23=(0,-2) 24=(0,-4)

Bonds: consecutive within each sub-star arm (center to tip), plus 4 bridge
bonds connecting centre sub-star arms to outer sub-star arms:
    M1↔M7   (centre +X tip ↔ +X sub-star -X tip)
    M2↔M11  (centre -X tip ↔ -X sub-star +X tip)
    M3↔M19  (centre +Y tip ↔ +Y sub-star -Y tip)
    M4↔M23  (centre -Y tip ↔ -Y sub-star +Y tip)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np


# 4-arm 2D directions (XY plane)
_DIRS_2D = [
    np.array([1.0, 0.0, 0.0]),   # +X
    np.array([-1.0, 0.0, 0.0]),  # -X
    np.array([0.0, 1.0, 0.0]),   # +Y
    np.array([0.0, -1.0, 0.0]),  # -Y
]


@dataclass(frozen=True)
class VicsekFaultScenario:
    n: int
    pos0: np.ndarray
    bonded0: np.ndarray
    fault_ids: List[str]
    fault_body_idxs: List[int]
    fault_adjacent: List[str]
    adjacent_map: Dict[str, int]     # {active_mid: fault_body_idx it neighbours}
    module_ids: List[str]            # active (non-fault) module IDs
    body_indices: Dict[str, int]
    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]]
    ghost_pos: Dict[str, np.ndarray]
    ghost_bonds: List[Tuple[str, str]]


def _build_substar(center: np.ndarray, start_idx: int):
    """Return (positions, intra_bonds) for a 5-module + shape."""
    positions = [center.copy()]
    bonds: List[Tuple[int, int]] = []
    for d in _DIRS_2D:
        tip_idx = start_idx + len(positions)
        positions.append(center + d)
        bonds.append((start_idx, tip_idx))
    return positions, bonds


def build_vicsek_fault_scenario() -> VicsekFaultScenario:
    """Build a level-2 4-arm 2D Vicsek fractal (25 modules, 5 faults)."""

    scale = 3.0
    substar_offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([scale, 0.0, 0.0]),
        np.array([-scale, 0.0, 0.0]),
        np.array([0.0, scale, 0.0]),
        np.array([0.0, -scale, 0.0]),
    ]

    all_positions: List[np.ndarray] = []
    all_bonds: List[Tuple[int, int]] = []
    substar_centers: List[int] = []

    for ss_offset in substar_offsets:
        start = len(all_positions)
        substar_centers.append(start)
        positions, bonds = _build_substar(ss_offset, start)
        all_positions.extend(positions)
        all_bonds.extend(bonds)

    # Bridge bonds: centre sub-star tips connect to outer sub-star tips.
    # Centre sub-star: indices 0(center), 1(+X tip), 2(-X tip), 3(+Y tip), 4(-Y tip)
    # +X sub-star:    indices 5(center), 6(+X tip), 7(-X tip), 8(+Y tip), 9(-Y tip)
    # -X sub-star:    indices 10(center), 11(+X tip), 12(-X tip), 13(+Y tip), 14(-Y tip)
    # +Y sub-star:    indices 15(center), 16(+X tip), 17(-X tip), 18(+Y tip), 19(-Y tip)
    # -Y sub-star:    indices 20(center), 21(+X tip), 22(-X tip), 23(+Y tip), 24(-Y tip)
    bridge_bonds = [
        (1, 7),    # centre +X tip (1,0,0) ↔ +X sub-star -X tip (2,0,0)
        (2, 11),   # centre -X tip (-1,0,0) ↔ -X sub-star +X tip (-2,0,0)
        (3, 19),   # centre +Y tip (0,1,0) ↔ +Y sub-star -Y tip (0,2,0)
        (4, 23),   # centre -Y tip (0,-1,0) ↔ -Y sub-star +Y tip (0,-2,0)
    ]
    all_bonds.extend(bridge_bonds)

    n = len(all_positions)
    pos0 = np.array(all_positions)
    bonded0 = np.zeros((n, n), dtype=bool)
    for i, j in all_bonds:
        bonded0[i, j] = True
        bonded0[j, i] = True

    fault_body_idxs = substar_centers  # [0, 5, 10, 15, 20]
    fault_ids = [f"M{i}" for i in fault_body_idxs]
    fault_set: Set[int] = set(fault_body_idxs)

    module_ids = [f"M{i}" for i in range(n) if i not in fault_set]
    body_indices = {f"M{i}": i for i in range(n) if i not in fault_set}

    # Fault-adjacent: each non-fault module that is bonded to any fault
    adjacent_map: Dict[str, int] = {}
    for i in range(n):
        if i in fault_set:
            continue
        mid = f"M{i}"
        for f_idx in fault_body_idxs:
            if bonded0[i, f_idx]:
                adjacent_map[mid] = f_idx
                break  # each arm module is adjacent to exactly one fault
    fault_adjacent = list(adjacent_map.keys())

    pre_damage_neighbor_slots: Dict[str, List[np.ndarray]] = {}
    for i in range(n):
        if i in fault_set:
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
    ghost_bonds = [(f"M{a}", f"M{b}") for a, b in all_bonds]

    return VicsekFaultScenario(
        n=n,
        pos0=pos0,
        bonded0=bonded0,
        fault_ids=fault_ids,
        fault_body_idxs=fault_body_idxs,
        fault_adjacent=fault_adjacent,
        adjacent_map=adjacent_map,
        module_ids=module_ids,
        body_indices=body_indices,
        pre_damage_neighbor_slots=pre_damage_neighbor_slots,
        ghost_pos=ghost_pos,
        ghost_bonds=ghost_bonds,
    )
