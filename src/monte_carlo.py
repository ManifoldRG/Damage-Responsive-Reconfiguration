"""
Monte Carlo Simulation Framework for Damage Response Algorithms

This module provides tools for running Monte Carlo simulations to evaluate
the performance of coagulation/encapsulation and reconstruction algorithms
across varying parameters (n modules, f faults).

Key metric:
- Shape Difference: diff(P, Q) = (|P| - |P ∩ Q|) / |P|
  where P, Q are sets of pairwise inter-module distances before/after damage.
"""

import copy
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from .configurations import create_random_configuration, create_random_tree_configuration
    from .udqdg_system import UDQDGSystem
except ImportError:
    from configurations import create_random_configuration, create_random_tree_configuration
    from udqdg_system import UDQDGSystem


# Configuration generation modes
CONFIG_MODE_RANDOM = "random"  # Random walk with full connectivity (default)
CONFIG_MODE_TREE = "tree"      # Tree structure (no cycles)

# Fault selection modes
FAULT_MODE_RANDOM = "random"             # existing behavior: uniform random sample
FAULT_MODE_CLUSTER = "cluster"           # contiguous cluster of size n_faults
FAULT_MODE_RANDOM_CLUSTERS = "random_clusters"  # multiple small random clusters
FAULT_MODE_LOCALIZED = "localized"       # one big contiguous cluster (alias for cluster)


def _find_articulation_points(system) -> Set[str]:
    """Return the set of articulation points (cut vertices) in the active graph.

    A module is an articulation point if removing it increases the number
    of connected components among the remaining active modules.  Uses
    Tarjan's DFS algorithm in O(V+E).
    """
    active = [mid for mid, m in system.modules.items() if m.is_active]
    if not active:
        return set()

    adj: Dict[str, List[str]] = {mid: [] for mid in active}
    active_set = set(active)
    for mid in active:
        for nbr in system.get_neighbors(mid):
            if nbr in active_set:
                adj[mid].append(nbr)

    disc: Dict[str, int] = {}
    low: Dict[str, int] = {}
    parent: Dict[str, Optional[str]] = {}
    ap: Set[str] = set()
    timer = [0]

    def dfs(u: str) -> None:
        disc[u] = low[u] = timer[0]
        timer[0] += 1
        child_count = 0
        for v in adj[u]:
            if v not in disc:
                child_count += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] is None and child_count > 1:
                    ap.add(u)
                if parent[u] is not None and low[v] >= disc[u]:
                    ap.add(u)
            elif v != parent.get(u):
                low[u] = min(low[u], disc[v])

    for mid in active:
        if mid not in disc:
            parent[mid] = None
            dfs(mid)

    return ap


def _causes_disconnection(system, fault_ids: List[str]) -> bool:
    """Check whether removing *fault_ids* disconnects the active graph."""
    removed = set(fault_ids)
    remaining = [
        mid for mid, m in system.modules.items()
        if m.is_active and mid not in removed
    ]
    if not remaining:
        return True

    visited: Set[str] = set()
    queue = deque([remaining[0]])
    visited.add(remaining[0])
    remaining_set = set(remaining)

    while queue:
        cur = queue.popleft()
        for nbr in system.get_neighbors(cur):
            if nbr in remaining_set and nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    return len(visited) < len(remaining_set)


def _bfs_grow(system, seed_module: str, target_size: int, excluded: Set[str]) -> List[str]:
    """Grow a contiguous cluster from seed_module via BFS on system neighbors."""
    selected = [seed_module]
    selected_set = {seed_module}
    queue = deque([seed_module])

    while queue and len(selected) < target_size:
        current = queue.popleft()
        for neighbor in system.get_neighbors(current):
            if neighbor not in selected_set and neighbor not in excluded:
                selected.append(neighbor)
                selected_set.add(neighbor)
                queue.append(neighbor)
                if len(selected) >= target_size:
                    break

    return selected


_MAX_REJECTION_ATTEMPTS = 200


def select_faulty_modules(
    system, n_faults: int, seed: int, fault_mode: str = FAULT_MODE_RANDOM
) -> List[str]:
    """
    Select modules to mark as faulty, **guaranteeing** that removing them
    disconnects the active graph.

    Strategy:
      * For single faults (n_faults == 1) we restrict the candidate pool
        to articulation points (cut vertices) — modules whose removal is
        guaranteed to disconnect the graph.
      * For multi-fault modes a rejection-sampling loop selects candidates
        using the original heuristic, then checks whether the removal
        actually disconnects the graph.  If not it retries with a new
        random seed (up to ``_MAX_REJECTION_ATTEMPTS``).

    Args:
        system: UDQDGSystem instance
        n_faults: Number of faults to select
        seed: Random seed for fault selection
        fault_mode: One of 'random', 'cluster', 'random_clusters', 'localized'

    Returns:
        List of module IDs to mark as faulty
    """
    random.seed(seed)
    module_ids = list(system.modules.keys())
    n_faults = min(n_faults, len(module_ids))

    if n_faults <= 0:
        return []

    # --- Fast path for single faults: use articulation points directly ---
    if n_faults == 1:
        ap = _find_articulation_points(system)
        if fault_mode == FAULT_MODE_RANDOM:
            candidates = [mid for mid in module_ids if mid in ap]
        elif fault_mode in (FAULT_MODE_CLUSTER, FAULT_MODE_LOCALIZED):
            candidates = [mid for mid in module_ids if mid in ap]
        elif fault_mode == FAULT_MODE_RANDOM_CLUSTERS:
            candidates = [mid for mid in module_ids if mid in ap]
        else:
            raise ValueError(f"Unknown fault_mode: {fault_mode}")

        if not candidates:
            return random.sample(module_ids, 1)
        return [random.choice(candidates)]

    # --- Multi-fault: rejection sampling ---
    for attempt in range(_MAX_REJECTION_ATTEMPTS):
        rng_seed = seed + attempt * 7919
        random.seed(rng_seed)

        faults = _select_faults_inner(system, module_ids, n_faults, fault_mode)
        if _causes_disconnection(system, faults):
            return faults

    # Fallback: return last attempt even if it didn't disconnect
    return faults


def _select_faults_inner(
    system, module_ids: List[str], n_faults: int, fault_mode: str
) -> List[str]:
    """Core selection logic (no disconnection guarantee)."""
    if fault_mode == FAULT_MODE_RANDOM:
        return random.sample(module_ids, n_faults)

    elif fault_mode in (FAULT_MODE_CLUSTER, FAULT_MODE_LOCALIZED):
        seed_module = random.choice(module_ids)
        return _bfs_grow(system, seed_module, n_faults, excluded=set())

    elif fault_mode == FAULT_MODE_RANDOM_CLUSTERS:
        selected: List[str] = []
        selected_set: Set[str] = set()
        remaining = n_faults

        while remaining > 0:
            cluster_size = min(random.randint(2, 5), remaining)
            available = [m for m in module_ids if m not in selected_set]
            if not available:
                break
            seed_module = random.choice(available)
            cluster = _bfs_grow(system, seed_module, cluster_size, excluded=selected_set)
            selected.extend(cluster)
            selected_set.update(cluster)
            remaining -= len(cluster)

        return selected

    else:
        raise ValueError(f"Unknown fault_mode: {fault_mode}")


@dataclass
class TrialResult:
    """Results from a single Monte Carlo trial."""
    trial_id: int
    n_modules: int
    n_faults: int
    seed: int                   # For reproducibility (structure + fault selection)

    # Algorithm outcome
    restored: bool              # True if structure reconnected after damage
    phase1_moves: int
    phase2_moves: int

    # Similarity metrics (None if not restored)
    shape_difference: Optional[float]     # diff(P, Q) after both phases - None if failed
    shape_difference_phase1: Optional[float] = None  # diff(P, Q) after phase 1 only

    # Additional metrics
    phase1_iterations: int = 0       # steps (iterations) to reconnection
    total_moves: int = 0             # phase1_moves + phase2_moves
    token_transmissions: int = 0     # total token transmissions from both phases

    # Fault mode metadata
    fault_mode: str = FAULT_MODE_RANDOM

    # Token selection strategy (ablation study)
    token_strategy: str = "furthest"

    # Safety radius for is_movable() hop check (ablation study)
    safety_radius: int = 2

    # Reconstruction method
    reconstruction_method: str = "token"


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation."""
    n_modules: int
    n_faults: int
    n_trials: int
    n_meaningful_trials: int    # Trials where fault caused disconnection (phase1_moves > 0)

    # Aggregated metrics (only from meaningful, successful trials)
    mean_shape_difference: float
    std_shape_difference: float

    # Success rates (relative to meaningful trials only)
    reconnection_rate: float    # % of meaningful trials that reconnected
    full_restoration_rate: float  # % of meaningful trials where all modules restored

    # Move statistics (only from meaningful trials)
    mean_phase1_moves: float
    mean_phase2_moves: float

    # Shape difference after phase 1 only (before restructuring)
    mean_shape_difference_phase1: float = float('nan')
    std_shape_difference_phase1: float = float('nan')

    # Steps to reconnection
    mean_steps_to_reconnection: float = float('nan')
    std_steps_to_reconnection: float = float('nan')

    # Total moves (combined)
    mean_total_moves: float = float('nan')
    std_total_moves: float = float('nan')

    # Token transmissions
    mean_token_transmissions: float = float('nan')
    std_token_transmissions: float = float('nan')

    # Reconnection rate std
    std_reconnection_rate: float = float('nan')

    # Token selection strategy (ablation study)
    token_strategy: str = "furthest"

    # Safety radius for is_movable() hop check (ablation study)
    safety_radius: int = 2

    # Reconstruction method
    reconstruction_method: str = "token"

    # Raw data
    trials: List[TrialResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_modules": self.n_modules,
            "n_faults": self.n_faults,
            "n_trials": self.n_trials,
            "n_meaningful_trials": self.n_meaningful_trials,
            "mean_shape_difference": self.mean_shape_difference,
            "std_shape_difference": self.std_shape_difference,
            "mean_shape_difference_phase1": self.mean_shape_difference_phase1,
            "std_shape_difference_phase1": self.std_shape_difference_phase1,
            "reconnection_rate": self.reconnection_rate,
            "full_restoration_rate": self.full_restoration_rate,
            "mean_phase1_moves": self.mean_phase1_moves,
            "mean_phase2_moves": self.mean_phase2_moves,
            "mean_steps_to_reconnection": self.mean_steps_to_reconnection,
            "std_steps_to_reconnection": self.std_steps_to_reconnection,
            "mean_total_moves": self.mean_total_moves,
            "std_total_moves": self.std_total_moves,
            "mean_token_transmissions": self.mean_token_transmissions,
            "std_token_transmissions": self.std_token_transmissions,
            "std_reconnection_rate": self.std_reconnection_rate,
            "token_strategy": self.token_strategy,
            "safety_radius": self.safety_radius,
            "reconstruction_method": self.reconstruction_method,
        }


def calculate_shape_difference(
    original_positions: Dict[str, np.ndarray],
    final_positions: Dict[str, np.ndarray],
    faulty_modules: Set[str],
    tolerance: float = 0.05,
) -> float:
    """
    Calculate shape difference using pairwise inter-module distances.

    Per the paper's metric: diff(P, Q) = (|P| - |P ∩ Q|) / |P|
    where P and Q are multisets of pairwise distances before/after damage.

    Uses tolerance-based matching: a distance d_p in P is considered
    matched if any unmatched distance d_q in Q satisfies
    |d_p - d_q| <= tolerance.  This avoids false mismatches from
    floating-point drift in physics-based simulators.

    Args:
        original_positions: Positions of ALL modules before fault
        final_positions: Positions of surviving modules after algorithm
        faulty_modules: Set of module IDs that were marked as faulty
        tolerance: Maximum absolute difference for two distances to be
            considered equal (default 0.05, roughly 5% of unit spacing)

    Returns:
        Shape difference diff(P, Q), anchored to pre-damage shape P.
    """
    active_orig = {
        mid: pos for mid, pos in original_positions.items()
        if mid not in faulty_modules
    }

    P_dists: List[float] = []
    orig_ids = list(active_orig.keys())
    for i, u in enumerate(orig_ids):
        for v in orig_ids[i + 1:]:
            P_dists.append(float(np.linalg.norm(active_orig[u] - active_orig[v])))

    Q_dists: List[float] = []
    final_ids = list(final_positions.keys())
    for i, u in enumerate(final_ids):
        for v in final_ids[i + 1:]:
            Q_dists.append(float(np.linalg.norm(final_positions[u] - final_positions[v])))

    if len(P_dists) == 0:
        return 0.0

    P_dists.sort()
    Q_dists.sort()

    matched = 0
    q_idx = 0
    q_used = [False] * len(Q_dists)

    for d_p in P_dists:
        while q_idx < len(Q_dists) and Q_dists[q_idx] < d_p - tolerance:
            q_idx += 1
        for j in range(q_idx, len(Q_dists)):
            if Q_dists[j] > d_p + tolerance:
                break
            if not q_used[j]:
                q_used[j] = True
                matched += 1
                break

    return (len(P_dists) - matched) / len(P_dists)


def run_single_graph_trial(
    n_modules: int,
    n_faults: int,
    seed: int,
    trial_id: int = 0,
    mode_2d: bool = False,
    fully_connected: bool = True,
    config_mode: str = CONFIG_MODE_RANDOM,
    fault_mode: str = FAULT_MODE_RANDOM,
    *,
    temperature: float = 0.5,
    pivot_exclusion_radius: int = 4,
    dt: float = 0.1,
    restructuring_method: str = "displacement",
    token_strategy: str = "furthest",
    safety_radius: int = 2,
    use_flood_echo: bool = False,
    token_gen_interval: float = 0.1,
    max_moves_per_module: int = 10,
    use_position_history: bool = True,
) -> TrialResult:
    """Execute one graph-based Monte Carlo trial with the decentralized agent
    policy on top of ``GraphSimulator`` (perfect kinematic pivots, no physics).

    Generates a structure, finds a fault set that disconnects the active
    graph (up to 500 retries), pre-marks all faults, builds a
    ``GraphSimulator``, and delegates the phase loop to
    ``src.mc_runner.run_trial`` — the same shared driver used by the
    PyBullet runner. Returns a ``TrialResult``.
    """
    from .graph_sim import GraphSimulator
    from .mc_runner import run_trial, udqdg_to_scenario

    for structure_attempt in range(500):
        gen_seed = seed + structure_attempt * 9973
        if config_mode == CONFIG_MODE_TREE:
            system = create_random_tree_configuration(
                n_modules, seed=gen_seed, mode_2d=mode_2d, balanced=False)
        else:
            system = create_random_configuration(
                n_modules, seed=gen_seed, mode_2d=mode_2d,
                fully_connected=fully_connected)

        faulty_module_ids = select_faulty_modules(
            system, n_faults, gen_seed + 1000, fault_mode)

        if _causes_disconnection(system, faulty_module_ids):
            break

    original_positions = {
        mid: module.position.copy()
        for mid, module in system.modules.items()
    }
    faulty_modules_set = set(faulty_module_ids)

    for fid in faulty_module_ids:
        if system.modules[fid].is_active:
            system.mark_fault(fid)

    scenario = udqdg_to_scenario(system, faulty_module_ids)
    sim = GraphSimulator(
        scenario.n_total, scenario.pos0, scenario.bonded0,
        module_shape="sphere")

    try:
        result = run_trial(
            sim=sim,
            scenario=scenario,
            faulty_modules_set=faulty_modules_set,
            original_positions=original_positions,
            trial_id=trial_id,
            n_modules=n_modules,
            n_faults=n_faults,
            temperature=temperature,
            pivot_exclusion_radius=pivot_exclusion_radius,
            dt=dt,
            restructuring_method=restructuring_method,
            token_strategy=token_strategy,
            safety_radius=safety_radius,
            use_flood_echo=use_flood_echo,
            token_gen_interval=token_gen_interval,
            max_moves_per_module=max_moves_per_module,
            use_position_history=use_position_history,
            forward_ap_cost=0.1,
            fault_mode=fault_mode,
        )
    finally:
        sim.disconnect()

    result.seed = seed
    result.token_strategy = token_strategy
    result.safety_radius = safety_radius
    result.reconstruction_method = restructuring_method
    return result


def run_monte_carlo(
    n_modules: int,
    n_faults: int = 1,
    n_trials: int = 100,
    seed: Optional[int] = None,
    mode_2d: bool = False,
    fully_connected: bool = True,
    verbose: bool = False,
    config_mode: str = CONFIG_MODE_RANDOM,
    n_jobs: int = 1,
    fault_mode: str = FAULT_MODE_RANDOM,
    token_strategy: str = "furthest",
    safety_radius: int = 2,
    restructuring_method: str = "displacement",
    *,
    temperature: float = 0.5,
    pivot_exclusion_radius: int = 4,
    dt: float = 0.1,
    use_flood_echo: bool = False,
    token_gen_interval: float = 0.1,
    max_moves_per_module: int = 10,
    use_position_history: bool = True,
) -> MonteCarloResults:
    """Run Monte Carlo simulation using ``GraphSimulator`` + the
    decentralized agent policies.

    Each trial dispatches ``run_single_graph_trial`` with the supplied
    knobs. ``n_jobs > 1`` runs trials in parallel via joblib's
    ``Parallel``; each worker holds its own ``GraphSimulator``.

    The simulation parameter defaults mirror the configuration the
    PyBullet sweep settled on after the action-points refactor:
    ``temperature=0.5``, ``use_flood_echo=False``,
    ``max_moves_per_module=10``, ``forward_ap_cost`` baked into
    ``mc_runner.run_trial`` at 0.1. ``restructuring_method`` replaces the
    legacy ``reconstruction_method`` flag — values are ``"displacement"``
    or ``"rendezvous"`` (token-based).
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    trial_kwargs_template = dict(
        n_modules=n_modules,
        n_faults=n_faults,
        mode_2d=mode_2d,
        fully_connected=fully_connected,
        config_mode=config_mode,
        fault_mode=fault_mode,
        temperature=temperature,
        pivot_exclusion_radius=pivot_exclusion_radius,
        dt=dt,
        restructuring_method=restructuring_method,
        token_strategy=token_strategy,
        safety_radius=safety_radius,
        use_flood_echo=use_flood_echo,
        token_gen_interval=token_gen_interval,
        max_moves_per_module=max_moves_per_module,
        use_position_history=use_position_history,
    )
    trial_kwarg_list = [
        dict(seed=seed + i, trial_id=i, **trial_kwargs_template)
        for i in range(n_trials)
    ]

    if n_jobs == 1:
        trials: List[TrialResult] = []
        iterator = tqdm(trial_kwarg_list, desc=f"n={n_modules}",
                        disable=not verbose)
        for kw in iterator:
            trials.append(run_single_graph_trial(**kw))
    else:
        trials = Parallel(n_jobs=n_jobs)(
            delayed(run_single_graph_trial)(**kw)
            for kw in tqdm(trial_kwarg_list, desc=f"n={n_modules}",
                           disable=not verbose)
        )

    # Filter out trials where fault didn't cause disconnection (phase1_moves = 0)
    # These are not meaningful tests of the reconnection algorithm
    meaningful_trials = [t for t in trials if t.phase1_moves > 0]

    # Aggregate results - only include successful meaningful trials for metrics
    successful_trials = [t for t in meaningful_trials if t.restored]
    shape_diffs = [t.shape_difference for t in successful_trials]

    # Rates are calculated relative to meaningful trials only
    n_meaningful = len(meaningful_trials)
    restored_count = len(successful_trials)

    # Full restoration = restored with zero shape difference
    full_restored_count = sum(
        1 for t in successful_trials
        if t.shape_difference is not None and t.shape_difference == 0.0
    )

    # Handle case with no successful trials
    if successful_trials:
        mean_shape_diff = float(np.mean(shape_diffs))
        std_shape_diff = float(np.std(shape_diffs))

        shape_diffs_p1 = [t.shape_difference_phase1 for t in successful_trials
                          if t.shape_difference_phase1 is not None]
        if shape_diffs_p1:
            mean_shape_diff_p1 = float(np.mean(shape_diffs_p1))
            std_shape_diff_p1 = float(np.std(shape_diffs_p1))
        else:
            mean_shape_diff_p1 = float('nan')
            std_shape_diff_p1 = float('nan')
    else:
        mean_shape_diff = float('nan')
        std_shape_diff = float('nan')
        mean_shape_diff_p1 = float('nan')
        std_shape_diff_p1 = float('nan')

    # Calculate rates relative to meaningful trials (avoid division by zero)
    if n_meaningful > 0:
        reconnection_rate = restored_count / n_meaningful
        full_restoration_rate = full_restored_count / n_meaningful
        mean_p1_moves = float(np.mean([t.phase1_moves for t in meaningful_trials]))
        mean_p2_moves = float(np.mean([t.phase2_moves for t in meaningful_trials]))

        # New aggregated metrics
        steps = [t.phase1_iterations for t in meaningful_trials]
        total_moves_list = [t.total_moves for t in meaningful_trials]
        tokens_list = [t.token_transmissions for t in meaningful_trials]

        mean_steps = float(np.mean(steps))
        std_steps = float(np.std(steps))
        mean_total_moves = float(np.mean(total_moves_list))
        std_total_moves = float(np.std(total_moves_list))
        mean_tokens = float(np.mean(tokens_list))
        std_tokens = float(np.std(tokens_list))
        std_reconn = float(np.sqrt(reconnection_rate * (1 - reconnection_rate) / n_meaningful))
    else:
        reconnection_rate = float('nan')
        full_restoration_rate = float('nan')
        mean_p1_moves = float('nan')
        mean_p2_moves = float('nan')
        mean_steps = float('nan')
        std_steps = float('nan')
        mean_total_moves = float('nan')
        std_total_moves = float('nan')
        mean_tokens = float('nan')
        std_tokens = float('nan')
        std_reconn = float('nan')

    return MonteCarloResults(
        n_modules=n_modules,
        n_faults=n_faults,
        n_trials=n_trials,
        n_meaningful_trials=n_meaningful,
        mean_shape_difference=mean_shape_diff,
        std_shape_difference=std_shape_diff,
        mean_shape_difference_phase1=mean_shape_diff_p1,
        std_shape_difference_phase1=std_shape_diff_p1,
        reconnection_rate=reconnection_rate,
        full_restoration_rate=full_restoration_rate,
        mean_phase1_moves=mean_p1_moves,
        mean_phase2_moves=mean_p2_moves,
        mean_steps_to_reconnection=mean_steps,
        std_steps_to_reconnection=std_steps,
        mean_total_moves=mean_total_moves,
        std_total_moves=std_total_moves,
        mean_token_transmissions=mean_tokens,
        std_token_transmissions=std_tokens,
        std_reconnection_rate=std_reconn,
        token_strategy=token_strategy,
        safety_radius=safety_radius,
        reconstruction_method=restructuring_method,
        trials=meaningful_trials  # Only include meaningful trials in raw data
    )


def print_results_summary(results: MonteCarloResults) -> None:
    """Print a formatted summary of Monte Carlo results."""
    print(f"\n{'='*60}")
    print(f"Monte Carlo Results: n={results.n_modules}, f={results.n_faults}")
    print(f"{'='*60}")
    print(f"Trials: {results.n_trials}")
    print(f"\nShape Difference:")
    print(f"  After Phase 1 only: {results.mean_shape_difference_phase1:.4f} "
          f"± {results.std_shape_difference_phase1:.4f}")
    print(f"  After Phase 1 + 2:  {results.mean_shape_difference:.4f} "
          f"± {results.std_shape_difference:.4f}")
    print(f"\nSuccess Rates:")
    print(f"  Reconnection Rate: {results.reconnection_rate:.1%}"
          f" ± {results.std_reconnection_rate:.4f}")
    print(f"  Full Restoration Rate: {results.full_restoration_rate:.1%}")
    print(f"\nSteps to Reconnection:")
    print(f"  Mean: {results.mean_steps_to_reconnection:.1f}"
          f" ± {results.std_steps_to_reconnection:.1f}")
    print(f"\nMove Statistics:")
    print(f"  Mean Phase 1 Moves: {results.mean_phase1_moves:.1f}")
    print(f"  Mean Phase 2 Moves: {results.mean_phase2_moves:.1f}")
    print(f"  Mean Total Moves: {results.mean_total_moves:.1f}"
          f" ± {results.std_total_moves:.1f}")
    print(f"\nToken Transmissions:")
    print(f"  Mean: {results.mean_token_transmissions:.1f}"
          f" ± {results.std_token_transmissions:.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    print("Running Monte Carlo simulation example...")

    # Run a small simulation
    results = run_monte_carlo(
        n_modules=10,
        n_faults=1,
        n_trials=20,
        seed=42,
        verbose=True
    )

    print_results_summary(results)
