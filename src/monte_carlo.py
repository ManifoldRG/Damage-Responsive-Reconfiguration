"""
Monte Carlo Simulation Framework for Damage Response Algorithms

This module provides tools for running Monte Carlo simulations to evaluate
the performance of coagulation/encapsulation and reconstruction algorithms
across varying parameters (n modules, f faults).

Key metric:
- Shape Difference: diff(P, Q) = (|P| - |P ∩ Q|) / |P|
  where P, Q are sets of pairwise inter-module distances before/after damage.
"""

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


def select_faulty_modules(
    system, n_faults: int, seed: int, fault_mode: str = FAULT_MODE_RANDOM
) -> List[str]:
    """
    Select modules to mark as faulty based on the fault mode.

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

    if fault_mode == FAULT_MODE_RANDOM:
        return random.sample(module_ids, n_faults)

    elif fault_mode in (FAULT_MODE_CLUSTER, FAULT_MODE_LOCALIZED):
        # Single contiguous cluster grown from a random seed module
        seed_module = random.choice(module_ids)
        return _bfs_grow(system, seed_module, n_faults, excluded=set())

    elif fault_mode == FAULT_MODE_RANDOM_CLUSTERS:
        # Partition n_faults into small clusters of size 2-5
        selected = []
        selected_set = set()
        remaining = n_faults

        while remaining > 0:
            # Pick cluster size 2-5, but don't exceed remaining
            cluster_size = min(random.randint(2, 5), remaining)

            # Pick a seed module not already selected
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
        }


def calculate_shape_difference(
    original_positions: Dict[str, np.ndarray],
    final_positions: Dict[str, np.ndarray],
    faulty_modules: Set[str]
) -> float:
    """
    Calculate shape difference using pairwise inter-module distances.

    Per the paper's metric: diff(P, Q) = (|P| - |P ∩ Q|) / |P|
    where P and Q are sets of pairwise distances before/after damage.

    Args:
        original_positions: Positions of ALL modules before fault
        final_positions: Positions of surviving modules after algorithm
        faulty_modules: Set of module IDs that were marked as faulty

    Returns:
        Shape difference diff(P, Q), anchored to pre-damage shape P.
    """
    # Build P: pairwise distances of active (non-faulty) modules before damage
    active_orig = {
        mid: pos for mid, pos in original_positions.items()
        if mid not in faulty_modules
    }

    P = set()
    orig_ids = list(active_orig.keys())
    for i, u in enumerate(orig_ids):
        for v in orig_ids[i + 1:]:
            d = round(float(np.linalg.norm(active_orig[u] - active_orig[v])), 6)
            P.add(d)

    # Build Q: pairwise distances of modules after algorithm
    Q = set()
    final_ids = list(final_positions.keys())
    for i, u in enumerate(final_ids):
        for v in final_ids[i + 1:]:
            d = round(float(np.linalg.norm(final_positions[u] - final_positions[v])), 6)
            Q.add(d)

    if len(P) == 0:
        return 0.0

    # diff(P, Q) = (|P| - |P ∩ Q|) / |P|
    intersection = P & Q
    return (len(P) - len(intersection)) / len(P)


def run_single_trial(
    n_modules: int,
    n_faults: int,
    seed: int,
    trial_id: int = 0,
    mode_2d: bool = False,
    fully_connected: bool = True,
    config_mode: str = CONFIG_MODE_RANDOM,
    fault_mode: str = FAULT_MODE_RANDOM,
    token_strategy: str = "furthest",
    safety_radius: int = 2
) -> TrialResult:
    """
    Execute a single Monte Carlo trial.

    Args:
        n_modules: Number of modules in the structure
        n_faults: Number of faults to inject
        seed: Random seed for reproducibility (structure uses seed, faults use seed+1000)
        trial_id: Identifier for this trial
        mode_2d: If True, use 2D mode
        fully_connected: If True (default), connect to all adjacent modules (more branches).
                        If False, connect to only one (chain-like). Only used for 'random' mode.
        config_mode: Configuration generation mode ('random' or 'tree')
        fault_mode: Fault selection strategy ('random', 'cluster', 'random_clusters', 'localized')

    Returns:
        TrialResult with metrics and outcomes
    """
    # Generate structure based on config mode
    if config_mode == CONFIG_MODE_TREE:
        system = create_random_tree_configuration(
            n_modules, seed=seed, mode_2d=mode_2d, balanced=False
        )
    else:
        system = create_random_configuration(
            n_modules, seed=seed, mode_2d=mode_2d, fully_connected=fully_connected
        )

    # Record original positions
    original_positions = {
        mid: module.position.copy()
        for mid, module in system.modules.items()
    }

    # Select fault modules based on fault mode
    faulty_module_ids = select_faulty_modules(system, n_faults, seed + 1000, fault_mode)
    faulty_modules_set = set(faulty_module_ids)

    # Run full damage response for each fault
    phase1_moves = 0
    phase2_moves = 0
    phase1_iterations = 0
    token_transmissions = 0
    restored = True  # Assume success until a fault fails to reconnect
    post_phase1_positions = None  # Snapshot after last fault's phase 1

    for fault_id in faulty_module_ids:
        # Skip if module is already inactive (from previous fault)
        if not system.modules[fault_id].is_active:
            continue

        # Run damage response
        result = system.full_damage_response(
            fault_module_id=fault_id,
            restore_positions=True,
            max_phase1_iterations=1000,
            max_phase2_iterations=1000,
            token_strategy=token_strategy,
            safety_radius=safety_radius
        )

        # Accumulate stats
        phase1_stats = result.get('phase1', {})
        phase2_stats = result.get('phase2', {})

        phase1_moves += phase1_stats.get('total_moves', 0)
        phase1_iterations += phase1_stats.get('iterations', 0)
        token_transmissions += phase1_stats.get('token_transmissions', 0)
        restored = restored and phase1_stats.get('reconnected', False)

        # Keep the post-phase1 snapshot from the last fault processed
        if result.get('post_phase1_positions'):
            post_phase1_positions = result['post_phase1_positions']

        if phase2_stats:
            phase2_moves += phase2_stats.get('restoration_moves', 0)
            token_transmissions += phase2_stats.get('token_transmissions', 0)

    # Only calculate metrics for successful trials
    if restored:
        # Get final positions of active modules
        final_positions = {
            mid: module.position.copy()
            for mid, module in system.modules.items()
            if module.is_active
        }

        # Calculate shape difference (after both phases)
        shape_diff = calculate_shape_difference(
            original_positions, final_positions, faulty_modules_set
        )

        # Calculate shape difference after phase 1 only (before restructuring)
        if post_phase1_positions:
            shape_diff_phase1 = calculate_shape_difference(
                original_positions, post_phase1_positions, faulty_modules_set
            )
        else:
            shape_diff_phase1 = shape_diff
    else:
        # Failed trial - metrics are None
        shape_diff = None
        shape_diff_phase1 = None

    return TrialResult(
        trial_id=trial_id,
        n_modules=n_modules,
        n_faults=n_faults,
        seed=seed,
        restored=restored,
        phase1_moves=phase1_moves,
        phase2_moves=phase2_moves,
        shape_difference=shape_diff,
        shape_difference_phase1=shape_diff_phase1,
        phase1_iterations=phase1_iterations,
        total_moves=phase1_moves + phase2_moves,
        token_transmissions=token_transmissions,
        fault_mode=fault_mode,
        token_strategy=token_strategy,
        safety_radius=safety_radius,
    )


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
    safety_radius: int = 2
) -> MonteCarloResults:
    """
    Run Monte Carlo simulation with given parameters.

    Args:
        n_modules: Number of modules in each structure
        n_faults: Number of faults to inject per trial
        n_trials: Number of trials to run
        seed: Base random seed (None for random)
        mode_2d: If True, use 2D mode
        fully_connected: If True (default), connect to all adjacent modules (more branches).
                        If False, connect to only one (chain-like). Only for 'random' mode.
        verbose: If True, print progress
        config_mode: Configuration generation mode ('random' or 'tree')
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)
        fault_mode: Fault selection strategy ('random', 'cluster', 'random_clusters', 'localized')

    Returns:
        MonteCarloResults with aggregated statistics
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    trial_args = [
        (n_modules, n_faults, seed + i, i, mode_2d, fully_connected, config_mode, fault_mode, token_strategy, safety_radius)
        for i in range(n_trials)
    ]

    if n_jobs == 1:
        # Sequential execution
        trials: List[TrialResult] = []
        iterator = tqdm(trial_args, desc=f"n={n_modules}", disable=not verbose)
        for args in iterator:
            result = run_single_trial(*args)
            trials.append(result)
    else:
        # Parallel execution
        trials = Parallel(n_jobs=n_jobs)(
            delayed(run_single_trial)(*args)
            for args in tqdm(trial_args, desc=f"n={n_modules}", disable=not verbose)
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
        trials=meaningful_trials  # Only include meaningful trials in raw data
    )


def run_parameter_sweep(
    n_range: Tuple[int, int],
    f_range: Tuple[int, int] = (1, 1),
    n_trials: int = 100,
    seed: Optional[int] = None,
    mode_2d: bool = False,
    fully_connected: bool = True,
    verbose: bool = False,
    config_mode: str = CONFIG_MODE_RANDOM,
    dynamic_faults: bool = False,
    n_jobs: int = 1,
    fault_mode: str = FAULT_MODE_RANDOM
) -> Dict[Tuple[int, int], MonteCarloResults]:
    """
    Run Monte Carlo simulations across parameter ranges.

    Args:
        n_range: (min_n, max_n) inclusive range for module count
        f_range: (min_f, max_f) inclusive range for fault count (ignored if dynamic_faults=True)
        n_trials: Number of trials per configuration
        seed: Base random seed
        mode_2d: If True, use 2D mode
        fully_connected: If True (default), connect to all adjacent modules (more branches).
                        If False, connect to only one (chain-like). Only for 'random' mode.
        verbose: If True, print progress
        config_mode: Configuration generation mode ('random' or 'tree')
        dynamic_faults: If True, faults = floor(n/10) for each n (ignores f_range)
        n_jobs: Number of parallel jobs (-1 for all cores, 1 for sequential)

    Returns:
        Dictionary mapping (n, f) tuples to MonteCarloResults
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    results: Dict[Tuple[int, int], MonteCarloResults] = {}
    config_seed = seed

    n_values = list(range(n_range[0], n_range[1] + 1))
    n_iterator = tqdm(n_values, desc="Parameter sweep", disable=not verbose)

    for n in n_iterator:
        if dynamic_faults:
            # Dynamic faults: f = floor(n/10), minimum 1
            f = max(1, n // 10)
            n_iterator.set_postfix(n=n, f=f)

            result = run_monte_carlo(
                n_modules=n,
                n_faults=f,
                n_trials=n_trials,
                seed=config_seed,
                mode_2d=mode_2d,
                fully_connected=fully_connected,
                verbose=False,
                config_mode=config_mode,
                n_jobs=n_jobs,
                fault_mode=fault_mode
            )
            results[(n, f)] = result
            config_seed += n_trials
        else:
            for f in range(f_range[0], min(f_range[1] + 1, n)):  # f < n
                n_iterator.set_postfix(n=n, f=f)

                result = run_monte_carlo(
                    n_modules=n,
                    n_faults=f,
                    n_trials=n_trials,
                    seed=config_seed,
                    mode_2d=mode_2d,
                    fully_connected=fully_connected,
                    verbose=False,
                    config_mode=config_mode,
                    n_jobs=n_jobs,
                    fault_mode=fault_mode
                )
                results[(n, f)] = result
                config_seed += n_trials

    return results


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
