"""
Monte Carlo Simulation Framework for Damage Response Algorithms

This module provides tools for running Monte Carlo simulations to evaluate
the performance of coagulation/encapsulation and reconstruction algorithms
across varying parameters (n modules, f faults).

Key metrics:
- Total Difference (symmetric): |P_start Δ P_end| / n
- Missing Portions (asymmetric): |P_start - P_end| / n
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

try:
    from .configurations import create_random_configuration, create_random_tree_configuration
    from .udqdg_system import UDQDGSystem
except ImportError:
    from configurations import create_random_configuration, create_random_tree_configuration
    from udqdg_system import UDQDGSystem


# Configuration generation modes
CONFIG_MODE_RANDOM = "random"  # Random walk with full connectivity (default)
CONFIG_MODE_TREE = "tree"      # Tree structure (no cycles)


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
    total_difference: Optional[float]     # Metric 1 (symmetric) - None if failed
    missing_portions: Optional[float]     # Metric 2 (asymmetric) - None if failed


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation."""
    n_modules: int
    n_faults: int
    n_trials: int
    n_meaningful_trials: int    # Trials where fault caused disconnection (phase1_moves > 0)

    # Aggregated metrics (only from meaningful, successful trials)
    mean_total_difference: float
    std_total_difference: float
    mean_missing_portions: float
    std_missing_portions: float

    # Success rates (relative to meaningful trials only)
    reconnection_rate: float    # % of meaningful trials that reconnected
    full_restoration_rate: float  # % of meaningful trials where all modules restored

    # Move statistics (only from meaningful trials)
    mean_phase1_moves: float
    mean_phase2_moves: float

    # Raw data
    trials: List[TrialResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "n_modules": self.n_modules,
            "n_faults": self.n_faults,
            "n_trials": self.n_trials,
            "n_meaningful_trials": self.n_meaningful_trials,
            "mean_total_difference": self.mean_total_difference,
            "std_total_difference": self.std_total_difference,
            "mean_missing_portions": self.mean_missing_portions,
            "std_missing_portions": self.std_missing_portions,
            "reconnection_rate": self.reconnection_rate,
            "full_restoration_rate": self.full_restoration_rate,
            "mean_phase1_moves": self.mean_phase1_moves,
            "mean_phase2_moves": self.mean_phase2_moves,
        }


def calculate_similarity_metrics(
    original_positions: Dict[str, np.ndarray],
    final_positions: Dict[str, np.ndarray],
    faulty_modules: Set[str]
) -> Tuple[float, float]:
    """
    Calculate similarity metrics between original and final positions.

    Args:
        original_positions: Positions of ALL modules before fault
        final_positions: Positions of surviving modules after algorithm
        faulty_modules: Set of module IDs that were marked as faulty

    Returns:
        Tuple of (total_difference, missing_portions):
        - total_difference: |P_start Δ P_end| / n (symmetric difference)
        - missing_portions: |P_start - P_end| / n (missing only)
    """
    n = len(original_positions)  # Original total
    if n == 0:
        return 0.0, 0.0

    # Get position sets (excluding faulty modules from start)
    start_positions = {
        tuple(pos.astype(int)) for mid, pos in original_positions.items()
        if mid not in faulty_modules
    }
    end_positions = {
        tuple(pos.astype(int)) for pos in final_positions.values()
    }

    # Metric 1: Symmetric difference (both missing AND excess)
    symmetric_diff = start_positions.symmetric_difference(end_positions)
    total_difference = len(symmetric_diff) / n

    # Metric 2: Missing only (in start but not in end)
    missing = start_positions - end_positions
    missing_portions = len(missing) / n

    return total_difference, missing_portions


def run_single_trial(
    n_modules: int,
    n_faults: int,
    seed: int,
    trial_id: int = 0,
    mode_2d: bool = False,
    fully_connected: bool = True,
    config_mode: str = CONFIG_MODE_RANDOM
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

    # Select random fault modules
    random.seed(seed + 1000)  # Different seed for fault selection
    module_ids = list(system.modules.keys())
    faulty_module_ids = random.sample(module_ids, min(n_faults, len(module_ids)))
    faulty_modules_set = set(faulty_module_ids)

    # Run full damage response for each fault
    phase1_moves = 0
    phase2_moves = 0
    restored = True  # Assume success until a fault fails to reconnect

    for fault_id in faulty_module_ids:
        # Skip if module is already inactive (from previous fault)
        if not system.modules[fault_id].is_active:
            continue

        # Run damage response
        result = system.full_damage_response(
            fault_module_id=fault_id,
            restore_positions=True,
            max_phase1_iterations=100,
            max_phase2_iterations=100
        )

        # Accumulate stats
        phase1_stats = result.get('phase1', {})
        phase2_stats = result.get('phase2', {})

        phase1_moves += phase1_stats.get('total_moves', 0)
        restored = restored and phase1_stats.get('reconnected', False)

        if phase2_stats:
            phase2_moves += phase2_stats.get('restoration_moves', 0)

    # Only calculate metrics for successful trials
    if restored:
        # Get final positions of active modules
        final_positions = {
            mid: module.position.copy()
            for mid, module in system.modules.items()
            if module.is_active
        }

        # Calculate similarity metrics
        total_diff, missing = calculate_similarity_metrics(
            original_positions, final_positions, faulty_modules_set
        )
    else:
        # Failed trial - metrics are None
        total_diff = None
        missing = None

    return TrialResult(
        trial_id=trial_id,
        n_modules=n_modules,
        n_faults=n_faults,
        seed=seed,
        restored=restored,
        phase1_moves=phase1_moves,
        phase2_moves=phase2_moves,
        total_difference=total_diff,
        missing_portions=missing,
    )


def run_monte_carlo(
    n_modules: int,
    n_faults: int = 1,
    n_trials: int = 100,
    seed: Optional[int] = None,
    mode_2d: bool = False,
    fully_connected: bool = True,
    verbose: bool = False,
    config_mode: str = CONFIG_MODE_RANDOM
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

    Returns:
        MonteCarloResults with aggregated statistics
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    trials: List[TrialResult] = []

    for i in range(n_trials):
        trial_seed = seed + i
        if verbose and i % 10 == 0:
            print(f"Running trial {i}/{n_trials}...")

        result = run_single_trial(
            n_modules=n_modules,
            n_faults=n_faults,
            seed=trial_seed,
            trial_id=i,
            mode_2d=mode_2d,
            fully_connected=fully_connected,
            config_mode=config_mode
        )
        trials.append(result)

    # Filter out trials where fault didn't cause disconnection (phase1_moves = 0)
    # These are not meaningful tests of the reconnection algorithm
    meaningful_trials = [t for t in trials if t.phase1_moves > 0]

    # Aggregate results - only include successful meaningful trials for metrics
    successful_trials = [t for t in meaningful_trials if t.restored]
    total_diffs = [t.total_difference for t in successful_trials]
    missing_portions = [t.missing_portions for t in successful_trials]

    # Rates are calculated relative to meaningful trials only
    n_meaningful = len(meaningful_trials)
    restored_count = len(successful_trials)

    # Full restoration = restored with zero total difference
    full_restored_count = sum(
        1 for t in successful_trials
        if t.total_difference is not None and t.total_difference == 0.0
    )

    # Handle case with no successful trials
    if successful_trials:
        mean_total_diff = float(np.mean(total_diffs))
        std_total_diff = float(np.std(total_diffs))
        mean_missing = float(np.mean(missing_portions))
        std_missing = float(np.std(missing_portions))
    else:
        mean_total_diff = float('nan')
        std_total_diff = float('nan')
        mean_missing = float('nan')
        std_missing = float('nan')

    # Calculate rates relative to meaningful trials (avoid division by zero)
    if n_meaningful > 0:
        reconnection_rate = restored_count / n_meaningful
        full_restoration_rate = full_restored_count / n_meaningful
        mean_p1_moves = float(np.mean([t.phase1_moves for t in meaningful_trials]))
        mean_p2_moves = float(np.mean([t.phase2_moves for t in meaningful_trials]))
    else:
        reconnection_rate = float('nan')
        full_restoration_rate = float('nan')
        mean_p1_moves = float('nan')
        mean_p2_moves = float('nan')

    return MonteCarloResults(
        n_modules=n_modules,
        n_faults=n_faults,
        n_trials=n_trials,
        n_meaningful_trials=n_meaningful,
        mean_total_difference=mean_total_diff,
        std_total_difference=std_total_diff,
        mean_missing_portions=mean_missing,
        std_missing_portions=std_missing,
        reconnection_rate=reconnection_rate,
        full_restoration_rate=full_restoration_rate,
        mean_phase1_moves=mean_p1_moves,
        mean_phase2_moves=mean_p2_moves,
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
    dynamic_faults: bool = False
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

    Returns:
        Dictionary mapping (n, f) tuples to MonteCarloResults
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    results: Dict[Tuple[int, int], MonteCarloResults] = {}
    config_seed = seed

    for n in range(n_range[0], n_range[1] + 1):
        if dynamic_faults:
            # Dynamic faults: f = floor(n/10), minimum 1
            f = max(1, n // 10)
            if verbose:
                print(f"Running configuration (n={n}, f={f} [dynamic])...")

            result = run_monte_carlo(
                n_modules=n,
                n_faults=f,
                n_trials=n_trials,
                seed=config_seed,
                mode_2d=mode_2d,
                fully_connected=fully_connected,
                verbose=False,
                config_mode=config_mode
            )
            results[(n, f)] = result
            config_seed += n_trials
        else:
            for f in range(f_range[0], min(f_range[1] + 1, n)):  # f < n
                if verbose:
                    print(f"Running configuration (n={n}, f={f})...")

                result = run_monte_carlo(
                    n_modules=n,
                    n_faults=f,
                    n_trials=n_trials,
                    seed=config_seed,
                    mode_2d=mode_2d,
                    fully_connected=fully_connected,
                    verbose=False,
                    config_mode=config_mode
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
    print(f"\nSimilarity Metrics:")
    print(f"  Total Difference (symmetric): {results.mean_total_difference:.4f} "
          f"± {results.std_total_difference:.4f}")
    print(f"  Missing Portions (asymmetric): {results.mean_missing_portions:.4f} "
          f"± {results.std_missing_portions:.4f}")
    print(f"\nSuccess Rates:")
    print(f"  Reconnection Rate: {results.reconnection_rate:.1%}")
    print(f"  Full Restoration Rate: {results.full_restoration_rate:.1%}")
    print(f"\nMove Statistics:")
    print(f"  Mean Phase 1 Moves: {results.mean_phase1_moves:.1f}")
    print(f"  Mean Phase 2 Moves: {results.mean_phase2_moves:.1f}")
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
