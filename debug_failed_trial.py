"""Re-run a single failed MC trial with full DEBUG logging."""
import sys
from loguru import logger
from run_bullet_monte_carlo import run_single_bullet_trial

logger.remove()
logger.add("debug_trial.log", level="DEBUG", enqueue=False,
           format="{time:HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} | {message}")
logger.add(sys.stderr, level="INFO")

if __name__ == "__main__":
    result = run_single_bullet_trial(
        n_modules=50,
        n_faults=5,
        seed=2529,
        trial_id=3,
        mode_2d=False,
        fully_connected=True,
        config_mode="random",
        fault_mode="random",
        restructuring_method="displacement",
        token_strategy="furthest",
        safety_radius=2,
        module_shape="sphere",
    )
    print("\nResult:")
    print(f"  restored={result.restored}")
    print(f"  phase1_moves={result.phase1_moves}")
    print(f"  phase2_moves={result.phase2_moves}")
    print(f"  phase1_iterations={result.phase1_iterations}")
    print(f"  shape_difference={result.shape_difference}")
