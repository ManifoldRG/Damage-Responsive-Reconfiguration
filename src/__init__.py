
# UDQDG System - Unit Dual Quaternion Directed Graph Framework
from .dual_quaternion import UnitDualQuaternion, LATTICE_DIRECTIONS
from .udqdg_system import UDQDGSystem, SphericalModule

# Optional visualization imports (Manim-based, may not be available)
try:
    from .visualization import UDQDG3DVisualizer
    from .animation_engine import AnimationEngine, PivotAnimation, PivotType
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

# Legacy imports for compatibility (may be in archive/)
_HAS_LEGACY = False
try:
    from .module import Module, Direction, ConnectionPort, Message
    from .modular_system import ModularSystem, GraphState
    from .test_bench import (
        TestBench,
        TestScenario,
        TestResult,
        AlgorithmResult,
        ReconfigurationAlgorithm,
        BaseReconfigurationAlgorithm
    )
    _HAS_LEGACY = True
except ImportError:
    pass

__version__ = "0.3.0"

# Build __all__ dynamically based on available modules
__all__ = [
    # UDQDG Framework (always available)
    "UnitDualQuaternion",
    "LATTICE_DIRECTIONS",
    "UDQDGSystem",
    "SphericalModule",
]

# Add visualization exports if available
if _HAS_VISUALIZATION:
    __all__.extend([
        "UDQDG3DVisualizer",
        "AnimationEngine",
        "PivotAnimation",
        "PivotType",
    ])

# Add legacy exports if available
if _HAS_LEGACY:
    __all__.extend([
        "Module",
        "Direction",
        "ConnectionPort",
        "Message",
        "ModularSystem",
        "GraphState",
        "TestBench",
        "TestScenario",
        "TestResult",
        "AlgorithmResult",
        "ReconfigurationAlgorithm",
        "BaseReconfigurationAlgorithm"
    ])