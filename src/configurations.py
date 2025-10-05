import numpy as np
from typing import Tuple
try:
    from .udqdg_system import UDQDGSystem
    from .dual_quaternion import LATTICE_DIRECTIONS
except ImportError:
    from udqdg_system import UDQDGSystem
    from dual_quaternion import LATTICE_DIRECTIONS


def create_star_configuration(size: int = 3) -> UDQDGSystem:
    """
    Create a star-shaped configuration with a central module and radial arms.

    Args:
        size: Length of each arm from center

    Returns:
        UDQDGSystem with star configuration
    """
    system = UDQDGSystem()

    # Central module at origin
    system.add_module("C", np.array([0, 0, 0]))

    # Create 6 arms along main axes
    directions = [
        ('PX', np.array([1, 0, 0])),   # Positive X
        ('NX', np.array([-1, 0, 0])),  # Negative X
        ('PY', np.array([0, 1, 0])),   # Positive Y
        ('NY', np.array([0, -1, 0])),  # Negative Y
        ('PZ', np.array([0, 0, 1])),   # Positive Z
        ('NZ', np.array([0, 0, -1]))   # Negative Z
    ]

    for prefix, direction in directions:
        prev_module = "C"
        for i in range(1, size + 1):
            module_id = f"{prefix}{i}"  # PX1, PX2, NX1, etc.
            position = i * direction
            system.add_module(module_id, position)
            system.connect_modules(prev_module, module_id)
            prev_module = module_id

    return system


def create_tree_configuration(depth: int = 3, branching: int = 2) -> UDQDGSystem:
    """
    Create a tree-shaped configuration with branching structure.

    Args:
        depth: Height of the tree
        branching: Number of children per node (2 or 3)

    Returns:
        UDQDGSystem with tree configuration
    """
    system = UDQDGSystem()

    # Root at origin
    system.add_module("R", np.array([0, 0, 0]))

    # Available directions for branching (avoiding overlap)
    branch_directions = [
        np.array([1, 0, 0]),   # +X
        np.array([-1, 0, 0]),  # -X
        np.array([0, 1, 0]),   # +Y
        np.array([0, -1, 0]),  # -Y
        np.array([0, 0, 1]),   # +Z
    ]

    def build_tree_recursive(parent_id: str, parent_pos: np.ndarray,
                            level: int, dir_index: int = 0):
        """Recursively build tree structure."""
        if level >= depth:
            return

        # Determine which directions to use for this level
        num_children = min(branching, len(branch_directions) - dir_index)

        for i in range(num_children):
            # Calculate child position
            direction = branch_directions[(dir_index + i) % len(branch_directions)]
            child_pos = parent_pos + direction

            # Create child module
            child_id = f"{parent_id}_{i}"
            system.add_module(child_id, child_pos)
            system.connect_modules(parent_id, child_id)

            # Recurse for next level
            build_tree_recursive(child_id, child_pos, level + 1, dir_index + i + 1)

    # Build from root
    build_tree_recursive("R", np.array([0, 0, 0]), 0)

    return system


def create_line_configuration(length: int = 5, axis: str = 'X') -> UDQDGSystem:
    """
    Create a straight line of modules along specified axis.

    Args:
        length: Number of modules in the line
        axis: 'X', 'Y', or 'Z' for primary axis direction

    Returns:
        UDQDGSystem with line configuration
    """
    system = UDQDGSystem()

    # Determine direction vector
    direction_map = {
        'X': np.array([1, 0, 0]),
        'Y': np.array([0, 1, 0]),
        'Z': np.array([0, 0, 1])
    }
    direction = direction_map.get(axis.upper(), np.array([1, 0, 0]))

    # Create line of modules
    prev_module = None
    for i in range(length):
        module_id = f"M{i}"
        position = i * direction
        system.add_module(module_id, position)

        if prev_module:
            system.connect_modules(prev_module, module_id)

        prev_module = module_id

    return system


def create_helix_configuration(turns: int = 2, modules_per_turn: int = 6) -> UDQDGSystem:
    """
    Create a helical configuration (spiral staircase).

    Args:
        turns: Number of complete turns in the helix
        modules_per_turn: Modules per full rotation

    Returns:
        UDQDGSystem with helix configuration
    """
    system = UDQDGSystem()

    total_modules = turns * modules_per_turn
    prev_module = None

    for i in range(total_modules):
        angle = (i / modules_per_turn) * 2 * np.pi
        z = i  # Vertical progression

        # Discrete lattice positions on circular path
        # Approximate circle with lattice points
        x = round(2 * np.cos(angle))
        y = round(2 * np.sin(angle))

        module_id = f"H{i}"
        position = np.array([x, y, z], dtype=float)

        system.add_module(module_id, position)

        if prev_module:
            # Only connect if it's a valid unit lattice step
            if system.connect_modules(prev_module, module_id):
                pass
            else:
                # If direct connection fails, try intermediate step
                prev_pos = system.modules[prev_module].position
                # Create stepping stone if needed
                pass

        prev_module = module_id

    return system


def create_cross_configuration(arm_length: int = 3) -> UDQDGSystem:
    """
    Create a 3D cross (+ shape in XY plane with Z extension).

    Args:
        arm_length: Length of each arm

    Returns:
        UDQDGSystem with cross configuration
    """
    system = UDQDGSystem()

    # Center module
    system.add_module("C", np.array([0, 0, 0]))

    # Four arms in XY plane
    directions = [
        ('X', np.array([1, 0, 0])),
        ('X', np.array([-1, 0, 0])),
        ('Y', np.array([0, 1, 0])),
        ('Y', np.array([0, -1, 0]))
    ]

    for axis, direction in directions:
        prev_module = "C"
        for i in range(1, arm_length + 1):
            sign = '+' if direction[direction != 0][0] > 0 else '-'
            module_id = f"{axis}{sign}{i}"
            position = i * direction
            system.add_module(module_id, position)
            system.connect_modules(prev_module, module_id)
            prev_module = module_id

    # Z axis extension
    for i in range(1, arm_length + 1):
        module_id = f"Z+{i}"
        position = np.array([0, 0, i], dtype=float)
        system.add_module(module_id, position)
        if i == 1:
            system.connect_modules("C", module_id)
        else:
            system.connect_modules(f"Z+{i-1}", module_id)

    return system


def create_l_shape_configuration() -> UDQDGSystem:
    """
    Create an L-shaped configuration for testing corner pivots.

    Returns:
        UDQDGSystem with L configuration
    """
    system = UDQDGSystem()

    # Vertical part
    system.add_module("M0", np.array([0, 0, 0]))
    system.add_module("M1", np.array([0, 0, 1]))
    system.add_module("M2", np.array([0, 0, 2]))

    # Horizontal part
    system.add_module("M3", np.array([1, 0, 2]))
    system.add_module("M4", np.array([2, 0, 2]))

    # Connect
    system.connect_modules("M0", "M1")
    system.connect_modules("M1", "M2")
    system.connect_modules("M2", "M3")
    system.connect_modules("M3", "M4")

    return system
