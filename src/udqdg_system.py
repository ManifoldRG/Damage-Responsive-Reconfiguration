import numpy as np
from typing import Any, Dict, List, Set, Tuple, Optional
import random
from dataclasses import dataclass
import networkx as nx
try:
    from .dual_quaternion import UnitDualQuaternion, LATTICE_DIRECTIONS, LATTICE_DIRECTIONS_2D
except ImportError:
    from dual_quaternion import UnitDualQuaternion, LATTICE_DIRECTIONS, LATTICE_DIRECTIONS_2D

@dataclass
class SphericalModule:
    """A spherical module in the UDQDG system."""
    id: str
    position: np.ndarray
    is_active: bool = True
    is_faulty: bool = False  # Damaged/faulty module flag
    is_moving: bool = False  # Set True during concurrent movement phase
    radius: float = 0.5  # Half of unit lattice step for perfect touching
    color: Tuple[float, float, float] = (0.7, 0.7, 0.9)


@dataclass
class PivotStep:
    """Records a single pivot operation for visualization/replay."""
    iteration: int
    pivot_type: str  # 'corner' or 'lateral'
    module_id: str
    param1: str      # axis_module (corner) or old_neighbor (lateral)
    param2: str      # new_direction (corner) or new_neighbor (lateral)
    from_pos: np.ndarray
    to_pos: np.ndarray

    def to_tuple(self) -> Tuple:
        """Convert to animation-compatible tuple format."""
        return (self.pivot_type, self.module_id, self.param1, self.param2)


@dataclass
class RestorationStep:
    """Records a single restoration move during Phase 2."""
    iteration: int
    pivot_type: str                     # 'corner' or 'lateral'
    module_id: str
    param1: str                         # axis_module or old_neighbor
    param2: str                         # new_direction or new_neighbor
    from_pos: np.ndarray
    to_pos: np.ndarray
    distance_before: float              # Distance to original before move
    distance_after: float               # Distance to original after move
    is_restoration_complete: bool       # True if reached original position

    def to_tuple(self) -> Tuple:
        """Convert to animation-compatible tuple format."""
        return (self.pivot_type, self.module_id, self.param1, self.param2)

    def distance_improvement(self) -> float:
        """Return how much closer to original position after this move."""
        return self.distance_before - self.distance_after


class UDQDGSystem:
    """
    Simple Unit Dual Quaternion Directed Graph system.

    Implements G_t = (V_t, E_t, m_t, Q_t, q̂_t) with:
    - Spherical modules
    - Dual quaternion edge gains for pure translations
    - Corner and lateral pivot operations
    """

    def __init__(self, mode_2d: bool = False):
        self.modules: Dict[str, SphericalModule] = {}
        self.edges: Dict[Tuple[str, str], UnitDualQuaternion] = {}
        self.time = 0
        self.mode_2d = mode_2d
        self.directions = LATTICE_DIRECTIONS_2D if mode_2d else LATTICE_DIRECTIONS

    def add_module(self, module_id: str, position: np.ndarray, active: bool = True) -> SphericalModule:
        """Add a spherical module to the system."""
        module = SphericalModule(module_id, np.array(position, dtype=float), active)
        self.modules[module_id] = module
        return module

    def validate_no_overlaps(self) -> None:
        """
        Validate that no two modules occupy the same position.
        Raises ValueError if overlapping modules are found.
        """
        positions = {}
        for module_id, module in self.modules.items():
            pos_tuple = tuple(module.position.round(6))  # Round to avoid floating point issues
            if pos_tuple in positions:
                raise ValueError(
                    f"Configuration error: Module '{module_id}' at position {module.position} "
                    f"overlaps with module '{positions[pos_tuple]}' at the same position!"
                )
            positions[pos_tuple] = module_id

    def connect_modules(self, module_a: str, module_b: str) -> bool:
        """Connect two modules with dual quaternion edge gain."""
        if module_a not in self.modules or module_b not in self.modules:
            return False

        pos_a = self.modules[module_a].position
        pos_b = self.modules[module_b].position

        # Calculate translation vector
        translation = pos_b - pos_a

        # Verify it's a unit lattice step
        if not self._is_unit_lattice_step(translation):
            return False

        # Create dual quaternion edge gains
        edge_gain_ab = UnitDualQuaternion(translation)
        edge_gain_ba = edge_gain_ab.inverse

        self.edges[(module_a, module_b)] = edge_gain_ab
        self.edges[(module_b, module_a)] = edge_gain_ba

        return True

    def _is_unit_lattice_step(self, translation: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if translation is a valid unit lattice step."""
        # Check if magnitude is approximately 1.0
        if not np.isclose(np.linalg.norm(translation), 1.0, atol=tolerance):
            return False

        # Check if direction is in allowed lattice directions
        direction = tuple(np.round(translation).astype(int))
        return direction in self.directions.values()

    def get_neighbors(self, module_id: str) -> List[str]:
        """Get all connected neighbors of a module."""
        neighbors = []
        for (a, b) in self.edges:
            if a == module_id:
                neighbors.append(b)
        return neighbors

    def corner_pivot(self, pivot_module: str, axis_module: str, new_direction: str) -> bool:
        """
        Perform corner pivot: move pivot_module around corner of axis_module.

        Args:
            pivot_module: Module to pivot
            axis_module: Module to pivot around
            new_direction: New lattice direction (e.g., 'POS_X')
        """
        if not self._can_corner_pivot(pivot_module, axis_module, new_direction):
            return False

        # Get all current neighbors before moving
        old_neighbors = self.get_neighbors(pivot_module)

        # Remove ALL connections (we'll reconnect valid ones after move)
        for neighbor in old_neighbors:
            if (pivot_module, neighbor) in self.edges:
                del self.edges[(pivot_module, neighbor)]
            if (neighbor, pivot_module) in self.edges:
                del self.edges[(neighbor, pivot_module)]

        # Calculate new position
        axis_pos = self.modules[axis_module].position
        new_translation = np.array(self.directions[new_direction])
        new_position = axis_pos + new_translation

        # Update module position
        self.modules[pivot_module].position = new_position

        # Reconnect to axis module (guaranteed to be adjacent after corner pivot)
        self.connect_modules(axis_module, pivot_module)

        return True

    def lateral_pivot(self, pivot_module: str, old_neighbor: str, new_neighbor: str) -> bool:
        """
        Perform lateral pivot: roll pivot_module from old_neighbor to new_neighbor.
        """
        if not self._can_lateral_pivot(pivot_module, old_neighbor, new_neighbor):
            return False

        # Get current direction
        old_edge = self.edges.get((old_neighbor, pivot_module))
        if not old_edge:
            return False

        direction = old_edge.translation

        # Get all current neighbors before moving
        old_neighbors = self.get_neighbors(pivot_module)

        # Remove ALL connections (we'll reconnect valid ones after move)
        for neighbor in old_neighbors:
            if (pivot_module, neighbor) in self.edges:
                del self.edges[(pivot_module, neighbor)]
            if (neighbor, pivot_module) in self.edges:
                del self.edges[(neighbor, pivot_module)]

        # Calculate new position
        new_neighbor_pos = self.modules[new_neighbor].position
        new_position = new_neighbor_pos + direction

        # Update module position
        self.modules[pivot_module].position = new_position

        # Reconnect to new_neighbor (guaranteed to be adjacent after lateral pivot)
        self.connect_modules(new_neighbor, pivot_module)

        return True

    def _is_pivotable(self, module_id: str) -> bool:
        """Check if module can pivot (must be active)."""
        if module_id not in self.modules:
            return False
        return self.modules[module_id].is_active

    def _are_orthogonal(self, dir1: np.ndarray, dir2: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if two directions are orthogonal."""
        return np.abs(np.dot(dir1, dir2)) < tolerance

    def _get_occupied_ports(self, module_id: str) -> Set[Tuple[int, int, int]]:
        """Get port directions occupied by edges to active modules."""
        occupied = set()
        for neighbor in self.get_neighbors(module_id):
            if not self.modules[neighbor].is_active:
                continue
            edge_gain = self.edges.get((module_id, neighbor))
            if edge_gain:
                direction = tuple(edge_gain.translation.astype(int))
                occupied.add(direction)
        return occupied

    def _is_port_available(self, module_id: str, direction: Tuple[int, int, int]) -> bool:
        """Check if a port in given direction is available."""
        occupied = self._get_occupied_ports(module_id)
        return direction not in occupied

    def _can_corner_pivot(self, pivot_module: str, axis_module: str, new_direction: str) -> bool:
        """Check if corner pivot is valid."""
        if not (pivot_module in self.modules and axis_module in self.modules):
            return False
        if not (self.modules[pivot_module].is_active and self.modules[axis_module].is_active):
            return False
        if new_direction not in self.directions:
            return False

        # Check if pivot module is pivotable (all neighbors active)
        if not self._is_pivotable(pivot_module):
            return False

        # Get current edge and check orthogonality
        current_edge = self.edges.get((pivot_module, axis_module))
        if current_edge:
            current_translation = current_edge.translation
            new_translation = np.array(self.directions[new_direction])

            if not self._are_orthogonal(current_translation, new_translation):
                return False

        # Check port exclusivity
        new_dir_tuple = self.directions[new_direction]
        neg_new_dir = tuple(-x for x in new_dir_tuple)

        # Axis module must have free port in new_direction (where pivot module will be)
        if not self._is_port_available(axis_module, new_dir_tuple):
            return False

        # Pivot module must have free port at -new_direction (toward axis)
        # Exception 1: if already connected to axis in this direction, port will be reused
        # Exception 2: if the blocking neighbor will no longer be adjacent after pivot
        current_dir = tuple(current_edge.translation.astype(int)) if current_edge else None
        if current_dir != neg_new_dir:
            if not self._is_port_available(pivot_module, neg_new_dir):
                # Port is occupied - check if the pivot will break that connection
                # Find which neighbor occupies that port
                blocking_neighbor = None
                for neighbor in self.get_neighbors(pivot_module):
                    edge = self.edges.get((pivot_module, neighbor))
                    if edge:
                        edge_dir = tuple(edge.translation.astype(int))
                        if edge_dir == neg_new_dir:
                            blocking_neighbor = neighbor
                            break

                if blocking_neighbor:
                    # Calculate new position after pivot
                    axis_pos = self.modules[axis_module].position
                    new_pos = axis_pos + np.array(new_dir_tuple)
                    blocking_pos = self.modules[blocking_neighbor].position

                    # Check if still adjacent after pivot (distance <= 1)
                    if np.linalg.norm(new_pos - blocking_pos) <= 1.0 + 1e-6:
                        # Still adjacent - port would remain occupied, block pivot
                        return False
                    # Not adjacent - pivot will break connection, port will be freed
                else:
                    # No blocking neighbor found but port is occupied - shouldn't happen
                    return False

        return True

    def _can_lateral_pivot(self, pivot_module: str, old_neighbor: str, new_neighbor: str) -> bool:
        """Check if lateral pivot is valid."""
        if not all(m in self.modules for m in [pivot_module, old_neighbor, new_neighbor]):
            return False
        if not all(self.modules[m].is_active for m in [pivot_module, old_neighbor, new_neighbor]):
            return False

        # Check if pivot module is pivotable (all neighbors active)
        if not self._is_pivotable(pivot_module):
            return False

        # Check if old_neighbor and new_neighbor are connected
        if not ((old_neighbor, new_neighbor) in self.edges or (new_neighbor, old_neighbor) in self.edges):
            return False

        # Get the direction to maintain
        old_edge = self.edges.get((old_neighbor, pivot_module))
        if not old_edge:
            return False

        direction = tuple(old_edge.translation.astype(int))
        neg_direction = tuple(-x for x in direction)

        # New neighbor must have free port at -direction
        if not self._is_port_available(new_neighbor, neg_direction):
            return False

        # Pivot module keeps same direction, so port will be reused
        return True

    def random_pivot(self) -> bool:
        """Perform a random valid pivot operation."""
        active_modules = [m for m in self.modules.values() if m.is_active]
        if len(active_modules) < 2:
            return False

        # Randomly choose corner or lateral pivot
        pivot_type = random.choice(['corner', 'lateral'])

        if pivot_type == 'corner':
            return self._random_corner_pivot()
        else:
            return self._random_lateral_pivot()

    def _random_corner_pivot(self) -> bool:
        """Perform random corner pivot."""
        # Find modules with neighbors
        candidates = []
        for module_id in self.modules:
            neighbors = self.get_neighbors(module_id)
            if neighbors and self.modules[module_id].is_active:
                candidates.extend([(module_id, n) for n in neighbors if self.modules[n].is_active])

        if not candidates:
            return False

        pivot_module, axis_module = random.choice(candidates)
        new_direction = random.choice(list(self.directions.keys()))

        return self.corner_pivot(pivot_module, axis_module, new_direction)

    def _random_lateral_pivot(self) -> bool:
        """Perform random lateral pivot."""
        # Find pivot candidates with multiple connection options
        for pivot_module in self.modules:
            if not self.modules[pivot_module].is_active:
                continue

            neighbors = self.get_neighbors(pivot_module)
            if len(neighbors) < 1:
                continue

            old_neighbor = random.choice(neighbors)

            # Find neighbors of the old_neighbor that could be new targets
            potential_new = self.get_neighbors(old_neighbor)
            potential_new = [n for n in potential_new if n != pivot_module and self.modules[n].is_active]

            if potential_new:
                new_neighbor = random.choice(potential_new)
                return self.lateral_pivot(pivot_module, old_neighbor, new_neighbor)

        return False

    def create_grid(self, size: int = 3) -> None:
        """Create a simple grid configuration for testing."""
        for x in range(size):
            for y in range(size):
                module_id = f"M{x}{y}"
                position = np.array([x, y, 0], dtype=float)
                self.add_module(module_id, position)

        # Connect grid
        for x in range(size):
            for y in range(size):
                current = f"M{x}{y}"
                if x < size - 1:
                    self.connect_modules(current, f"M{x+1}{y}")
                if y < size - 1:
                    self.connect_modules(current, f"M{x}{y+1}")

    def get_all_positions(self) -> Dict[str, np.ndarray]:
        """Get positions of all modules."""
        return {mid: module.position.copy() for mid, module in self.modules.items()}

    def get_all_edges(self) -> List[Tuple[str, str]]:
        """Get all edges (undirected)."""
        edges = set()
        for (a, b) in self.edges:
            edge = tuple(sorted([a, b]))
            edges.add(edge)
        return list(edges)

    def mark_fault(self, module_id: str) -> bool:
        """Mark a module as faulty (damaged)."""
        if module_id not in self.modules:
            return False
        self.modules[module_id].is_faulty = True
        self.modules[module_id].is_active = False
        return True

    def get_active_neighbors(self, module_id: str) -> List[str]:
        """Get active, non-faulty neighbors of a module."""
        return [n for n in self.get_neighbors(module_id)
                if self.modules[n].is_active and not self.modules[n].is_faulty]

    def has_moving_neighbor_of_neighbor(self, module_id: str) -> bool:
        """Check if any neighbor's neighbor is currently moving (2-hop check).

        Each module queries its neighbors, which in turn check their own
        neighbors for the is_moving flag.  This is fully local — no module
        inspects beyond its direct neighbors.
        """
        for nbr in self.get_active_neighbors(module_id):
            if self.modules[nbr].is_moving:
                return True
            for nbr2 in self.get_active_neighbors(nbr):
                if nbr2 == module_id:
                    continue
                if self.modules[nbr2].is_moving:
                    return True
        return False

    def _bfs_reachable(self, start: str, max_depth: int, excluded: set) -> set:
        """BFS from start up to max_depth hops, skipping nodes in excluded."""
        visited = {start}
        frontier = {start}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for nbr in self.get_active_neighbors(node):
                    if nbr not in excluded and nbr not in visited:
                        next_frontier.add(nbr)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(start)
        return visited

    def is_movable(self, module_id: str, safety_radius: int = 2) -> bool:
        """
        Local connectivity test per paper Section III-C (universal variant).

        u is movable if:
        - Leaf: |N_t(u)| = 1
        - Or: FOR ALL v in N_t(u), a neighbor of u can reach a neighbor
          of v within safety_radius-1 hops WITHOUT going through u or v.

        safety_radius=2 matches the original 2-hop check. Higher values
        (3, 4) are more permissive — they consider longer alternative paths.
        """
        if module_id not in self.modules:
            return False
        module = self.modules[module_id]
        if not module.is_active or module.is_faulty:
            return False

        neighbors = self.get_active_neighbors(module_id)
        if len(neighbors) == 0:
            return False
        if len(neighbors) == 1:
            return True  # leaf

        # For each neighbor v, BFS from u's other neighbors up to
        # (safety_radius - 1) hops, excluding u and v, then check
        # intersection with N(v). ALL neighbors must have an alternative path.
        for v in neighbors:
            excluded = {module_id, v}
            reachable_without_v = set()
            for w in neighbors:
                if w == v:
                    continue
                reachable_without_v |= self._bfs_reachable(w, safety_radius - 1, excluded)
            v_neighbors = set(self.get_active_neighbors(v))
            if not (reachable_without_v & v_neighbors):
                return False

        return True

    def get_all_admissible_pivots(
        self,
        module_id: str,
    ) -> List[Tuple]:
        """
        Get ALL physically admissible pivot moves with displacement vectors.

        Returns list of (pivot_type, module_id, param1, param2, delta_p) tuples.
        No distance/alignment filtering — returns all valid moves.
        """
        if module_id not in self.modules:
            return []

        current_pos = self.modules[module_id].position
        neighbors = self.get_neighbors(module_id)
        pivots = []

        # Corner pivots
        for neighbor in neighbors:
            if not self.modules[neighbor].is_active:
                continue
            for dir_name in self.directions:
                if self._can_corner_pivot(module_id, neighbor, dir_name):
                    neighbor_pos = self.modules[neighbor].position
                    new_pos = neighbor_pos + np.array(self.directions[dir_name])

                    # Collision check (only with other active modules)
                    if self._position_is_occupied(new_pos, exclude_module=module_id):
                        continue

                    delta_p = new_pos - current_pos
                    pivots.append(('corner', module_id, neighbor, dir_name, delta_p))

        # Lateral pivots
        for old_neighbor in neighbors:
            if not self.modules[old_neighbor].is_active:
                continue
            neighbor_neighbors = self.get_neighbors(old_neighbor)
            for new_neighbor in neighbor_neighbors:
                if new_neighbor == module_id:
                    continue
                if not self.modules[new_neighbor].is_active:
                    continue
                if self._can_lateral_pivot(module_id, old_neighbor, new_neighbor):
                    old_edge = self.edges.get((old_neighbor, module_id))
                    if old_edge:
                        direction = old_edge.translation
                        new_pos = self.modules[new_neighbor].position + direction

                        if self._position_is_occupied(new_pos, exclude_module=module_id):
                            continue

                        delta_p = new_pos - current_pos
                        pivots.append(('lateral', module_id, old_neighbor, new_neighbor, delta_p))

        return pivots

    def select_pivot_by_alignment(
        self,
        module_id: str,
        direction: np.ndarray,
    ) -> Optional[Tuple]:
        """
        Select pivot maximizing ⟨Δp, direction⟩ subject to positive alignment.

        Returns (pivot_type, module_id, param1, param2, delta_p) or None.
        """
        pivots = self.get_all_admissible_pivots(module_id)
        best_pivot = None
        best_alignment = 0.0
        for pivot in pivots:
            delta_p = pivot[4]
            alignment = float(np.dot(delta_p, direction))
            if alignment > best_alignment:
                best_alignment = alignment
                best_pivot = pivot
        return best_pivot

    def _position_is_occupied(self, position: np.ndarray, exclude_module: Optional[str] = None) -> bool:
        """Check if a position is occupied by any active module."""
        for module_id, module in self.modules.items():
            if exclude_module and module_id == exclude_module:
                continue
            if not module.is_active:
                continue
            if np.allclose(module.position, position):
                return True
        return False

    def _form_new_connections(self) -> int:
        """
        Scan for and form new connections between adjacent modules.

        Returns:
            Number of new connections formed
        """
        new_connections = 0
        active_modules = [mid for mid, m in self.modules.items() if m.is_active]

        for module_id in active_modules:
            module_pos = self.modules[module_id].position

            # Check all other active modules
            for other_id in active_modules:
                if module_id == other_id:
                    continue

                # Skip if already connected
                if (module_id, other_id) in self.edges:
                    continue

                other_pos = self.modules[other_id].position
                translation = other_pos - module_pos

                # Check if they're at unit lattice distance
                if self._is_unit_lattice_step(translation):
                    # Check if both modules have free ports for connection
                    direction = tuple(translation.astype(int))
                    neg_direction = tuple((-translation).astype(int))

                    if self._is_port_available(module_id, direction) and \
                       self._is_port_available(other_id, neg_direction):
                        # Form the connection
                        if self.connect_modules(module_id, other_id):
                            new_connections += 1

        return new_connections

    def _calculate_pivot_destination(
        self,
        module_id: str,
        pivot: Tuple[str, str, str, str]
    ) -> np.ndarray:
        """
        Calculate where a module will end up after executing a pivot.

        Args:
            module_id: The module being pivoted
            pivot: Tuple of (pivot_type, module_id, param1, param2)

        Returns:
            The destination position as numpy array
        """
        pivot_type, _, param1, param2 = pivot
        module_pos = self.modules[module_id].position

        if pivot_type == 'corner':
            # Corner pivot: rotate 90 degrees around axis_module
            axis_module = param1
            new_direction = param2  # String like 'POS_X', 'NEG_Y', etc.
            axis_pos = self.modules[axis_module].position

            # Get direction vector from LATTICE_DIRECTIONS or self.directions
            if new_direction in self.directions:
                direction_vec = np.array(self.directions[new_direction], dtype=float)
            else:
                # Fallback: return axis position
                return axis_pos.copy()

            # New position is axis position + direction
            return axis_pos + direction_vec

        elif pivot_type == 'lateral':
            # Lateral pivot: roll from old_neighbor to new_neighbor
            old_neighbor = param1
            new_neighbor = param2
            new_neighbor_pos = self.modules[new_neighbor].position

            # Calculate direction from new_neighbor to module's destination
            # The module ends up on the opposite side of new_neighbor from old_neighbor
            old_neighbor_pos = self.modules[old_neighbor].position

            # Module moves to position that's adjacent to new_neighbor,
            # continuing in same direction as old_neighbor -> new_neighbor
            direction = new_neighbor_pos - old_neighbor_pos
            direction = direction / np.linalg.norm(direction)  # Normalize
            return new_neighbor_pos + direction

        # Fallback: return current position
        return module_pos.copy()

    def coagulation(
        self,
        fault_id: str,
        max_iterations: int = 1000,
        record_steps: bool = True,
        safety_radius: int = 2
    ) -> Dict[str, Any]:
        """
        Phase 1: Decentralized coagulation via distress token propagation.

        First-responder model with single-token selection:
        - Fault-adjacent modules generate distress tokens (direction only)
        - Each module that receives tokens selects the CLOSEST one
        - If movable: consume it and move toward the fault
        - If not movable: propagate just that one selected token to neighbors
        - Tokens regenerated fresh each iteration from fault-adjacent sources
        """
        if fault_id not in self.modules:
            return {"success": False, "reason": "Fault module not found"}
        if not self.modules[fault_id].is_faulty:
            return {"success": False, "reason": "Module is not marked as faulty"}

        stats = {
            "iterations": 0,
            "total_moves": 0,
            "modules_responded": set(),
            "reconnected": False,
            "new_connections_formed": 0,
            "success": True,
            "token_transmissions": 0
        }
        if record_steps:
            stats["steps"] = []
            stats["parallel_steps"] = []

        # Token storage: {module_id: Optional[direction_vector ξ]}
        # Each module holds at most ONE token after selection — the closest.
        tokens: Dict[str, Optional[np.ndarray]] = {
            mid: None for mid, m in self.modules.items()
            if m.is_active and not m.is_faulty
        }

        # Track which modules consumed a token last iteration
        moved_last_iteration: Set[str] = set()

        # Oscillation prevention
        position_history: Dict[str, Set[Tuple]] = {}

        no_progress_count = 0

        for iteration in range(max_iterations):
            stats["iterations"] = iteration + 1

            if self.is_connected(active_only=True):
                stats["reconnected"] = True
                break

            # ── Token Generation + Propagation ──
            # Collect all incoming tokens per module, then select one.
            incoming: Dict[str, List[np.ndarray]] = {
                mid: [] for mid in tokens if mid in self.modules
                and self.modules[mid].is_active and not self.modules[mid].is_faulty
            }

            for u in list(incoming.keys()):
                # Generation: fault-adjacent modules emit direction tokens
                for f in self.get_neighbors(u):
                    if not self.modules[f].is_active or self.modules[f].is_faulty:
                        edge_uf = self.edges.get((u, f))
                        if edge_uf:
                            incoming[u].append(edge_uf.translation.copy())

                # Propagation: modules that didn't move last iteration
                # forward their single selected token to neighbors
                if u not in moved_last_iteration and tokens[u] is not None:
                    for w in self.get_active_neighbors(u):
                        if w not in incoming:
                            continue
                        edge_wu = self.edges.get((w, u))
                        if edge_wu:
                            xi_w = edge_wu.translation + tokens[u]
                            incoming[w].append(xi_w)
                            stats["token_transmissions"] += 1

            # Selection: each module keeps only the closest token
            moved_last_iteration = set()
            for mid in incoming:
                if incoming[mid]:
                    tokens[mid] = min(incoming[mid],
                                      key=lambda xi: np.linalg.norm(xi))
                else:
                    tokens[mid] = None

            # ── Movement Phase (concurrent with 2-hop exclusion) ──
            # Each module only checks its neighbors' neighbors for the
            # is_moving flag — fully local, no global knowledge.
            candidates = []
            for u, xi in tokens.items():
                if xi is None:
                    continue
                if u not in self.modules or not self.modules[u].is_active:
                    continue
                candidates.append((u, float(np.linalg.norm(xi))))

            # Priority: closest to fault first, then ID for determinism
            candidates.sort(key=lambda x: (x[1], x[0]))

            # ── Communication round ──
            # Each candidate checks the is_moving flag on its neighbors'
            # neighbors.  If clear, it sets its own flag and announces
            # intent.  No graph mutation happens in this round.
            planned_moves = []  # [(module_id, pivot, from_pos)]
            for u, _ in candidates:
                if not self.modules[u].is_active:
                    continue
                if not self.is_movable(u, safety_radius=safety_radius):
                    continue
                if self.has_moving_neighbor_of_neighbor(u):
                    continue

                xi_star = tokens.get(u)
                if xi_star is None:
                    continue

                pivot = self.select_pivot_by_alignment(u, xi_star)
                if pivot is None:
                    continue

                from_pos = self.modules[u].position.copy()
                dest_pos = from_pos + pivot[4]
                dest_key = tuple(np.round(dest_pos).astype(int))

                if dest_key in position_history.get(u, set()):
                    continue
                if self._position_is_occupied(dest_pos, exclude_module=u):
                    continue

                # Commit intent — neighbors can now see this flag
                self.modules[u].is_moving = True
                planned_moves.append((u, pivot, from_pos))

            # ── Action round ──
            # All committed modules execute their pivots.
            # Re-check destination occupancy: a distant mover that
            # executed earlier in this round may have landed there.
            moves_this_iteration = 0
            iteration_steps = []
            for u, pivot, from_pos in planned_moves:
                pivot_type, mid, param1, param2, delta_p = pivot

                # Local collision detection at action time
                dest_pos = from_pos + delta_p
                if self._position_is_occupied(dest_pos, exclude_module=mid):
                    self.modules[mid].is_moving = False
                    continue

                success = False
                if pivot_type == 'corner':
                    success = self.corner_pivot(mid, param1, param2)
                elif pivot_type == 'lateral':
                    success = self.lateral_pivot(mid, param1, param2)

                if success:
                    moves_this_iteration += 1
                    stats["total_moves"] += 1
                    stats["modules_responded"].add(mid)

                    to_pos = self.modules[mid].position.copy()

                    if mid not in position_history:
                        position_history[mid] = set()
                    position_history[mid].add(tuple(np.round(from_pos).astype(int)))

                    moved_last_iteration.add(mid)
                    tokens[mid] = None

                    if record_steps:
                        step = PivotStep(
                            iteration=iteration + 1,
                            pivot_type=pivot_type,
                            module_id=mid,
                            param1=param1,
                            param2=param2,
                            from_pos=from_pos,
                            to_pos=to_pos
                        )
                        stats["steps"].append(step)
                        iteration_steps.append(step)

            # Clear moving flags
            for u, _, _ in planned_moves:
                self.modules[u].is_moving = False

            if record_steps and iteration_steps:
                stats["parallel_steps"].append(iteration_steps)

            new_conn = self._form_new_connections()
            stats["new_connections_formed"] += new_conn

            if moves_this_iteration == 0 and new_conn == 0:
                no_progress_count += 1
                if no_progress_count >= 5:
                    break
            else:
                no_progress_count = 0

        stats["modules_moved"] = set(stats["modules_responded"])
        stats["modules_responded"] = list(stats["modules_responded"])
        return stats

    @staticmethod
    def _select_token(relevant_toks: Dict[str, np.ndarray],
                      strategy: str) -> Tuple[str, np.ndarray]:
        """Select a token from relevant_toks based on strategy.

        Returns (selected_key, selected_vector).
        """
        if strategy == "nearest":
            sel = min(relevant_toks.keys(),
                      key=lambda v: np.linalg.norm(relevant_toks[v]))
        elif strategy == "random":
            sel = random.choice(list(relevant_toks.keys()))
        else:  # "furthest" (default)
            sel = max(relevant_toks.keys(),
                      key=lambda v: np.linalg.norm(relevant_toks[v]))
        return sel, relevant_toks[sel]

    def restructuring(
        self,
        pre_damage_neighbors: Dict[str, Dict[str, np.ndarray]],
        original_positions: Optional[Dict[str, np.ndarray]] = None,
        max_iterations: int = 100,
        record_steps: bool = True,
        token_strategy: str = "furthest",
        safety_radius: int = 2,
        coag_moved: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Phase 2: Decentralized restructuring via unlabelled slot-filling.

        Only modules that did NOT move during coagulation generate tokens,
        since their pre-damage neighbor directions are still accurate.
        Tokens propagate through the structure; the first out-of-place
        movable module consumes and moves toward the empty slot.

        Args:
            coag_moved: Set of module IDs that moved during coagulation.
                       Only non-movers generate restructuring tokens.
        """
        stats = {
            "iterations": 0,
            "restoration_moves": 0,
            "success": False,
            "token_transmissions": 0
        }
        if record_steps:
            stats["steps"] = []
            stats["parallel_steps"] = []

        if not pre_damage_neighbors:
            stats["success"] = True
            return stats

        # Token storage: {module_id: Optional[direction_vector ζ]}
        # Each module holds at most ONE token after selection.
        tokens: Dict[str, Optional[np.ndarray]] = {
            mid: None for mid, m in self.modules.items()
            if m.is_active and not m.is_faulty
        }

        # Track which modules consumed a token last iteration
        moved_last_iteration: Set[str] = set()

        # Oscillation prevention
        position_history: Dict[str, Set[Tuple]] = {}

        no_progress_count = 0

        # Helper: check if a slot direction from u is occupied by any neighbor
        def _slot_occupied(u: str, rho_uv: np.ndarray) -> bool:
            for w in self.get_active_neighbors(u):
                edge_uw = self.edges.get((u, w))
                if edge_uw and np.allclose(edge_uw.translation, rho_uv):
                    return True
            return False

        # Helper: check if all pre-damage neighbor slots are filled
        def _all_slots_filled() -> bool:
            for u, pre_nbrs in pre_damage_neighbors.items():
                if u not in self.modules or not self.modules[u].is_active:
                    continue
                if self.modules[u].is_faulty:
                    continue
                for v, rho_uv in pre_nbrs.items():
                    if v not in self.modules or not self.modules[v].is_active:
                        continue
                    if self.modules[v].is_faulty:
                        continue
                    if not _slot_occupied(u, rho_uv):
                        return False
            return True

        # Helper: check if module u has all its own slots filled
        def _module_in_place(u: str) -> bool:
            if u not in pre_damage_neighbors:
                return True
            for v, rho_uv in pre_damage_neighbors[u].items():
                if v not in self.modules or not self.modules[v].is_active:
                    continue
                if self.modules[v].is_faulty:
                    continue
                if not _slot_occupied(u, rho_uv):
                    return False
            return True

        # Helper: select token from list based on strategy
        def _pick_token(token_list: List[np.ndarray]) -> np.ndarray:
            if token_strategy == "nearest":
                return min(token_list, key=lambda z: np.linalg.norm(z))
            elif token_strategy == "random":
                return random.choice(token_list)
            else:  # furthest (default per paper)
                return max(token_list, key=lambda z: np.linalg.norm(z))

        for iteration in range(max_iterations):
            stats["iterations"] = iteration + 1

            if _all_slots_filled():
                stats["success"] = True
                break

            # ── Token Generation + Propagation ──
            # Collect all incoming tokens per module, then select one.
            incoming: Dict[str, List[np.ndarray]] = {
                mid: [] for mid in tokens if mid in self.modules
                and self.modules[mid].is_active and not self.modules[mid].is_faulty
            }

            for u in list(incoming.keys()):
                if self.modules[u].is_faulty:
                    continue

                # Generation: only modules that did NOT move during
                # coagulation emit tokens — their rho_uv directions
                # are still accurate from their original positions.
                if u in pre_damage_neighbors and (
                        coag_moved is None or u not in coag_moved):
                    for v, rho_uv in pre_damage_neighbors[u].items():
                        if v not in self.modules or not self.modules[v].is_active:
                            continue
                        if self.modules[v].is_faulty:
                            continue
                        if _slot_occupied(u, rho_uv):
                            continue
                        incoming[u].append(rho_uv.copy())

                # Propagation: modules that didn't move last iteration
                # forward their single selected token to neighbors
                if u not in moved_last_iteration and tokens[u] is not None:
                    for w in self.get_active_neighbors(u):
                        if w not in incoming:
                            continue
                        edge_wu = self.edges.get((w, u))
                        if edge_wu:
                            zeta_w = edge_wu.translation + tokens[u]
                            incoming[w].append(zeta_w)
                            stats["token_transmissions"] += 1

            # Selection: each module keeps one token based on strategy
            moved_last_iteration = set()
            for mid in incoming:
                if incoming[mid]:
                    tokens[mid] = _pick_token(incoming[mid])
                else:
                    tokens[mid] = None

            # ── Movement Phase (concurrent with 2-hop exclusion) ──
            # Only modules that moved during coagulation are candidates —
            # they're the displaced ones that need to find a new slot.
            # Non-movers are already in position and should stay put.
            candidates = []
            for u, zeta in tokens.items():
                if zeta is None:
                    continue
                if u not in self.modules or not self.modules[u].is_active:
                    continue
                if coag_moved is not None and u not in coag_moved:
                    continue  # didn't move in coag, stay in place
                if _module_in_place(u):
                    continue  # already in a good spot
                dist = float(np.linalg.norm(zeta))
                candidates.append((u, dist))

            # Sort by strategy
            if token_strategy == "random":
                random.shuffle(candidates)
            elif token_strategy == "nearest":
                candidates.sort(key=lambda x: (x[1], x[0]))
            else:  # furthest
                candidates.sort(key=lambda x: (-x[1], x[0]))

            # Plan moves — each candidate checks neighbors' neighbors
            # ── Communication round ──
            planned_moves = []
            for u, _ in candidates:
                if not self.modules[u].is_active:
                    continue
                if not self.is_movable(u, safety_radius=safety_radius):
                    continue
                if self.has_moving_neighbor_of_neighbor(u):
                    continue

                zeta_star = tokens.get(u)
                if zeta_star is None:
                    continue

                pivot = self.select_pivot_by_alignment(u, zeta_star)
                if pivot is None:
                    continue

                from_pos = self.modules[u].position.copy()
                dest_pos = from_pos + pivot[4]
                dest_key = tuple(np.round(dest_pos).astype(int))

                if dest_key in position_history.get(u, set()):
                    continue
                if self._position_is_occupied(dest_pos, exclude_module=u):
                    continue

                self.modules[u].is_moving = True
                planned_moves.append((u, pivot, from_pos, zeta_star))

            # ── Action round ──
            moves_this_iteration = 0
            iteration_steps = []
            for u, pivot, from_pos, zeta_star in planned_moves:
                pivot_type, mid, param1, param2, delta_p = pivot

                # Local collision detection at action time
                dest_pos = from_pos + delta_p
                if self._position_is_occupied(dest_pos, exclude_module=mid):
                    self.modules[mid].is_moving = False
                    continue

                success = False
                if pivot_type == 'corner':
                    success = self.corner_pivot(mid, param1, param2)
                elif pivot_type == 'lateral':
                    success = self.lateral_pivot(mid, param1, param2)

                if success:
                    moves_this_iteration += 1
                    stats["restoration_moves"] += 1

                    to_pos = self.modules[mid].position.copy()

                    if mid not in position_history:
                        position_history[mid] = set()
                    position_history[mid].add(tuple(np.round(from_pos).astype(int)))

                    moved_last_iteration.add(mid)
                    tokens[mid] = None

                    if record_steps:
                        token_dist_before = float(np.linalg.norm(zeta_star))
                        actual_delta = to_pos - from_pos
                        token_dist_after = float(np.linalg.norm(
                            zeta_star - actual_delta))
                        step = RestorationStep(
                            iteration=iteration + 1,
                            pivot_type=pivot_type,
                            module_id=mid,
                            param1=param1,
                            param2=param2,
                            from_pos=from_pos,
                            to_pos=to_pos,
                            distance_before=token_dist_before,
                            distance_after=token_dist_after,
                            is_restoration_complete=False
                        )
                        stats["steps"].append(step)
                        iteration_steps.append(step)

            # Clear moving flags
            for u, _, _, _ in planned_moves:
                self.modules[u].is_moving = False

            if record_steps and iteration_steps:
                stats["parallel_steps"].append(iteration_steps)

            new_conn = self._form_new_connections()

            if moves_this_iteration == 0 and new_conn == 0:
                no_progress_count += 1
                if no_progress_count >= 5:
                    break
            else:
                no_progress_count = 0

        return stats

    def restructuring_displacement(
        self,
        original_positions: Dict[str, np.ndarray],
        max_iterations: int = 100,
        record_steps: bool = True
    ) -> Dict[str, Any]:
        """
        Phase 2 (displacement-guided): Position restoration via displacement vectors.

        Each module computes its displacement from its original position and
        moves toward it. Modules with largest displacement move first (greedy).
        Uses alignment-based pivot selection toward original position.

        This is the paper-code baseline for comparison with token-based restructuring.
        """
        stats = {
            "iterations": 0,
            "restoration_moves": 0,
            "success": False,
            "token_transmissions": 0
        }
        if record_steps:
            stats["steps"] = []
            stats["parallel_steps"] = []

        if original_positions is None:
            stats["success"] = True
            return stats

        for iteration in range(max_iterations):
            stats["iterations"] = iteration + 1

            # Compute displacement for each active module
            displacements = {}
            for mid, orig_pos in original_positions.items():
                if mid not in self.modules or not self.modules[mid].is_active:
                    continue
                if self.modules[mid].is_faulty:
                    continue
                disp = orig_pos - self.modules[mid].position
                dist = float(np.linalg.norm(disp))
                if dist > 0.5:  # Only modules that have moved significantly
                    displacements[mid] = (disp, dist)

            if not displacements:
                stats["success"] = True
                break

            # Sort by displacement (largest first)
            sorted_modules = sorted(displacements.keys(),
                                    key=lambda m: displacements[m][1],
                                    reverse=True)

            moves_this_iteration = 0
            occupied_destinations = set()

            for mid in sorted_modules:
                disp, dist = displacements[mid]

                if not self.is_movable(mid):
                    continue

                # Select pivot aligned with displacement direction
                pivot = self.select_pivot_by_alignment(mid, disp)
                if pivot is None:
                    continue

                # Check destination not already claimed
                from_pos = self.modules[mid].position.copy()
                dest_pos = from_pos + pivot[4]
                dest_key = tuple(np.round(dest_pos).astype(int))
                if dest_key in occupied_destinations:
                    continue

                # Verify move reduces displacement (prevent oscillation)
                new_disp = np.linalg.norm(original_positions[mid] - dest_pos)
                if new_disp >= dist - 0.01:
                    continue  # Skip if not strictly improving

                occupied_destinations.add(dest_key)
                pivot_type, _, param1, param2, delta_p = pivot
                success = False
                if pivot_type == 'corner':
                    success = self.corner_pivot(mid, param1, param2)
                elif pivot_type == 'lateral':
                    success = self.lateral_pivot(mid, param1, param2)

                if success:
                    moves_this_iteration += 1
                    stats["restoration_moves"] += 1
                    to_pos = self.modules[mid].position.copy()

                    if record_steps:
                        stats["steps"].append(RestorationStep(
                            iteration=iteration + 1,
                            pivot_type=pivot_type,
                            module_id=mid,
                            param1=param1,
                            param2=param2,
                            from_pos=from_pos,
                            to_pos=to_pos,
                            distance_before=dist,
                            distance_after=float(np.linalg.norm(original_positions[mid] - to_pos)),
                            is_restoration_complete=False
                        ))

            # Form new connections
            self._form_new_connections()

            if moves_this_iteration == 0:
                break

        return stats

    def is_connected(self, active_only: bool = True) -> bool:
        """
        Check if the graph is connected.

        Args:
            active_only: If True, only check connectivity of active modules

        Returns:
            True if connected, False otherwise
        """
        if not self.modules:
            return True

        # Build networkx graph
        G = nx.Graph()

        # Add nodes
        for module_id, module in self.modules.items():
            if not active_only or module.is_active:
                G.add_node(module_id)

        # Add edges
        for (src, dst) in self.edges:
            if src in G.nodes and dst in G.nodes:
                G.add_edge(src, dst)

        # Check connectivity
        return nx.is_connected(G) if G.number_of_nodes() > 0 else True

    def get_connected_components(self, active_only: bool = True) -> List[Set[str]]:
        """
        Get all connected components.

        Args:
            active_only: If True, only consider active modules

        Returns:
            List of sets, each set containing module IDs in a component
        """
        if not self.modules:
            return []

        # Build networkx graph
        G = nx.Graph()

        # Add nodes
        for module_id, module in self.modules.items():
            if not active_only or module.is_active:
                G.add_node(module_id)

        # Add edges
        for (src, dst) in self.edges:
            if src in G.nodes and dst in G.nodes:
                G.add_edge(src, dst)

        # Get connected components
        return [set(component) for component in nx.connected_components(G)]

    def to_2d_array(self, padding: int = 1) -> np.ndarray:
        """
        Convert system to 2D array for visualization (assumes z=0 plane).

        Args:
            padding: Number of empty cells to pad around the modules (default: 1)

        Returns:
            2D numpy array where:
            0 = empty space
            1 = active module
            -1 = faulty module
        """
        if not self.modules:
            return np.array([[]])

        # Find bounds
        positions = [m.position for m in self.modules.values()]
        x_coords = [int(p[0]) for p in positions]
        y_coords = [int(p[1]) for p in positions]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # Create array with padding (note: y-axis is inverted for visualization)
        width = max_x - min_x + 1 + 2 * padding
        height = max_y - min_y + 1 + 2 * padding
        array = np.zeros((height, width), dtype=int)

        # Fill array (offset by padding)
        for module in self.modules.values():
            x = int(module.position[0]) - min_x + padding
            y = int(module.position[1]) - min_y + padding

            if module.is_faulty:
                array[y, x] = -1
            elif module.is_active:
                array[y, x] = 1
            else:
                array[y, x] = 0

        return array

    def full_damage_response(
        self,
        fault_module_id: str,
        restore_positions: bool = True,
        max_phase1_iterations: int = 1000,
        max_phase2_iterations: int = 100,
        record_steps: bool = True,
        token_strategy: str = "furthest",
        safety_radius: int = 2,
        reconstruction_method: str = "displacement",
        # Legacy params (ignored, kept for API compat)
        one_per_subgraph: bool = True,
        parallel_subgraphs: bool = True
    ) -> Dict[str, Any]:
        """
        Complete damage response: Phase 1 (coagulation) + Phase 2 (restructuring).

        Uses decentralized token-based algorithms per paper Algorithms 1 & 2:
        1. Captures pre-damage neighbor sets N_0(u) for all active modules
        2. Marks the specified module as faulty
        3. Runs coagulation (distress tokens) to restore connectivity
        4. Runs restructuring (rendezvous tokens) to recover shape
        """
        result: Dict[str, Any] = {
            "phase1": None,
            "phase2": None,
            "overall_success": False,
            "total_moves": 0,
            "fault_position": None
        }

        if fault_module_id not in self.modules:
            result["error"] = "Fault module not found"
            return result

        # Record fault position
        result["fault_position"] = tuple(
            self.modules[fault_module_id].position.astype(int)
        )

        # ── Capture N_0(u): pre-damage neighbor sets with edge transformations ──
        pre_damage_neighbors: Dict[str, Dict[str, np.ndarray]] = {}
        original_positions: Dict[str, np.ndarray] = {}
        for mid, module in self.modules.items():
            if module.is_active and not module.is_faulty:
                original_positions[mid] = module.position.copy()
                neighbors_info = {}
                for n in self.get_neighbors(mid):
                    edge = self.edges.get((mid, n))
                    if edge:
                        neighbors_info[n] = edge.translation.copy()  # ρ_uv
                pre_damage_neighbors[mid] = neighbors_info

        # Mark fault
        if not self.modules[fault_module_id].is_faulty:
            self.mark_fault(fault_module_id)

        # Phase 1: Coagulation
        phase1_stats = self.coagulation(
            fault_id=fault_module_id,
            max_iterations=max_phase1_iterations,
            record_steps=record_steps,
            safety_radius=safety_radius
        )
        result["phase1"] = phase1_stats
        result["total_moves"] = phase1_stats.get("total_moves", 0)

        # Snapshot positions after phase 1 (before restructuring)
        result["post_phase1_positions"] = {
            mid: module.position.copy()
            for mid, module in self.modules.items()
            if module.is_active and not module.is_faulty
        }

        if not phase1_stats.get("reconnected", False):
            result["overall_success"] = False
            return result

        # Phase 2: Restructuring
        if restore_positions:
            if reconstruction_method == "displacement":
                phase2_stats = self.restructuring_displacement(
                    original_positions=original_positions,
                    max_iterations=max_phase2_iterations,
                    record_steps=record_steps
                )
            else:
                coag_moved = phase1_stats.get("modules_moved", set())
                phase2_stats = self.restructuring(
                    pre_damage_neighbors=pre_damage_neighbors,
                    original_positions=original_positions,
                    max_iterations=max_phase2_iterations,
                    record_steps=record_steps,
                    token_strategy=token_strategy,
                    safety_radius=safety_radius,
                    coag_moved=coag_moved
                )
            result["phase2"] = phase2_stats
            result["total_moves"] += phase2_stats.get("restoration_moves", 0)
            result["overall_success"] = True  # Phase 1 reconnected
        else:
            result["overall_success"] = True

        return result

    def get_damage_response_summary(
        self,
        response_result: Dict[str, Any]
    ) -> str:
        """
        Generate a human-readable summary of a damage response.

        Args:
            response_result: Result from full_damage_response()

        Returns:
            Formatted string summary
        """
        lines = []
        lines.append("=" * 60)
        lines.append("DAMAGE RESPONSE SUMMARY")
        lines.append("=" * 60)

        # Fault info
        fault_pos = response_result.get("fault_position")
        if fault_pos:
            lines.append(f"Fault Position: {fault_pos}")

        # Phase 1 summary
        p1 = response_result.get("phase1", {})
        if p1:
            lines.append("")
            lines.append("PHASE 1: Reconnection")
            lines.append("-" * 40)
            lines.append(f"  Iterations: {p1.get('iterations', 0)}")
            lines.append(f"  Moves: {p1.get('total_moves', 0)}")
            lines.append(f"  Modules responded: {len(p1.get('modules_responded', []))}")
            lines.append(f"  Reconnected: {'YES' if p1.get('reconnected') else 'NO'}")

        # Phase 2 summary
        p2 = response_result.get("phase2")
        if p2:
            lines.append("")
            lines.append("PHASE 2: Position Restoration")
            lines.append("-" * 40)
            lines.append(f"  Iterations: {p2.get('iterations', 0)}")
            lines.append(f"  Moves: {p2.get('restoration_moves', 0)}")
            lines.append(f"  Fully restored: {len(p2.get('fully_restored', []))}")
            lines.append(f"  Partially restored: {len(p2.get('partially_restored', []))}")
            lines.append(f"  Could not restore: {len(p2.get('could_not_restore', []))}")

            metrics = p2.get("metrics")
            if metrics:
                lines.append(f"  Displacement reduction: {metrics.displacement_reduction:.2f}")
                lines.append(f"  Reduction percent: {metrics.displacement_reduction_percent:.1f}%")

        # Overall result
        lines.append("")
        lines.append("OVERALL RESULT")
        lines.append("-" * 40)
        lines.append(f"  Total moves: {response_result.get('total_moves', 0)}")
        success = response_result.get("overall_success", False)
        lines.append(f"  Success: {'YES' if success else 'NO'}")
        lines.append("=" * 60)

        return "\n".join(lines)
