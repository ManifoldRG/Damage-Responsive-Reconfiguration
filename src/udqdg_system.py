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

    # ─── New Token-Based Algorithm Helpers ───

    def get_active_neighbors(self, module_id: str) -> List[str]:
        """Get active, non-faulty neighbors of a module."""
        return [n for n in self.get_neighbors(module_id)
                if self.modules[n].is_active and not self.modules[n].is_faulty]

    def is_movable(self, module_id: str) -> bool:
        """
        2-hop criticality test per paper Section III-C.

        u is movable if:
        - Leaf: |N̄_t(u)| = 1
        - Or: one of u's neighbor's 2-hop neighbors overlaps with another
          of u's neighbors. Equivalently:
          exists v in N_t(u) such that N_t_2hop(u) minus {u,v} intersects N_t(v)
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

        # Compute N̄_t^{(2)}(u) excluding u: all active modules within 2 hops
        two_hop = set()
        for w in neighbors:
            for x in self.get_active_neighbors(w):
                if x != module_id:
                    two_hop.add(x)

        # Check condition: ∃ v ∈ N̄(u) s.t. (two_hop \ {v}) ∩ N̄(v) ≠ ∅
        for v in neighbors:
            remaining = two_hop - {v}
            v_neighbors = set(self.get_active_neighbors(v))
            if remaining & v_neighbors:
                return True

        return False

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

    # ─── Phase 1: Token-Based Coagulation (Algorithm 1) ───

    def coagulation(
        self,
        fault_id: str,
        max_iterations: int = 1000,
        record_steps: bool = True
    ) -> Dict[str, Any]:
        """
        Phase 1: Decentralized coagulation via distress token propagation.

        Per paper Algorithm 1:
        - Modules detect inactive neighbors and generate distress tokens
        - Tokens propagate hop-by-hop with direction composition
        - Movable modules pivot toward closest token using alignment selection
        - Terminates when connected or no progress

        Token: {fault_id: direction_vector} where direction ξ points from
        the module toward the suspected fault location.
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
            "success": True
        }
        if record_steps:
            stats["steps"] = []

        # Token storage: {module_id: {fault_id: direction_vector ξ}}
        # ξ = approximate displacement from module toward fault
        tokens: Dict[str, Dict[str, np.ndarray]] = {
            mid: {} for mid, m in self.modules.items()
            if m.is_active and not m.is_faulty
        }

        # Oscillation prevention: track recent positions per module (tabu list)
        position_history: Dict[str, Set[Tuple]] = {}

        for iteration in range(max_iterations):
            stats["iterations"] = iteration + 1

            # Check connectivity (simulation-level stopping condition)
            if self.is_connected(active_only=True):
                stats["reconnected"] = True
                break

            # ── Token Generation + Propagation (double-buffered) ──
            next_tokens: Dict[str, Dict[str, np.ndarray]] = {
                mid: dict(toks) for mid, toks in tokens.items()
                if mid in self.modules and self.modules[mid].is_active
            }
            tokens_changed = False

            for u in list(tokens.keys()):
                if u not in self.modules or not self.modules[u].is_active:
                    continue

                # Generation: detect inactive/faulty neighbors
                for f in self.get_neighbors(u):
                    if not self.modules[f].is_active or self.modules[f].is_faulty:
                        edge_uf = self.edges.get((u, f))
                        if edge_uf:
                            xi = edge_uf.translation.copy()  # ρ_uf
                            f_key = f
                            if f_key not in next_tokens[u] or \
                               np.linalg.norm(xi) < np.linalg.norm(next_tokens[u][f_key]):
                                if f_key not in tokens.get(u, {}):
                                    tokens_changed = True
                                next_tokens[u][f_key] = xi

                # Propagation: broadcast current tokens to active neighbors
                for f_key, xi in tokens[u].items():
                    for w in self.get_active_neighbors(u):
                        if w not in next_tokens:
                            continue
                        edge_wu = self.edges.get((w, u))
                        if edge_wu:
                            rho_wu = edge_wu.translation  # displacement w→u
                            xi_w = rho_wu + xi
                            if f_key not in next_tokens[w] or \
                               np.linalg.norm(xi_w) < np.linalg.norm(next_tokens[w][f_key]):
                                if f_key not in tokens.get(w, {}):
                                    tokens_changed = True
                                next_tokens[w][f_key] = xi_w

            tokens = next_tokens

            # ── Movement Phase ──
            # Build candidate list: modules with tokens that pass criticality
            candidates = []
            for u, u_tokens in tokens.items():
                if not u_tokens:
                    continue
                if u not in self.modules or not self.modules[u].is_active:
                    continue
                # Closest token distance for priority ordering
                closest_f = min(u_tokens.keys(),
                                key=lambda f: np.linalg.norm(u_tokens[f]))
                dist = float(np.linalg.norm(u_tokens[closest_f]))
                candidates.append((u, dist))

            # Sort by distance to fault (closest first), tiebreak by ID
            candidates.sort(key=lambda x: (x[1], x[0]))

            # Execute moves sequentially, re-checking movability each time
            moves_this_iteration = 0
            for u, _ in candidates:
                if not self.modules[u].is_active:
                    continue
                if not self.is_movable(u):
                    continue

                u_tokens = tokens.get(u, {})
                if not u_tokens:
                    continue

                # Select closest token: argmin ||ξ||
                closest_f = min(u_tokens.keys(),
                                key=lambda f: np.linalg.norm(u_tokens[f]))
                xi_star = u_tokens[closest_f]

                # Select pivot by alignment with ξ*
                pivot = self.select_pivot_by_alignment(u, xi_star)
                if pivot is None:
                    continue

                from_pos = self.modules[u].position.copy()
                dest_pos = from_pos + pivot[4]
                dest_key = tuple(np.round(dest_pos).astype(int))

                # Oscillation prevention: skip if returning to any recent position
                if dest_key in position_history.get(u, set()):
                    continue

                # Collision check: skip if destination already occupied
                if self._position_is_occupied(dest_pos, exclude_module=u):
                    continue

                # Stranding prevention: skip if move would isolate a leaf neighbor
                would_strand = False
                for n in self.get_active_neighbors(u):
                    n_nbrs = self.get_active_neighbors(n)
                    if len(n_nbrs) == 1 and n_nbrs[0] == u:
                        would_strand = True
                        break
                if would_strand:
                    continue

                pivot_type, mid, param1, param2, delta_p = pivot

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
                    actual_delta = to_pos - from_pos

                    # Track position history for oscillation prevention
                    if mid not in position_history:
                        position_history[mid] = set()
                    position_history[mid].add(tuple(np.round(from_pos).astype(int)))

                    # Update token directions after move: ξ ← ξ - Δp
                    if mid in tokens:
                        for f_key in tokens[mid]:
                            tokens[mid][f_key] = tokens[mid][f_key] - actual_delta

                    if record_steps:
                        stats["steps"].append(PivotStep(
                            iteration=iteration + 1,
                            pivot_type=pivot_type,
                            module_id=mid,
                            param1=param1,
                            param2=param2,
                            from_pos=from_pos,
                            to_pos=to_pos
                        ))

            # Form new connections between adjacent modules
            new_conn = self._form_new_connections()
            stats["new_connections_formed"] += new_conn

            # Prune tokens for inactive modules
            tokens = {mid: toks for mid, toks in tokens.items()
                      if mid in self.modules and self.modules[mid].is_active}

            # Stop if no progress (no moves, no new connections, no new tokens)
            if moves_this_iteration == 0 and new_conn == 0 and not tokens_changed:
                break

        stats["modules_responded"] = list(stats["modules_responded"])
        return stats

    # ─── Phase 2: Token-Based Restructuring (Algorithm 2) ───

    def restructuring(
        self,
        pre_damage_neighbors: Dict[str, Dict[str, np.ndarray]],
        original_positions: Optional[Dict[str, np.ndarray]] = None,
        max_iterations: int = 100,
        record_steps: bool = True
    ) -> Dict[str, Any]:
        """
        Phase 2: Position restoration via displacement-guided movement.

        Each module tries to return to its pre-damage position.
        Modules with largest displacement move first (greedy).
        Movement uses alignment-based pivot selection toward original position.
        """
        stats = {
            "iterations": 0,
            "restoration_moves": 0,
            "success": False
        }
        if record_steps:
            stats["steps"] = []

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
        result = {
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
            record_steps=record_steps
        )
        result["phase1"] = phase1_stats
        result["total_moves"] = phase1_stats.get("total_moves", 0)

        if not phase1_stats.get("reconnected", False):
            result["overall_success"] = False
            return result

        # Phase 2: Restructuring
        if restore_positions:
            phase2_stats = self.restructuring(
                pre_damage_neighbors=pre_damage_neighbors,
                original_positions=original_positions,
                max_iterations=max_phase2_iterations,
                record_steps=record_steps
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
