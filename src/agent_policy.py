"""
Decentralized, asynchronous agent policies for modular spacecraft.

Each module is an independent agent that:
- Discovers neighbors by physical proximity (bonds in PyBullet)
- Receives/propagates tokens (direction vectors)
- Decides locally whether to move based on its token
- Picks a target lattice position aligned with its token
- Physically pivots to that position via torque
- Waits for pivot convergence (see ``BulletSimulator.is_pivot_complete``)
  before acting again

Token forwarding has a 1-second processing delay.
"""

import random
import numpy as np
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from loguru import logger


def bond_graph_neighbors(bond_matrix: np.ndarray, n: int, body_idx: int) -> List[int]:
    """Neighbors of ``body_idx`` in the undirected graph defined by a boolean bond matrix."""
    return [j for j in range(n) if j != body_idx and bond_matrix[body_idx, j]]


_LATTICE_DELTAS = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
)

# Eligibility: angle between pivot–axis arm and axis→target leg (same vectors as
# pick_target).  Perpendicular lattice steps should give cos≈0 for corners.
# Old threshold 0.99 only rejected within ~8° of 0°/180° (~171° still passed).
_CORNER_PIVOT_MAX_ABS_COS = 0.35  # ~90° family: reject |cos| above this
_LATERAL_PIVOT_MAX_ABS_COS = 0.92  # reject near-colinear (within ~23° of 0°/180°)


def _lateral_axis_handoff_step_cardinal_for_axis(
    sim,
    axis_idx: int,
    delta_axis_to_handoff_world: np.ndarray,
) -> bool:
    """True if axis→handoff COM vector has nominal length and is one lattice
    axis in the *axis* body frame.

    Lateral targets are built in the axis lattice frame (same as ``target_cell``
    rounding). Using the *pivot* frame here wrongly rejected laterals after a
    corner reoriented the pivot while the axis–handoff bond stayed lattice-aligned.
    """
    nom = float(sim.NOMINAL_DIST)
    d = np.asarray(delta_axis_to_handoff_world, dtype=float).reshape(3)
    if np.linalg.norm(d) < 0.5 * nom:
        return False
    if abs(np.linalg.norm(d) - nom) > 0.15 * nom:
        return False
    R = sim.body_rotation_matrix(axis_idx)
    d_loc = R.T @ d
    for dir6 in sim.CONNECTOR_DIRS:
        if np.linalg.norm(d_loc - nom * dir6) < 0.12 * nom:
            return True
    return False


def _cube_lateral_swept_clear(
    occupied: set,
    my_cell: Tuple[int, int, int],
    target_cell: Tuple[int, int, int],
    arm_axis_local: np.ndarray,
    R_axis: np.ndarray,
    nom: float,
) -> bool:
    """For cube modules, a 90° lever around the shared edge sweeps a quarter-disk
    that intrudes into the two cells "above" the start and target cells, where
    "above" is the arm direction (axis→pivot) extended by one lattice step.

    Reject the candidate if either of those cells is occupied.
    """
    arm_loc = R_axis.T @ arm_axis_local / nom
    arm_unit = np.round(arm_loc).astype(int)
    if int(np.sum(np.abs(arm_unit))) != 1:
        # Non-cardinal arm in axis frame: skip the test (be permissive — caller
        # may still reject via other checks).
        return True
    above_pivot = tuple((np.array(my_cell, dtype=int) + arm_unit).tolist())
    above_target = tuple((np.array(target_cell, dtype=int) + arm_unit).tolist())
    if above_pivot in occupied or above_target in occupied:
        return False
    return True


def _lattice_delta_perpendicular_to_arm(
    delta: Tuple[int, int, int],
    arm_world: np.ndarray,
    R_axis: np.ndarray,
) -> bool:
    """True if world step ``nom * R_axis @ delta`` is ⟂ ``arm_world`` (pivot−axis).

    Corner / lateral lattice moves are expressed in the *axis* body frame;
    only the four cardinal steps perpendicular to the current attachment in
    world frame are eligible (90° family).
    """
    an = np.linalg.norm(arm_world)
    if an < 1e-12:
        return False
    step_w = R_axis @ np.array(delta, dtype=float)
    return abs(np.dot(step_w, arm_world)) / an < 0.1


class ModuleState(Enum):
    IDLE = auto()           # Waiting, no token
    HAS_TOKEN = auto()      # Has a token, deciding what to do
    PROCESSING = auto()     # Processing delay before forwarding token
    PIVOTING = auto()       # Physically moving to target
    REVERSING = auto()      # Reversing to pre-pivot position after collision/timeout
    WAITING = auto()        # Post-reversal random cooldown before resuming IDLE


@dataclass
class Token:
    """Direction vector pointing toward fault (phase 1) or empty slot (phase 2)."""
    direction: np.ndarray   # relative vector
    source_id: str          # who generated/forwarded this token


@dataclass
class DiscoverMessage:
    """Flood phase message for distributed connected-component discovery."""
    flood_id: int
    initiator_mid: str
    sender_mid: str


@dataclass
class EchoMessage:
    """Echo phase message carrying aggregated component IDs back to the root."""
    flood_id: int
    initiator_mid: str
    sender_mid: str
    component_ids: Set[str]


@dataclass
class FloodParticipation:
    """Per-agent state for participating in one flood/echo round."""
    flood_id: int
    parent_mid: Optional[str]
    children: Set[str] = field(default_factory=set)
    pending_echoes: Set[str] = field(default_factory=set)
    aggregated_component: Set[str] = field(default_factory=set)


@dataclass
class ModuleAgent:
    """
    Independent agent for one module.

    Each agent maintains its own local view and acts asynchronously.
    """
    module_id: str
    body_idx: int                           # PyBullet body index
    state: ModuleState = ModuleState.IDLE
    token: Optional[Token] = None           # current held token
    incoming_tokens: List[Token] = field(default_factory=list)
    target_pos: Optional[np.ndarray] = None  # cached world goal (for logs / legacy)
    target_pos_local: Optional[np.ndarray] = None  # goal COM offset in lattice_ref frame
    lattice_ref_body_idx: Optional[int] = None     # frozen for maneuver (usually first axis)
    attract_body_idx: Optional[int] = None
    attract_connector: Optional[int] = None
    pivot_axis_idx: Optional[int] = None    # which neighbor we're torquing against
    pivot_type: Optional[str] = None        # "corner" or "lateral"
    handoff_idx: Optional[int] = None       # body idx of handoff module (lateral only)
    handoff_done: bool = False              # whether lateral handoff has occurred
    process_ready_time: float = 0.0         # sim time when processing delay ends
    is_faulty: bool = False
    position_history: Set[Tuple] = field(default_factory=set)  # anti-oscillation
    pre_pivot_pos_local: Optional[np.ndarray] = None   # lattice pos before pivot (for reversal)
    pre_pivot_lattice_ref: Optional[int] = None        # ref body for pre_pivot_pos_local
    pending_retry: Optional[Tuple] = None              # (target_local, axis, type, handoff)
    wait_until: float = 0.0                            # sim time when WAITING expires
    token_hold_until: float = 0.0                        # sim time: stay IDLE before acting on token
    moving_token_received_tick: int = -999               # tick when last "moving" token was received
    reversal_handoff_idx: Optional[int] = None           # original axis to bond-switch to during lateral reversal
    discover_inbox: List["DiscoverMessage"] = field(default_factory=list)
    echo_inbox: List["EchoMessage"] = field(default_factory=list)
    flood_participations: Dict[int, "FloodParticipation"] = field(default_factory=dict)


def _token_origin_world_pos(
    token: Token,
    pos: np.ndarray,
    agents: Dict[str, ModuleAgent],
    *,
    fault_id: Optional[str] = None,
    fault_body_idx: Optional[int] = None,
    fidx_to_fid: Optional[Dict[int, str]] = None,
) -> Optional[np.ndarray]:
    """World position of the module (or fault body) that emitted the token."""
    if fidx_to_fid:
        fid_to_fidx = {v: k for k, v in fidx_to_fid.items()}
        if token.source_id in fid_to_fidx:
            idx = fid_to_fidx[token.source_id]
            if 0 <= idx < len(pos):
                return pos[idx].copy()
            return None
    if fault_id is not None and token.source_id == fault_id:
        if fault_body_idx is not None and 0 <= fault_body_idx < len(pos):
            return pos[fault_body_idx].copy()
        return None
    if token.source_id in agents:
        idx = agents[token.source_id].body_idx
        if 0 <= idx < len(pos):
            return pos[idx].copy()
    return None


# (target_pos_local, axis_body_idx, pivot_type, handoff_idx or None for corner)
PivotPick = Tuple[np.ndarray, int, str, Optional[int]]


class DecentralizedCoagulation:
    """
    Async decentralized coagulation (phase 1).

    Drives an event loop: advance physics, then let each idle agent
    check its state and act. Agents discover neighbors from the
    physics world, not from a graph data structure.
    """

    TOKEN_PROCESS_DELAY = 1.0   # seconds before forwarding a token
    BOND_THRESHOLD = 1.05       # physical distance gate for bond creation

    # Pivot concurrency control.
    #   None  -> global lock: only one module may be PIVOTING at a time
    #            (legacy default, used by the 5-module demo and AsyncSimRunner).
    #   int k -> local  k-hop exclusion: a module may start a pivot iff no body
    #            within k bonded hops is currently PIVOTING.
    PIVOT_EXCLUSION_RADIUS: Optional[int] = None

    # If True, the fault body is a valid pivot axis / lateral handoff / post-
    # pivot proximity-bond neighbor. Default False matches legacy behavior
    # where the fault is treated as an unreliable anchor. Set True when the
    # fault is a real, full-mass, collision-on body (e.g. line-fault demo) so
    # modules can "swing around" it and close the loop.
    ALLOW_FAULT_AS_PIVOT_NEIGHBOR: bool = True

    # Probability (per tick) of accepting a random safe move when no
    # distance-reducing move exists.  Set > 0 to break symmetric deadlocks
    # (e.g. tie-fighter degeneracy on a center-fault line).
    TEMPERATURE: float = 0.01

    # Flood/echo distributed component-discovery protocol. When True
    # (default), fault-adjacent modules withhold tokens until a flood/echo
    # round completes and stop emitting once their local component grows
    # ("fault resolved"). When False, mirrors the non-bullet MC: tokens are
    # emitted every TOKEN_GEN_INTERVAL unconditionally; coag termination is
    # purely global is_connected(). Useful for diagnosing whether the
    # flood/echo gating slows or destabilizes multi-fault scenarios.
    USE_FLOOD_ECHO: bool = True

    def __init__(self, sim, fault_id: str, module_ids: List[str],
                 body_indices: Dict[str, int]):
        """
        Args:
            sim: BulletSimulator instance
            fault_id: ID of the faulty module (removed from world)
            module_ids: list of active (non-faulty) module IDs
            body_indices: {module_id: body_idx} mapping
        """
        self.sim = sim
        self.fault_id = fault_id

        # Create an agent for each active module
        self.agents: Dict[str, ModuleAgent] = {}
        for mid in module_ids:
            self.agents[mid] = ModuleAgent(
                module_id=mid,
                body_idx=body_indices[mid],
            )

        # Reverse mapping: body_idx -> module_id
        self._idx_to_mid: Dict[int, str] = {v: k for k, v in body_indices.items()}

        # Fault-adjacent agents and their directions
        self._fault_adjacent: Set[str] = set()
        self._fault_directions: Dict[str, np.ndarray] = {}
        self._last_token_gen_time: float = -999.0
        self.TOKEN_GEN_INTERVAL = 1.0  # re-emit tokens every second

        # Statistics
        self.total_moves = 0
        self.successful_moves = 0
        self.move_log: List[Dict] = []
        self._tick_count: int = 0

        # Flood/echo protocol state
        self._flood_counter: int = 0
        self._known_components: Dict[str, FrozenSet[str]] = {}
        self._fault_resolved_mids: Set[str] = set()
        self._pending_floods: Dict[str, int] = {}
        self.FLOOD_ECHO_INTERVAL: float = 5.0
        self._last_flood_time: Dict[str, float] = {}
        self._awaiting_first_flood: Set[str] = set()

    def _decision_graph_neighbors(self, body_idx: int) -> List[int]:
        """Bond neighbors for topology checks: frozen bond matrix at tick start, else live."""
        bm = getattr(self, "_decision_bond_matrix", None)
        if bm is None:
            bm = self.sim.get_bond_matrix()
        return bond_graph_neighbors(bm, self.sim.N, body_idx)

    def _bfs_reachable(self, start_idx: int, max_depth: int,
                       excluded: Set[int]) -> Set[int]:
        """BFS on the decision-time bond graph (snapshot during tick, else live)."""
        visited = {start_idx}
        frontier = {start_idx}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for nbr in self._decision_graph_neighbors(node):
                    if nbr not in excluded and nbr not in visited:
                        next_frontier.add(nbr)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(start_idx)
        return visited


    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        """
        Two-graph movability test.

        **Leaf check** uses the full physical graph (fault included) so that
        fault-adjacent modules are not mistaken for leaves.

        **Articulation-point check** uses the active-only subgraph (fault
        excluded from traversal) so that paths through the passive fault
        body are not counted as valid alternative routes.
        """
        fault_idxs = self._get_fault_idxs()

        all_neighbors = self._decision_graph_neighbors(body_idx)
        if len(all_neighbors) == 0:
            return False
        if len(all_neighbors) == 1:
            return True

        active_neighbors = [n for n in all_neighbors if n not in fault_idxs]
        if len(active_neighbors) == 0:
            return False

        for v in active_neighbors:
            excluded = {body_idx, v} | fault_idxs
            reachable_without_v = set()
            for w in active_neighbors:
                if w == v:
                    continue
                reachable_without_v |= self._bfs_reachable(
                    w, safety_radius - 1, excluded)
            v_neighbors = set(self._decision_graph_neighbors(v)) - fault_idxs
            if not (reachable_without_v & v_neighbors):
                return False

        return True

    def set_fault_adjacent(self, adjacent_mids: List[str],
                           fault_body_idx: int):
        """Register fault-adjacent modules and the fault body index."""
        self._fault_adjacent = set(adjacent_mids)
        self._fault_body_idx = fault_body_idx
        self._fault_body_idxs: Set[int] = {fault_body_idx}
        self._fault_ids: Set[str] = {self.fault_id}
        self._adj_to_fault_idx: Dict[str, int] = {
            mid: fault_body_idx for mid in adjacent_mids}
        self._fidx_to_fid: Dict[int, str] = {fault_body_idx: self.fault_id}
        self._seed_flood_echo_state(adjacent_mids)

    def set_multi_fault_adjacent(
        self,
        fault_ids: List[str],
        fault_body_idxs: List[int],
        adjacent_map: Dict[str, int],
    ):
        """Register multiple faults and their adjacent modules.

        Args:
            fault_ids: IDs of all faulty modules.
            fault_body_idxs: body indices of all faulty modules.
            adjacent_map: {active_module_id: fault_body_idx} mapping each
                fault-adjacent module to the fault body it neighbours.
        """
        self._fault_ids = set(fault_ids)
        self._fault_body_idxs = set(fault_body_idxs)
        self._fault_body_idx = fault_body_idxs[0] if fault_body_idxs else None
        self._fault_adjacent = set(adjacent_map.keys())
        self._adj_to_fault_idx = dict(adjacent_map)
        self._fidx_to_fid = dict(zip(fault_body_idxs, fault_ids))
        self._seed_flood_echo_state(list(adjacent_map.keys()))

    def _seed_flood_echo_state(self, adjacent_mids: List[str]):
        """Initialize flood/echo protocol for newly registered fault-adjacent modules.

        Preserves resolved status for modules that were already resolved from
        a prior call (e.g. stall-recovery re-registration in bullet_bridge).
        """
        if not self.USE_FLOOD_ECHO:
            # Skip token-gating: emit tokens immediately from every fault-
            # adjacent module on the next _generate_fault_tokens tick.
            return
        for mid in adjacent_mids:
            if mid in self._fault_resolved_mids:
                continue
            self._awaiting_first_flood.add(mid)
            self._last_flood_time[mid] = -999.0

    def _get_fault_idxs(self) -> Set[int]:
        """Return the set of all fault body indices."""
        idxs = getattr(self, "_fault_body_idxs", None)
        if idxs is not None:
            return idxs
        single = getattr(self, "_fault_body_idx", None)
        if single is not None:
            return {single}
        return set()

    def _generate_fault_tokens(self):
        """Fault-adjacent modules emit tokens every TOKEN_GEN_INTERVAL seconds.

        Token direction = displacement from agent to fault in the emitter's
        body frame (drift-invariant).  Recomputed each time.

        Multi-fault: each adjacent module points at its specific fault body
        via ``_adj_to_fault_idx``.
        """
        if self.sim.sim_time - self._last_token_gen_time < self.TOKEN_GEN_INTERVAL:
            return
        self._last_token_gen_time = self.sim.sim_time

        pos = self.sim.get_positions()
        adj_map = getattr(self, "_adj_to_fault_idx", None)

        for mid in self._fault_adjacent:
            if mid in self._fault_resolved_mids:
                continue
            if mid in self._awaiting_first_flood:
                continue
            if mid not in self.agents:
                continue
            agent = self.agents[mid]
            if agent.state != ModuleState.IDLE:
                continue
            if adj_map is not None and mid in adj_map:
                f_idx = adj_map[mid]
            else:
                f_idx = self._fault_body_idx
            fault_pos = pos[f_idx]
            R = self.sim.body_rotation_matrix(agent.body_idx)
            direction = R.T @ (fault_pos - pos[agent.body_idx])
            fidx_to_fid = getattr(self, "_fidx_to_fid", {})
            source = fidx_to_fid.get(f_idx, self.fault_id)
            agent.incoming_tokens.append(
                Token(direction=direction.copy(), source_id=source))

    # ------------------------------------------------------------------
    # Flood / Echo distributed component-discovery protocol
    # ------------------------------------------------------------------

    def _initiate_flood(self, mid: str):
        """Start a new flood/echo round from fault-adjacent module *mid*."""
        if mid not in self.agents:
            return
        agent = self.agents[mid]

        self._flood_counter += 1
        fid = self._flood_counter
        self._pending_floods[mid] = fid

        bond_matrix = self.sim.get_bond_matrix()
        fault_idxs = self._get_fault_idxs()
        neighbors = [
            j for j in range(self.sim.N)
            if j != agent.body_idx
            and bond_matrix[agent.body_idx, j]
            and j not in fault_idxs
        ]

        children: Set[str] = set()
        for j in neighbors:
            n_mid = self._idx_to_mid.get(j)
            if n_mid and n_mid in self.agents:
                self.agents[n_mid].discover_inbox.append(
                    DiscoverMessage(flood_id=fid, initiator_mid=mid,
                                    sender_mid=mid))
                children.add(n_mid)

        participation = FloodParticipation(
            flood_id=fid,
            parent_mid=None,
            children=set(children),
            pending_echoes=set(children),
            aggregated_component={mid},
        )
        agent.flood_participations[fid] = participation

        if not children:
            self._on_flood_complete(mid, frozenset({mid}))

        self._last_flood_time[mid] = self.sim.sim_time

    def _maybe_initiate_floods(self):
        """Initiate flood/echo rounds for fault-adjacent modules on heartbeat."""
        for mid in self._fault_adjacent:
            if mid in self._fault_resolved_mids:
                continue
            if mid in self._pending_floods:
                continue
            last = self._last_flood_time.get(mid, -999.0)
            if self.sim.sim_time - last < self.FLOOD_ECHO_INTERVAL:
                continue
            self._initiate_flood(mid)

    def _process_discover_messages(self):
        """Deliver one hop of the flood wave: each agent processes its discover inbox."""
        fault_idxs = self._get_fault_idxs()
        bond_matrix = self.sim.get_bond_matrix()

        for mid, agent in self.agents.items():
            if not agent.discover_inbox:
                continue
            messages = list(agent.discover_inbox)
            agent.discover_inbox.clear()

            for msg in messages:
                if msg.flood_id in agent.flood_participations:
                    continue

                neighbors = [
                    j for j in range(self.sim.N)
                    if j != agent.body_idx
                    and bond_matrix[agent.body_idx, j]
                    and j not in fault_idxs
                ]
                children: Set[str] = set()
                for j in neighbors:
                    n_mid = self._idx_to_mid.get(j)
                    if n_mid and n_mid in self.agents and n_mid != msg.sender_mid:
                        if msg.flood_id not in self.agents[n_mid].flood_participations:
                            self.agents[n_mid].discover_inbox.append(
                                DiscoverMessage(
                                    flood_id=msg.flood_id,
                                    initiator_mid=msg.initiator_mid,
                                    sender_mid=mid,
                                ))
                            children.add(n_mid)

                participation = FloodParticipation(
                    flood_id=msg.flood_id,
                    parent_mid=msg.sender_mid,
                    children=set(children),
                    pending_echoes=set(children),
                    aggregated_component={mid},
                )
                agent.flood_participations[msg.flood_id] = participation

                if not children:
                    parent_agent = self.agents.get(msg.sender_mid)
                    if parent_agent is not None:
                        parent_agent.echo_inbox.append(
                            EchoMessage(
                                flood_id=msg.flood_id,
                                initiator_mid=msg.initiator_mid,
                                sender_mid=mid,
                                component_ids={mid},
                            ))

    def _process_echo_messages(self):
        """Deliver one hop of the echo wave: aggregate component IDs toward root."""
        for mid, agent in self.agents.items():
            if not agent.echo_inbox:
                continue
            messages = list(agent.echo_inbox)
            agent.echo_inbox.clear()

            for msg in messages:
                part = agent.flood_participations.get(msg.flood_id)
                if part is None:
                    continue

                part.aggregated_component |= msg.component_ids
                part.pending_echoes.discard(msg.sender_mid)

                if not part.pending_echoes:
                    if part.parent_mid is None:
                        self._on_flood_complete(
                            mid, frozenset(part.aggregated_component))
                    else:
                        parent_agent = self.agents.get(part.parent_mid)
                        if parent_agent is not None:
                            parent_agent.echo_inbox.append(
                                EchoMessage(
                                    flood_id=msg.flood_id,
                                    initiator_mid=msg.initiator_mid,
                                    sender_mid=mid,
                                    component_ids=set(part.aggregated_component),
                                ))

    def _on_flood_complete(self, mid: str, component: FrozenSet[str]):
        """Handle completion of a flood/echo round for fault-adjacent module *mid*."""
        self._pending_floods.pop(mid, None)

        if mid in self._awaiting_first_flood:
            self._known_components[mid] = component
            self._awaiting_first_flood.discard(mid)
            logger.debug("Flood/echo baseline for {}: {} modules", mid, len(component))
            return

        prev = self._known_components.get(mid, frozenset())
        if component > prev:
            self._fault_resolved_mids.add(mid)
            logger.info(
                "Flood/echo: {} component grew ({} -> {}), fault resolved",
                mid, len(prev), len(component))
        else:
            self._known_components[mid] = component

    def _cleanup_stale_flood_participations(self):
        """Remove flood participation records for completed floods."""
        active_flood_ids: Set[int] = set(self._pending_floods.values())
        for agent in self.agents.values():
            stale = [fid for fid in agent.flood_participations
                     if fid not in active_flood_ids]
            for fid in stale:
                del agent.flood_participations[fid]

    def get_physical_neighbors(self, body_idx: int) -> List[int]:
        """Discover bonded neighbors from PyBullet bond state."""
        neighbors = []
        bond_matrix = self.sim.get_bond_matrix()
        for j in range(self.sim.N):
            if j != body_idx and bond_matrix[body_idx, j]:
                neighbors.append(j)
        return neighbors

    def get_nearby_lattice_positions(self, body_idx: int) -> List[np.ndarray]:
        """
        Get unoccupied lattice positions near this module's neighbors.

        Looks at neighbors and neighbors' neighbors to find lattice
        positions that are not currently occupied.
        """
        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        neighbors = self.get_physical_neighbors(body_idx)

        # Collect all occupied positions (snapped to lattice)
        occupied = set()
        for i in range(self.sim.N):
            lp = tuple(np.round(pos[i]).astype(int))
            occupied.add(lp)

        # Candidate positions: lattice positions adjacent to my neighbors
        # (and neighbors' neighbors) that are unoccupied
        candidate_set = set()
        check_indices = set(neighbors)
        for n_idx in neighbors:
            for nn_idx in self.get_physical_neighbors(n_idx):
                check_indices.add(nn_idx)

        for idx in check_indices:
            nbr_pos = np.round(pos[idx]).astype(int)
            # 6-connected lattice neighbors
            for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                candidate = tuple(nbr_pos + np.array(delta))
                if candidate not in occupied:
                    candidate_set.add(candidate)

        # Also include positions adjacent to myself
        my_lattice = np.round(my_pos).astype(int)
        for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            candidate = tuple(my_lattice + np.array(delta))
            if candidate not in occupied:
                candidate_set.add(candidate)

        return [np.array(c, dtype=float) for c in candidate_set]

    def _origin_world_for_pick_target(
            self, agent: ModuleAgent, pos: np.ndarray) -> Optional[np.ndarray]:
        """World point used to score pivot candidates (minimize goal COM distance).

        Default: emitter COM from ``_token_origin_world_pos`` (fault body or
        ``source_id`` module). Subclasses may override for e.g. a virtual goal
        carried in assembly body frame (see five-module pivot agent demo).
        """
        if agent.token is None:
            return None
        return _token_origin_world_pos(
            agent.token,
            pos,
            self.agents,
            fault_id=self.fault_id,
            fault_body_idx=getattr(self, "_fault_body_idx", None),
            fidx_to_fid=getattr(self, "_fidx_to_fid", None),
        )

    def _on_no_pick_target(self, agent: ModuleAgent) -> None:
        """Hook when a movable agent with a token has no ``pick_target``.

        Default: no-op. Subclasses may e.g. advance a turn counter before the
        agent enters ``PROCESSING`` and forwards the token.
        """

    def _pick_target_occupancy_skip_body_indices(self) -> Set[int]:
        """Bodies omitted when building the occupied lattice set in ``pick_target``.

        Default: none. A ghost goal body (collision-off, same cell as the nominal
        empty slot) should be listed so pivots **into** that cell are not
        rejected as ``target_cell in occupied``.
        """
        return set()

    def _ghost_goal_world_pos_for_axial_corner(
            self, pos: np.ndarray) -> Optional[np.ndarray]:
        """If not None, allow a **colinear** (spine) corner step into that cell.

        Standard corners require a lattice step perpendicular to ``pivot−axis``;
        the last move onto a ghost goal at the nominal chain tip is parallel to
        that arm and is otherwise never enumerated.
        """
        return None

    def pick_target(
            self,
            agent: ModuleAgent,
            *,
            require_closer_to_origin: bool = False,
            random_choice: bool = False,
    ) -> Optional[PivotPick]:
        """
        Enumerate eligible **corner** and **lateral** pivots, then pick the
        candidate whose goal COM is **closest** to the scoring origin from
        ``_origin_world_for_pick_target`` (default: token emitter COM). Pivot
        type and handoff index are fixed at selection time.

        If ``require_closer_to_origin`` is True, drop any candidate whose goal COM is not **strictly closer** to the origin than the pivoting module's
        current COM (so the agent stops when no pivot improves distance).

        If ``random_choice`` is True, return a uniformly random candidate from
        the eligible set instead of the best-scoring one. Used by the
        temperature-based deadlock breaker to explore non-greedy moves.

        **Corner** (for each graph neighbor ``axis_idx`` of the pivot):

        - Step ``nom * delta`` in the **axis** body frame; the world step
          ``R_axis @ delta`` must be perpendicular to ``arm = pivot − axis``
          (``_lattice_delta_perpendicular_to_arm``).
        - Target cell (axis lattice coords) must not be occupied and must not
          be the pivot's current cell; ``(axis_idx, target_cell)`` must not
          appear in ``agent.position_history``.
        - ``|cos ∠(arm, target − axis)| ≤ _CORNER_PIVOT_MAX_ABS_COS`` (0.35) so
          the corner is **~90°**, not ~0° or ~180° (the old 0.99 bound allowed
          ~171°).

        **Not** applied: rejecting a target that lies **on the chain backbone**
        through ``axis`` (e.g. one more step past M2 colinear with M1–M2 when
        the pivot is still **sideways** to that line). Such a cell is still a
        valid **90°** corner from a sideways arm, so it competes with other
        perpendiculars; only an explicit **spine** filter would forbid it.

        **Lateral:** handoff neighbor ``handoff_idx`` with axis–handoff step
        cardinal in the pivot frame; ``target = pos[handoff] + (pivot − axis)``;
        same occupancy/history checks and ``|cos| ≤ _LATERAL_PIVOT_MAX_ABS_COS``.

        **Tie-break:** higher score first, then lower ``axis_idx``, then lower
        ``handoff_idx`` (see ``scored.sort``).
        """
        if agent.token is None:
            return None

        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        token_dir = agent.token.direction
        if np.linalg.norm(token_dir) < 1e-8:
            return None

        origin = self._origin_world_for_pick_target(agent, pos)
        if origin is None:
            return None

        neighbors = self._decision_graph_neighbors(agent.body_idx)
        if not neighbors:
            return None

        fault_idxs = self._get_fault_idxs()
        # score = negative distance to token origin (maximize = minimize dist)
        scored: List[Tuple[float, np.ndarray, int, str, Optional[int]]] = []
        nom = float(self.sim.NOMINAL_DIST)
        module_shape = getattr(self.sim, "_module_shape", "sphere")

        for axis_idx in neighbors:
            if (fault_idxs and axis_idx in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            R = self.sim.body_rotation_matrix(axis_idx)
            p_ref = pos[axis_idx]
            occupied = set()
            skip_occ = self._pick_target_occupancy_skip_body_indices()
            for i in range(self.sim.N):
                if i in skip_occ:
                    continue
                u = R.T @ (pos[i] - p_ref) / nom
                occupied.add(tuple(np.round(u).astype(int)))
            my_cell = tuple(np.round(R.T @ (my_pos - p_ref) / nom).astype(int))

            arm_axis_to_pivot = my_pos - pos[axis_idx]

            gpos_ax = self._ghost_goal_world_pos_for_axial_corner(pos)
            ghost_goal_cell: Optional[Tuple[int, int, int]] = None
            if gpos_ax is not None:
                ghost_goal_cell = tuple(
                    np.round(R.T @ (gpos_ax - p_ref) / nom).astype(int))

            # Cube modules cannot do sphere-style corner pivots: a 90° lever
            # around the contact edge lands the cube at a cell *diagonal* to
            # axis (= the lateral-pivot destination), not at a perpendicular
            # neighbor of axis. Skip the corner enumeration entirely for cubes.
            corner_enabled = (module_shape != "cube")

            # ── Corner: 4 sites on axis — steps ⟂ (pivot − axis) in ref frame ──
            for delta in _LATTICE_DELTAS:
                if not corner_enabled:
                    break
                perp = _lattice_delta_perpendicular_to_arm(
                    delta, arm_axis_to_pivot, R)
                dw = nom * np.array(delta, dtype=float)
                target_world = p_ref + R @ dw
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                axial_into_ghost = (
                    ghost_goal_cell is not None
                    and target_cell == ghost_goal_cell)
                if not perp and not axial_into_ghost:
                    continue
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (axis_idx, target_cell) in agent.position_history:
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_tar_n = np.linalg.norm(r_target)
                if r_tar_n < 1e-12:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * r_tar_n + 1e-12)
                if axial_into_ghost:
                    if gpos_ax is None:
                        continue
                    d_tgt = float(np.linalg.norm(target_world - gpos_ax))
                    d_my = float(np.linalg.norm(my_pos - gpos_ax))
                    if d_tgt >= d_my - 1e-9:
                        continue
                else:
                    if abs(cos_angle) > _CORNER_PIVOT_MAX_ABS_COS:
                        continue
                dist_to_origin = float(np.linalg.norm(target_world - origin))
                score = -dist_to_origin
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "corner", None))

            # ── Lateral: copy axis–pivot offset to handoff (one φ from handoff COM) ──
            axis_neighbors = self._decision_graph_neighbors(axis_idx)
            for handoff_idx in axis_neighbors:
                if handoff_idx == agent.body_idx:
                    continue
                if (fault_idxs and handoff_idx in fault_idxs):
                    continue
                d_ah = pos[handoff_idx] - pos[axis_idx]
                if not _lateral_axis_handoff_step_cardinal_for_axis(
                        self.sim, axis_idx, d_ah):
                    continue
                target_world = pos[handoff_idx] + (my_pos - pos[axis_idx])
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (axis_idx, target_cell) in agent.position_history:
                    continue
                if np.linalg.norm(target_world - pos[axis_idx]) < 1.05:
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_vec_n = np.linalg.norm(r_vec)
                r_tar_n = np.linalg.norm(r_target)
                if r_vec_n < 1e-8 or r_tar_n < 1e-8:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (r_vec_n * r_tar_n)
                if abs(cos_angle) > _LATERAL_PIVOT_MAX_ABS_COS:
                    continue
                if module_shape == "cube":
                    if not _cube_lateral_swept_clear(
                            occupied, my_cell, target_cell,
                            arm_axis_to_pivot, R, nom):
                        continue
                dist_to_origin = float(np.linalg.norm(target_world - origin))
                score = -dist_to_origin
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "lateral", handoff_idx))

        if not scored:
            return None

        if require_closer_to_origin:
            current_d = float(np.linalg.norm(my_pos - origin))
            eps = 1e-9 * max(1.0, nom)
            scored = [s for s in scored if (-s[0]) < current_d - eps]
            if not scored:
                return None

        if random_choice:
            pick = random.choice(scored)
        else:
            scored.sort(key=lambda x: (-x[0], x[2], x[4] if x[4] is not None else -1))
            pick = scored[0]
        return pick[1].copy(), pick[2], pick[3], pick[4]

    def _propagate_moving_tokens(self):
        """Flood-fill moving tokens from PIVOTING/REVERSING agents."""
        if self.PIVOT_EXCLUSION_RADIUS is None:
            return
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        bm = self._decision_bond_matrix.copy()
        for a in self.agents.values():
            if a.state in _moving and a.pivot_axis_idx is not None:
                bm[a.body_idx, a.pivot_axis_idx] = True
                bm[a.pivot_axis_idx, a.body_idx] = True

        seeds: set = set()
        for a in self.agents.values():
            if a.state in _moving:
                a.moving_token_received_tick = self._tick_count
                seeds.add(a.body_idx)

        frontier = set(seeds)
        for _hop in range(self.PIVOT_EXCLUSION_RADIUS):
            nxt: set = set()
            for i in frontier:
                for j in range(self.sim.N):
                    if bm[i, j] and j not in seeds:
                        mid_j = self._idx_to_mid.get(j)
                        if mid_j and mid_j in self.agents:
                            self.agents[mid_j].moving_token_received_tick = self._tick_count
                        seeds.add(j)
                        nxt.add(j)
            frontier = nxt
            if not frontier:
                break

    def tick(self) -> bool:
        """
        Run one decision cycle for all agents.

        Called after each physics time slice. Each agent checks its state
        and acts if ready. Returns True if any agent is still active
        (pivoting or has tokens to process).
        """
        self._tick_count += 1

        # Flood/echo protocol: process messages, then check heartbeats.
        # Skipped entirely when USE_FLOOD_ECHO=False (matches non-bullet MC).
        if self.USE_FLOOD_ECHO:
            self._process_discover_messages()
            self._process_echo_messages()
            self._maybe_initiate_floods()
            if self._tick_count % 500 == 0:
                self._cleanup_stale_flood_participations()

        # Fault-adjacent modules emit tokens (skips resolved / awaiting-baseline)
        self._generate_fault_tokens()

        # One consistent bond graph for all move decisions this tick.
        # Includes the fault body so fault-adjacent modules are non-leaves.
        self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

        self._propagate_moving_tokens()

        any_active = False
        agent_items = list(self.agents.items())
        random.shuffle(agent_items)

        for mid, agent in agent_items:
            if agent.is_faulty:
                continue

            if agent.state == ModuleState.IDLE:
                neighbors = self._decision_graph_neighbors(agent.body_idx)
                bm = self._decision_bond_matrix
                pos = self.sim.get_positions()
                triangle_found = False
                for ni_idx in range(len(neighbors)):
                    for nj_idx in range(ni_idx + 1, len(neighbors)):
                        a, b = neighbors[ni_idx], neighbors[nj_idx]
                        if bm[a, b]:
                            d_a = float(np.linalg.norm(pos[agent.body_idx] - pos[a]))
                            d_b = float(np.linalg.norm(pos[agent.body_idx] - pos[b]))
                            if d_a <= d_b:
                                close, far = a, b
                            else:
                                close, far = b, a
                            self.sim.remove_bond(agent.body_idx, far)
                            self._decision_bond_matrix = self.sim.get_bond_matrix().copy()
                            self._start_corrective_pivot(agent, close)
                            self._propagate_moving_tokens()
                            triangle_found = True
                            any_active = True
                            break
                    if triangle_found:
                        break
                if triangle_found:
                    continue

            if agent.state == ModuleState.PIVOTING:
                # Lateral handoff: when the module reaches the handoff
                # module mid-pivot, bond to it and disconnect from axis.
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        # Bond-level handoff
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        # Physics constraint switch: restart pivot
                        # around the handoff module toward the target
                        self.sim.stop_pivot(agent.body_idx)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} lateral handoff: bonded to {}, "
                            "pivot switched from {}",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx),
                            self._idx_to_mid.get(agent.pivot_axis_idx,
                                                 agent.pivot_axis_idx))

                # Check if pivot converged
                if self.sim.is_pivot_complete(agent.body_idx):
                    collided = self.sim.pivot_collided(agent.body_idx)
                    timed_out = self.sim.pivot_timed_out(agent.body_idx)
                    needs_reversal = (
                        (collided or timed_out)
                        and agent.pre_pivot_pos_local is not None)

                    self.sim.stop_pivot(agent.body_idx)
                    if (agent.lattice_ref_body_idx is not None
                            and agent.target_pos_local is not None):
                        nom = float(self.sim.NOMINAL_DIST)
                        cell = tuple(
                            np.round(agent.target_pos_local / nom).astype(int))
                        agent.position_history.add(
                            (agent.lattice_ref_body_idx, cell))
                    self._reconnect_bonds(agent.body_idx)
                    self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

                    if needs_reversal:
                        agent.pending_retry = None
                        self._start_reversal(agent)
                        any_active = True
                        if collided:
                            logger.info(
                                "Module {} collision detected — reversing",
                                mid)
                        else:
                            logger.info(
                                "Module {} pivot timed out — reversing", mid)
                    else:
                        agent.state = ModuleState.IDLE
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.token = None
                        self.successful_moves += 1
                        logger.info("Module {} pivot complete", mid)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.REVERSING:
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        self.sim.stop_pivot(agent.body_idx)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} reversal handoff: bonded to {}, "
                            "pivot switched from {}",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx),
                            self._idx_to_mid.get(agent.pivot_axis_idx,
                                                 agent.pivot_axis_idx))

                if self.sim.is_pivot_complete(agent.body_idx):
                    rev_collided = self.sim.pivot_collided(agent.body_idx)
                    rev_timed_out = self.sim.pivot_timed_out(agent.body_idx)
                    self.sim.stop_pivot(agent.body_idx)
                    self._reconnect_bonds(agent.body_idx)
                    self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

                    if rev_collided or rev_timed_out:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None
                        self._emergency_snap_to_lattice(agent)
                        any_active = True
                        logger.info(
                            "Module {} reversal failed — snapping to "
                            "lattice", mid)
                    else:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None

                        cooldown = random.uniform(0, 10)
                        agent.wait_until = self.sim.sim_time + cooldown
                        agent.state = ModuleState.WAITING
                        any_active = True
                        logger.info(
                            "Module {} reversal complete — waiting "
                            "{:.1f}s", mid, cooldown)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.WAITING:
                if self.sim.sim_time >= agent.wait_until:
                    agent.state = ModuleState.IDLE
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.PROCESSING:
                # Wait for processing delay
                if self.sim.sim_time >= agent.process_ready_time:
                    # Forward token to bonded neighbors
                    self._forward_token(agent)
                    agent.state = ModuleState.IDLE
                    agent.token = None
                else:
                    any_active = True
                continue

            # IDLE state: check for incoming tokens
            if agent.incoming_tokens:
                best = min(agent.incoming_tokens,
                           key=lambda t: np.linalg.norm(t.direction))
                agent.token = best
                agent.incoming_tokens.clear()
                agent.token_hold_until = self.sim.sim_time + 1.0
                agent.state = ModuleState.HAS_TOKEN

            if agent.state == ModuleState.HAS_TOKEN:
                if self.sim.sim_time < agent.token_hold_until:
                    any_active = True
                    continue
                if not self.is_movable(agent.body_idx):
                    # Not safe to move -- forward the token instead
                    agent.state = ModuleState.PROCESSING
                    agent.process_ready_time = (self.sim.sim_time
                                                + self.TOKEN_PROCESS_DELAY)
                    any_active = True
                    continue

                # Pivot concurrency check.
                _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
                if self.PIVOT_EXCLUSION_RADIUS is None:
                    if any(a.state in _moving
                           for a in self.agents.values()):
                        any_active = True
                        continue
                else:
                    if (self._tick_count - agent.moving_token_received_tick
                            < self.PIVOT_EXCLUSION_RADIUS):
                        any_active = True
                        continue

                result = self.pick_target(agent, require_closer_to_origin=True)
                if result is not None:
                    target_pos_local, axis_idx, pivot_type, handoff_idx = result
                    self._start_pivot(
                        agent, target_pos_local, axis_idx,
                        pivot_type, handoff_idx)
                    self._propagate_moving_tokens()
                    any_active = True
                    continue

                if self.TEMPERATURE > 0 and random.random() < self.TEMPERATURE:
                    result = self.pick_target(
                        agent, require_closer_to_origin=False,
                        random_choice=True)
                    if result is not None:
                        target_pos_local, axis_idx, pivot_type, handoff_idx = result
                        self._start_pivot(
                            agent, target_pos_local, axis_idx,
                            pivot_type, handoff_idx)
                        self._propagate_moving_tokens()
                        any_active = True
                        continue

                self._on_no_pick_target(agent)
                agent.state = ModuleState.PROCESSING
                agent.process_ready_time = (self.sim.sim_time
                                            + self.TOKEN_PROCESS_DELAY)
                any_active = True
                continue

        # Also check if any tokens or flood/echo messages are still in flight
        for agent in self.agents.values():
            if (agent.state != ModuleState.IDLE or
                    agent.incoming_tokens or
                    agent.token is not None or
                    agent.discover_inbox or
                    agent.echo_inbox):
                any_active = True
                break

        if not any_active and self._pending_floods:
            any_active = True

        self._decision_bond_matrix = None
        return any_active

    def is_connected(self) -> bool:
        """Check if all active (non-faulty) modules form a single connected component.

        Excludes the fault body — it's physically present but doesn't
        count for algorithmic connectivity.
        """
        active_indices = set(a.body_idx for a in self.agents.values())
        if not active_indices:
            return True
        bond_matrix = self.sim.get_bond_matrix()
        start = next(iter(active_indices))
        visited = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for j in active_indices:
                if j not in visited and bond_matrix[node, j]:
                    visited.add(j)
                    queue.append(j)
        return visited == active_indices

    def _start_pivot(
        self,
        agent: ModuleAgent,
        target_pos_local: np.ndarray,
        axis_idx: int,
        pivot_type: str,
        handoff_idx: Optional[int],
    ):
        """Start a physical pivot for an agent.

        *pivot_type* and *handoff_idx* must match the maneuver chosen in
        ``pick_target`` (``handoff_idx`` is ``None`` for corner pivots).
        """
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        axis_pos = pos[axis_idx]
        lattice_ref_body_idx = axis_idx

        R_ax = self.sim.body_rotation_matrix(axis_idx)
        agent.pre_pivot_pos_local = (R_ax.T @ (my_pos - axis_pos)).copy()
        agent.pre_pivot_lattice_ref = axis_idx

        target_pos_local = np.asarray(target_pos_local, dtype=float).reshape(3)
        target_world = self.sim.target_world_from_local(
            lattice_ref_body_idx, target_pos_local)

        if pivot_type == "lateral" and handoff_idx is not None:
            attract_body = handoff_idx
            attract_conn = self.sim.lateral_handoff_attract_connector(
                axis_idx, agent.body_idx, handoff_idx)
        else:
            attract_body = axis_idx
            attract_conn = self.sim.nearest_connector(
                axis_idx, target_world - pos[axis_idx])

        # Remove bonds to non-axis neighbors regardless of pivot type
        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n in neighbors:
            if n != axis_idx:
                self.sim.remove_bond(agent.body_idx, n)

        if pivot_type == "corner":
            # Spring mode keeps COM–COM bond to axis; legacy uses point-to-point only.
            if not self.sim.USE_SPRING_BONDS:
                self.sim.remove_bond(agent.body_idx, axis_idx)
        # Lateral: keep axis bond for now; handoff will swap it mid-pivot

        # Compute rotation parameters
        r_vec = my_pos - axis_pos
        rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)

        r_target = target_world - axis_pos
        cos_angle = np.clip(
            np.dot(r_vec, r_target) / (np.linalg.norm(r_vec) *
                                        np.linalg.norm(r_target) + 1e-12),
            -1, 1)
        angle = np.arccos(cos_angle)

        kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

        self.sim.start_pivot(
            agent.body_idx, axis_idx, rot_axis, angle, kp, kd,
            duration=12.0,
            lattice_ref_body_idx=lattice_ref_body_idx,
            target_pos_local=target_pos_local,
            attract_body_idx=attract_body,
            attract_connector=attract_conn,
            pivot_type=pivot_type)

        agent.state = ModuleState.PIVOTING
        agent.target_pos = target_world.copy()
        agent.target_pos_local = target_pos_local.copy()
        agent.lattice_ref_body_idx = lattice_ref_body_idx
        agent.attract_body_idx = attract_body
        agent.attract_connector = attract_conn
        agent.pivot_axis_idx = axis_idx
        agent.pivot_type = pivot_type
        agent.handoff_idx = handoff_idx
        agent.handoff_done = False
        self.total_moves += 1

        self.move_log.append({
            "module": agent.module_id,
            "from": my_pos.tolist(),
            "to": target_world.tolist(),
            "axis": self._idx_to_mid.get(axis_idx, f"idx_{axis_idx}"),
            "sim_time": self.sim.sim_time,
            "pivot_type": pivot_type,
        })

        logger.info("Module {} starting {} pivot to {} (axis={}{})",
                     agent.module_id, pivot_type, target_world,
                     self._idx_to_mid.get(axis_idx, axis_idx),
                     f", handoff={self._idx_to_mid.get(handoff_idx, handoff_idx)}"
                     if handoff_idx is not None else "")

    def _start_reversal(self, agent: ModuleAgent):
        """Pivot the module back to its pre-pivot lattice position.

        For lateral pivots that already completed a handoff, the reversal
        is itself a lateral pivot in reverse: roll back on the current
        axis (handoff module) toward the original axis, bond-switch, then
        roll on the original axis back to pre_pivot_pos_local.
        """
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]

        neighbors = self.get_physical_neighbors(agent.body_idx)
        if not neighbors:
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            self.sim.create_bond(agent.body_idx, nearest)
            neighbors = [nearest]

        ref = agent.pre_pivot_lattice_ref
        target_local = agent.pre_pivot_pos_local
        if target_local is None:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None
            return

        original_axis = ref if ref is not None else None
        is_lateral_reversal = (
            agent.pivot_type == "lateral"
            and agent.handoff_done
            and original_axis is not None
            and original_axis != agent.pivot_axis_idx
        )

        best_ax = min(neighbors,
                      key=lambda n: np.linalg.norm(pos[n] - my_pos))
        if ref is None:
            ref = best_ax
        target_world = self.sim.target_world_from_local(ref, target_local)
        axis_pos = pos[best_ax]

        for n in neighbors:
            if n != best_ax:
                self.sim.remove_bond(agent.body_idx, n)

        if is_lateral_reversal:
            attract_conn = self.sim.lateral_handoff_attract_connector(
                best_ax, agent.body_idx, original_axis)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=original_axis,
                attract_connector=attract_conn,
                pivot_type="lateral",
                repel_idx=-1)

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = original_axis
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "lateral"
            agent.handoff_idx = original_axis
            agent.handoff_done = False
            agent.reversal_handoff_idx = original_axis
        else:
            attract_conn = self.sim.nearest_connector(
                best_ax, target_world - axis_pos)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=best_ax,
                attract_connector=attract_conn,
                pivot_type="corner")

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = best_ax
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "corner"
            agent.handoff_idx = None
            agent.handoff_done = False
            agent.reversal_handoff_idx = None

    def _emergency_snap_to_lattice(self, agent: ModuleAgent):
        """Ensure module is bonded and pivot to the nearest empty lattice site.

        Called when a reversal pivot fails (timeout/collision).  Guarantees
        the module keeps at least one bond and attempts a corner pivot to
        the closest unoccupied lattice cell so it ends up in a valid
        position.
        """
        neighbors = self.get_physical_neighbors(agent.body_idx)
        pos = self.sim.get_positions()

        if not neighbors:
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            self.sim.create_bond(agent.body_idx, nearest)
            neighbors = [nearest]

        axis_idx = min(neighbors,
                       key=lambda n: np.linalg.norm(
                           pos[n] - pos[agent.body_idx]))

        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)
        my_pos = pos[agent.body_idx]

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is not None:
            self._start_pivot(agent, best_target_local, axis_idx, "corner",
                              None)
            logger.info("Module {} emergency snap pivot around {}",
                        agent.module_id,
                        self._idx_to_mid.get(axis_idx, axis_idx))
        else:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None

    def _start_corrective_pivot(self, agent: ModuleAgent, axis_idx: int):
        """Corner pivot to nearest empty lattice site on *axis_idx* to fix a triangular bond."""
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        arm = my_pos - p_ref
        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            if not _lattice_delta_perpendicular_to_arm(delta, arm, R):
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is None:
            return

        self._start_pivot(agent, best_target_local, axis_idx, "corner", None)
        logger.info("Module {} corrective pivot (triangle fix) around {}",
                    agent.module_id, self._idx_to_mid.get(axis_idx, axis_idx))

    def _forward_token(self, agent: ModuleAgent):
        """Forward agent's token to all bonded neighbors."""
        if agent.token is None:
            return
        pos = self.sim.get_positions()
        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid is None or n_mid not in self.agents:
                continue
            n_agent = self.agents[n_mid]
            if n_agent.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
                continue
            my_pos = pos[agent.body_idx]
            nbr_pos = pos[n_idx]
            R_me = self.sim.body_rotation_matrix(agent.body_idx)
            R_nbr = self.sim.body_rotation_matrix(n_idx)
            world_dir = R_me @ agent.token.direction + (my_pos - nbr_pos)
            propagated_dir = R_nbr.T @ world_dir
            n_agent.incoming_tokens.append(
                Token(direction=propagated_dir, source_id=agent.module_id))

    def _reconnect_bonds(self, body_idx: int):
        """Reconnect bonds dictated by the policy after a pivot.

        Only creates bonds to the pivot axis module and to modules that
        occupy lattice-adjacent positions of the target, provided they
        are within the distance threshold.  No blind proximity scan.
        """
        agent = None
        for a in self.agents.values():
            if a.body_idx == body_idx:
                agent = a
                break
        if agent is None:
            return

        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        if (agent.lattice_ref_body_idx is not None
                and agent.target_pos_local is not None):
            target = self.sim.target_world_from_local(
                agent.lattice_ref_body_idx, agent.target_pos_local)
        else:
            target = agent.target_pos if agent.target_pos is not None else my_pos

        # Determine which body indices the policy says we should bond to:
        # 1) the pivot axis module
        # 2) any module at a ±1 lattice step from goal in *lattice ref* frame
        expected_neighbors: Set[int] = set()
        if agent.pivot_axis_idx is not None:
            expected_neighbors.add(agent.pivot_axis_idx)

        if agent.lattice_ref_body_idx is not None:
            ref = agent.lattice_ref_body_idx
            R = self.sim.body_rotation_matrix(ref)
            p_ref = pos[ref]
            nom = float(self.sim.NOMINAL_DIST)
            goal_cell = np.round(
                agent.target_pos_local / nom).astype(int)  # type: ignore[union-attr]
            for delta in _LATTICE_DELTAS:
                nbr_cell = (int(goal_cell[0] + delta[0]),
                            int(goal_cell[1] + delta[1]),
                            int(goal_cell[2] + delta[2]))
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    j_cell = tuple(np.round(
                        R.T @ (pos[j] - p_ref) / nom).astype(int))
                    if j_cell == nbr_cell:
                        expected_neighbors.add(j)
        else:
            target_key = tuple(np.round(target).astype(int))
            for delta in _LATTICE_DELTAS:
                nbr_key = (target_key[0] + delta[0],
                           target_key[1] + delta[1],
                           target_key[2] + delta[2])
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    if tuple(np.round(pos[j]).astype(int)) == nbr_key:
                        expected_neighbors.add(j)

        # Create bonds only for policy-approved neighbors within threshold
        for j in expected_neighbors:
            dist = np.linalg.norm(my_pos - pos[j])
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

        # Proximity: fixed joints to every other module within bond distance
        # (e.g. after rolling), excluding passive fault bodies.
        fault_idxs = self._get_fault_idxs()
        for j in range(self.sim.N):
            if j == body_idx:
                continue
            if (fault_idxs and j in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            dist = float(np.linalg.norm(my_pos - pos[j]))
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

    def _has_pivoting_neighbor_of_neighbor(self, body_idx: int) -> bool:
        """Check if any neighbor's neighbor is currently pivoting (2-hop exclusion)."""
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        neighbors = self.get_physical_neighbors(body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid and n_mid in self.agents:
                if self.agents[n_mid].state in _moving:
                    return True
            nn_indices = self.get_physical_neighbors(n_idx)
            for nn_idx in nn_indices:
                nn_mid = self._idx_to_mid.get(nn_idx)
                if nn_mid and nn_mid in self.agents:
                    if self.agents[nn_mid].state in _moving:
                        return True
        return False


class DecentralizedRestructuring:
    """
    Async decentralized restructuring (phase 2).

    Same architecture as coagulation but tokens point to empty slots.
    Only modules that moved during coagulation are candidates to move.
    """

    TOKEN_PROCESS_DELAY = 1.0
    BOND_THRESHOLD = 1.05

    # See DecentralizedCoagulation.PIVOT_EXCLUSION_RADIUS for semantics.
    PIVOT_EXCLUSION_RADIUS: Optional[int] = None

    # See DecentralizedCoagulation.ALLOW_FAULT_AS_PIVOT_NEIGHBOR.
    ALLOW_FAULT_AS_PIVOT_NEIGHBOR: bool = True

    @staticmethod
    def _fault_body_index(sim, agents: Dict[str, ModuleAgent]) -> Optional[int]:
        have = {a.body_idx for a in agents.values()}
        rest = [i for i in range(sim.N) if i not in have]
        return rest[0] if len(rest) == 1 else None

    @staticmethod
    def _fault_body_indices(sim, agents: Dict[str, ModuleAgent]) -> Set[int]:
        have = {a.body_idx for a in agents.values()}
        return {i for i in range(sim.N) if i not in have}

    def _get_fault_idxs(self) -> Set[int]:
        """Return the set of all fault body indices."""
        idxs = getattr(self, "_fault_body_idxs", None)
        if idxs is not None:
            return idxs
        single = getattr(self, "_fault_body_idx", None)
        if single is not None:
            return {single}
        return self._fault_body_indices(self.sim, self.agents)

    def __init__(self, sim, module_ids: List[str],
                 body_indices: Dict[str, int],
                 coag_moved: Set[str],
                 pre_damage_neighbor_slots: Dict[str, List[np.ndarray]],
                 token_strategy: str = "furthest"):
        """
        Args:
            sim: BulletSimulator
            module_ids: active module IDs
            body_indices: {mid: body_idx}
            coag_moved: set of module IDs that moved during coagulation
            pre_damage_neighbor_slots: {mid: [relative_direction_vectors]}
                for modules that did NOT move (their slot knowledge is accurate)
            token_strategy: "furthest", "nearest", or "random"
        """
        self.sim = sim
        self.coag_moved = coag_moved
        self.pre_damage_neighbor_slots = pre_damage_neighbor_slots
        self.token_strategy = token_strategy

        self.agents: Dict[str, ModuleAgent] = {}
        for mid in module_ids:
            self.agents[mid] = ModuleAgent(
                module_id=mid,
                body_idx=body_indices[mid],
            )

        self._idx_to_mid: Dict[int, str] = {v: k for k, v in body_indices.items()}
        self._fault_body_idx: Optional[int] = self._fault_body_index(
            sim, self.agents)
        self._fault_body_idxs: Set[int] = self._fault_body_indices(
            sim, self.agents)
        self.total_moves = 0
        self.successful_moves = 0
        self.move_log: List[Dict] = []
        self._tick_count: int = 0

    def generate_initial_tokens(self):
        """Non-movers emit tokens for their empty neighbor slots."""
        pos = self.sim.get_positions()
        for mid, slots in self.pre_damage_neighbor_slots.items():
            if mid in self.coag_moved:
                continue  # only non-movers generate
            if mid not in self.agents:
                continue
            agent = self.agents[mid]
            R = self.sim.body_rotation_matrix(agent.body_idx)
            for slot_dir in slots:
                slot_pos = pos[agent.body_idx] + R @ slot_dir
                occupied = False
                for j in range(self.sim.N):
                    if np.linalg.norm(pos[j] - slot_pos) < 0.5:
                        occupied = True
                        break
                if not occupied:
                    agent.incoming_tokens.append(
                        Token(direction=slot_dir.copy(), source_id=mid))

    # The rest follows the same pattern as DecentralizedCoagulation
    # but with different candidate filtering and token selection

    def get_physical_neighbors(self, body_idx: int) -> List[int]:
        neighbors = []
        bond_matrix = self.sim.get_bond_matrix()
        for j in range(self.sim.N):
            if j != body_idx and bond_matrix[body_idx, j]:
                neighbors.append(j)
        return neighbors

    def _decision_graph_neighbors(self, body_idx: int) -> List[int]:
        bm = getattr(self, "_decision_bond_matrix", None)
        if bm is None:
            bm = self.sim.get_bond_matrix()
        return bond_graph_neighbors(bm, self.sim.N, body_idx)

    def _bfs_reachable(self, start_idx: int, max_depth: int,
                       excluded: Set[int]) -> Set[int]:
        visited = {start_idx}
        frontier = {start_idx}
        for _ in range(max_depth):
            next_frontier = set()
            for node in frontier:
                for nbr in self._decision_graph_neighbors(node):
                    if nbr not in excluded and nbr not in visited:
                        next_frontier.add(nbr)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break
        visited.discard(start_idx)
        return visited

    def is_movable(self, body_idx: int, safety_radius: int = 2) -> bool:
        """Two-graph movability test (see DecentralizedCoagulation.is_movable)."""
        fault_idxs = self._get_fault_idxs()

        all_neighbors = self._decision_graph_neighbors(body_idx)
        if len(all_neighbors) == 0:
            return False
        if len(all_neighbors) == 1:
            return True

        active_neighbors = [n for n in all_neighbors if n not in fault_idxs]
        if len(active_neighbors) == 0:
            return False

        for v in active_neighbors:
            excluded = {body_idx, v} | fault_idxs
            reachable_without_v = set()
            for w in active_neighbors:
                if w == v:
                    continue
                reachable_without_v |= self._bfs_reachable(
                    w, safety_radius - 1, excluded)
            v_neighbors = set(self._decision_graph_neighbors(v)) - fault_idxs
            if not (reachable_without_v & v_neighbors):
                return False
        return True

    def get_nearby_lattice_positions(self, body_idx: int) -> List[np.ndarray]:
        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        neighbors = self.get_physical_neighbors(body_idx)
        occupied = set()
        for i in range(self.sim.N):
            occupied.add(tuple(np.round(pos[i]).astype(int)))

        candidate_set = set()
        check_indices = set(neighbors)
        for n_idx in neighbors:
            for nn_idx in self.get_physical_neighbors(n_idx):
                check_indices.add(nn_idx)

        for idx in check_indices:
            nbr_pos = np.round(pos[idx]).astype(int)
            for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                candidate = tuple(nbr_pos + np.array(delta))
                if candidate not in occupied:
                    candidate_set.add(candidate)

        my_lattice = np.round(my_pos).astype(int)
        for delta in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            candidate = tuple(my_lattice + np.array(delta))
            if candidate not in occupied:
                candidate_set.add(candidate)

        return [np.array(c, dtype=float) for c in candidate_set]

    def _origin_world_for_pick_target(
            self, agent: ModuleAgent, pos: np.ndarray) -> Optional[np.ndarray]:
        if agent.token is None:
            return None
        return _token_origin_world_pos(
            agent.token, pos, self.agents,
            fault_id=None, fault_body_idx=None)

    def _on_no_pick_target(self, agent: ModuleAgent) -> None:
        """See ``DecentralizedCoagulation._on_no_pick_target``."""

    def _pick_target_occupancy_skip_body_indices(self) -> Set[int]:
        """See ``DecentralizedCoagulation._pick_target_occupancy_skip_body_indices``."""
        return set()

    def _ghost_goal_world_pos_for_axial_corner(
            self, pos: np.ndarray) -> Optional[np.ndarray]:
        """See ``DecentralizedCoagulation._ghost_goal_world_pos_for_axial_corner``."""
        return None

    def pick_target(
            self,
            agent: ModuleAgent,
            *,
            require_closer_to_origin: bool = False,
    ) -> Optional[PivotPick]:
        """Same corner / lateral enumeration and scoring as
        ``DecentralizedCoagulation.pick_target`` (see that docstring)."""
        if agent.token is None:
            return None
        if agent.module_id not in self.coag_moved:
            return None

        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        token_dir = agent.token.direction
        if np.linalg.norm(token_dir) < 1e-8:
            return None

        origin = self._origin_world_for_pick_target(agent, pos)
        if origin is None:
            return None

        neighbors = self._decision_graph_neighbors(agent.body_idx)
        if not neighbors:
            return None

        fault_idxs = self._get_fault_idxs()
        scored: List[Tuple[float, np.ndarray, int, str, Optional[int]]] = []
        nom = float(self.sim.NOMINAL_DIST)
        module_shape = getattr(self.sim, "_module_shape", "sphere")

        for axis_idx in neighbors:
            if (fault_idxs and axis_idx in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            R = self.sim.body_rotation_matrix(axis_idx)
            p_ref = pos[axis_idx]
            occupied = set()
            skip_occ = self._pick_target_occupancy_skip_body_indices()
            for i in range(self.sim.N):
                if i in skip_occ:
                    continue
                u = R.T @ (pos[i] - p_ref) / nom
                occupied.add(tuple(np.round(u).astype(int)))
            my_cell = tuple(np.round(R.T @ (my_pos - p_ref) / nom).astype(int))

            arm_axis_to_pivot = my_pos - pos[axis_idx]

            gpos_ax = self._ghost_goal_world_pos_for_axial_corner(pos)
            ghost_goal_cell: Optional[Tuple[int, int, int]] = None
            if gpos_ax is not None:
                ghost_goal_cell = tuple(
                    np.round(R.T @ (gpos_ax - p_ref) / nom).astype(int))

            corner_enabled = (module_shape != "cube")

            for delta in _LATTICE_DELTAS:
                if not corner_enabled:
                    break
                perp = _lattice_delta_perpendicular_to_arm(
                    delta, arm_axis_to_pivot, R)
                dw = nom * np.array(delta, dtype=float)
                target_world = p_ref + R @ dw
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                axial_into_ghost = (
                    ghost_goal_cell is not None
                    and target_cell == ghost_goal_cell)
                if not perp and not axial_into_ghost:
                    continue
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (axis_idx, target_cell) in agent.position_history:
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_tar_n = np.linalg.norm(r_target)
                if r_tar_n < 1e-12:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * r_tar_n + 1e-12)
                if axial_into_ghost:
                    if gpos_ax is None:
                        continue
                    d_tgt = float(np.linalg.norm(target_world - gpos_ax))
                    d_my = float(np.linalg.norm(my_pos - gpos_ax))
                    if d_tgt >= d_my - 1e-9:
                        continue
                else:
                    if abs(cos_angle) > _CORNER_PIVOT_MAX_ABS_COS:
                        continue
                score = -float(np.linalg.norm(target_world - origin))
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "corner", None))

            axis_neighbors = self._decision_graph_neighbors(axis_idx)
            for handoff_idx in axis_neighbors:
                if handoff_idx == agent.body_idx:
                    continue
                if (fault_idxs and handoff_idx in fault_idxs):
                    continue
                d_ah = pos[handoff_idx] - pos[axis_idx]
                if not _lateral_axis_handoff_step_cardinal_for_axis(
                        self.sim, axis_idx, d_ah):
                    continue
                target_world = pos[handoff_idx] + (my_pos - pos[axis_idx])
                target_cell = tuple(
                    np.round(R.T @ (target_world - p_ref) / nom).astype(int))
                if target_cell in occupied or target_cell == my_cell:
                    continue
                if (axis_idx, target_cell) in agent.position_history:
                    continue
                if np.linalg.norm(target_world - pos[axis_idx]) < 1.05:
                    continue
                r_vec = my_pos - pos[axis_idx]
                r_target = target_world - pos[axis_idx]
                r_vec_n = np.linalg.norm(r_vec)
                r_tar_n = np.linalg.norm(r_target)
                if r_vec_n < 1e-8 or r_tar_n < 1e-8:
                    continue
                cos_angle = np.dot(r_vec, r_target) / (r_vec_n * r_tar_n)
                if abs(cos_angle) > _LATERAL_PIVOT_MAX_ABS_COS:
                    continue
                if module_shape == "cube":
                    if not _cube_lateral_swept_clear(
                            occupied, my_cell, target_cell,
                            arm_axis_to_pivot, R, nom):
                        continue
                score = -float(np.linalg.norm(target_world - origin))
                t_local = R.T @ (target_world - p_ref)
                scored.append((score, t_local, axis_idx, "lateral", handoff_idx))

        if not scored:
            return None

        if require_closer_to_origin:
            current_d = float(np.linalg.norm(my_pos - origin))
            eps = 1e-9 * max(1.0, nom)
            scored = [s for s in scored if (-s[0]) < current_d - eps]
            if not scored:
                return None

        scored.sort(key=lambda x: (-x[0], x[2], x[4] if x[4] is not None else -1))
        best = scored[0]
        return best[1].copy(), best[2], best[3], best[4]

    def _propagate_moving_tokens(self):
        """Flood-fill moving tokens from PIVOTING/REVERSING agents."""
        if self.PIVOT_EXCLUSION_RADIUS is None:
            return
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        bm = self._decision_bond_matrix.copy()
        for a in self.agents.values():
            if a.state in _moving and a.pivot_axis_idx is not None:
                bm[a.body_idx, a.pivot_axis_idx] = True
                bm[a.pivot_axis_idx, a.body_idx] = True

        seeds: set = set()
        for a in self.agents.values():
            if a.state in _moving:
                a.moving_token_received_tick = self._tick_count
                seeds.add(a.body_idx)

        frontier = set(seeds)
        for _hop in range(self.PIVOT_EXCLUSION_RADIUS):
            nxt: set = set()
            for i in frontier:
                for j in range(self.sim.N):
                    if bm[i, j] and j not in seeds:
                        mid_j = self._idx_to_mid.get(j)
                        if mid_j and mid_j in self.agents:
                            self.agents[mid_j].moving_token_received_tick = self._tick_count
                        seeds.add(j)
                        nxt.add(j)
            frontier = nxt
            if not frontier:
                break

    def tick(self) -> bool:
        self._tick_count += 1
        self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

        self._propagate_moving_tokens()

        any_active = False
        agent_items = list(self.agents.items())
        random.shuffle(agent_items)

        for mid, agent in agent_items:
            if agent.is_faulty:
                continue

            if agent.state == ModuleState.IDLE:
                neighbors = self._decision_graph_neighbors(agent.body_idx)
                bm = self._decision_bond_matrix
                pos = self.sim.get_positions()
                triangle_found = False
                for ni_idx in range(len(neighbors)):
                    for nj_idx in range(ni_idx + 1, len(neighbors)):
                        a, b = neighbors[ni_idx], neighbors[nj_idx]
                        if bm[a, b]:
                            d_a = float(np.linalg.norm(pos[agent.body_idx] - pos[a]))
                            d_b = float(np.linalg.norm(pos[agent.body_idx] - pos[b]))
                            if d_a <= d_b:
                                close, far = a, b
                            else:
                                close, far = b, a
                            self.sim.remove_bond(agent.body_idx, far)
                            self._decision_bond_matrix = self.sim.get_bond_matrix().copy()
                            self._start_corrective_pivot(agent, close)
                            self._propagate_moving_tokens()
                            triangle_found = True
                            any_active = True
                            break
                    if triangle_found:
                        break
                if triangle_found:
                    continue

            if agent.state == ModuleState.PIVOTING:
                # Lateral handoff check
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        self.sim.stop_pivot(agent.body_idx)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} lateral handoff (phase 2): "
                            "bonded to {}, pivot switched",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx))

                if self.sim.is_pivot_complete(agent.body_idx):
                    collided = self.sim.pivot_collided(agent.body_idx)
                    timed_out = self.sim.pivot_timed_out(agent.body_idx)
                    needs_reversal = (
                        (collided or timed_out)
                        and agent.pre_pivot_pos_local is not None)

                    self.sim.stop_pivot(agent.body_idx)
                    if (agent.lattice_ref_body_idx is not None
                            and agent.target_pos_local is not None):
                        nom = float(self.sim.NOMINAL_DIST)
                        cell = tuple(
                            np.round(agent.target_pos_local / nom).astype(int))
                        agent.position_history.add(
                            (agent.lattice_ref_body_idx, cell))
                    self._reconnect_bonds(agent.body_idx)
                    self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

                    if needs_reversal:
                        agent.pending_retry = None
                        self._start_reversal(agent)
                        any_active = True
                        if collided:
                            logger.info(
                                "Module {} collision detected (phase 2) "
                                "— reversing", mid)
                        else:
                            logger.info(
                                "Module {} pivot timed out (phase 2) "
                                "— reversing", mid)
                    else:
                        agent.state = ModuleState.IDLE
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.token = None
                        self.successful_moves += 1
                        logger.info("Module {} pivot complete (phase 2)", mid)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.REVERSING:
                if (agent.pivot_type == "lateral"
                        and agent.handoff_idx is not None
                        and not agent.handoff_done):
                    pos = self.sim.get_positions()
                    d = np.linalg.norm(
                        pos[agent.body_idx] - pos[agent.handoff_idx])
                    if d < self.sim.handoff_contact_distance():
                        self.sim.create_bond(
                            agent.body_idx, agent.handoff_idx)
                        if agent.pivot_axis_idx is not None:
                            self.sim.remove_bond(
                                agent.body_idx, agent.pivot_axis_idx)
                        self.sim.stop_pivot(agent.body_idx)
                        new_ax = agent.handoff_idx
                        my_pos = pos[agent.body_idx]
                        new_ax_pos = pos[new_ax]
                        tgt_w = self.sim.target_world_from_local(
                            agent.lattice_ref_body_idx,
                            agent.target_pos_local)
                        r_vec = my_pos - new_ax_pos
                        rot_axis = self.sim.get_rotation_axis(
                            my_pos, new_ax_pos, tgt_w)
                        r_target = tgt_w - new_ax_pos
                        cos_a = np.clip(
                            np.dot(r_vec, r_target) / (
                                np.linalg.norm(r_vec) *
                                np.linalg.norm(r_target) + 1e-12),
                            -1, 1)
                        angle = np.arccos(cos_a)
                        kp, kd = self.sim.compute_pd_gains(
                            r_vec, duration=12.0)
                        self.sim.start_pivot(
                            agent.body_idx, new_ax, rot_axis, angle,
                            kp, kd, duration=12.0,
                            lattice_ref_body_idx=agent.lattice_ref_body_idx,
                            target_pos_local=agent.target_pos_local,
                            attract_body_idx=agent.attract_body_idx,
                            attract_connector=agent.attract_connector,
                            pivot_type="lateral",
                            repel_idx=-1)
                        agent.pivot_axis_idx = new_ax
                        agent.handoff_done = True
                        logger.debug(
                            "Module {} reversal handoff (phase 2): bonded to {}, "
                            "pivot switched from {}",
                            mid,
                            self._idx_to_mid.get(agent.handoff_idx,
                                                 agent.handoff_idx),
                            self._idx_to_mid.get(agent.pivot_axis_idx,
                                                 agent.pivot_axis_idx))

                if self.sim.is_pivot_complete(agent.body_idx):
                    rev_collided = self.sim.pivot_collided(agent.body_idx)
                    rev_timed_out = self.sim.pivot_timed_out(agent.body_idx)
                    self.sim.stop_pivot(agent.body_idx)
                    self._reconnect_bonds(agent.body_idx)
                    self._decision_bond_matrix = self.sim.get_bond_matrix().copy()

                    if rev_collided or rev_timed_out:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None
                        self._emergency_snap_to_lattice(agent)
                        any_active = True
                        logger.info(
                            "Module {} reversal failed (phase 2) — "
                            "snapping to lattice", mid)
                    else:
                        agent.target_pos = None
                        agent.target_pos_local = None
                        agent.lattice_ref_body_idx = None
                        agent.attract_body_idx = None
                        agent.attract_connector = None
                        agent.pivot_axis_idx = None
                        agent.pivot_type = None
                        agent.handoff_idx = None
                        agent.handoff_done = False
                        agent.pending_retry = None
                        agent.token = None
                        agent.reversal_handoff_idx = None

                        cooldown = random.uniform(0, 10)
                        agent.wait_until = self.sim.sim_time + cooldown
                        agent.state = ModuleState.WAITING
                        any_active = True
                        logger.info(
                            "Module {} reversal complete (phase 2) "
                            "— waiting {:.1f}s", mid, cooldown)
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.WAITING:
                if self.sim.sim_time >= agent.wait_until:
                    agent.state = ModuleState.IDLE
                else:
                    any_active = True
                continue

            if agent.state == ModuleState.PROCESSING:
                if self.sim.sim_time >= agent.process_ready_time:
                    self._forward_token(agent)
                    agent.state = ModuleState.IDLE
                    agent.token = None
                else:
                    any_active = True
                continue

            if agent.incoming_tokens:
                if self.token_strategy == "nearest":
                    best = min(agent.incoming_tokens,
                               key=lambda t: np.linalg.norm(t.direction))
                elif self.token_strategy == "random":
                    best = random.choice(agent.incoming_tokens)
                else:
                    best = max(agent.incoming_tokens,
                               key=lambda t: np.linalg.norm(t.direction))
                agent.token = best
                agent.incoming_tokens.clear()
                agent.token_hold_until = self.sim.sim_time + 1.0
                agent.state = ModuleState.HAS_TOKEN

            if agent.state == ModuleState.HAS_TOKEN:
                if self.sim.sim_time < agent.token_hold_until:
                    any_active = True
                    continue
                if not self.is_movable(agent.body_idx):
                    agent.state = ModuleState.PROCESSING
                    agent.process_ready_time = (self.sim.sim_time
                                                + self.TOKEN_PROCESS_DELAY)
                    any_active = True
                    continue

                # Pivot concurrency check.
                _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
                if self.PIVOT_EXCLUSION_RADIUS is None:
                    if any(a.state in _moving
                           for a in self.agents.values()):
                        any_active = True
                        continue
                else:
                    if (self._tick_count - agent.moving_token_received_tick
                            < self.PIVOT_EXCLUSION_RADIUS):
                        any_active = True
                        continue

                result = self.pick_target(agent, require_closer_to_origin=True)
                if result is not None:
                    target_pos_local, axis_idx, pivot_type, handoff_idx = result
                    self._start_pivot(
                        agent, target_pos_local, axis_idx,
                        pivot_type, handoff_idx)
                    self._propagate_moving_tokens()
                    any_active = True
                    continue

                self._on_no_pick_target(agent)
                agent.state = ModuleState.PROCESSING
                agent.process_ready_time = (self.sim.sim_time
                                            + self.TOKEN_PROCESS_DELAY)
                any_active = True

        for agent in self.agents.values():
            if (agent.state != ModuleState.IDLE or
                    agent.incoming_tokens or agent.token is not None):
                any_active = True
                break

        self._decision_bond_matrix = None
        return any_active

    def _start_pivot(
        self,
        agent: ModuleAgent,
        target_pos_local: np.ndarray,
        axis_idx: int,
        pivot_type: str,
        handoff_idx: Optional[int],
    ):
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        axis_pos = pos[axis_idx]
        lattice_ref_body_idx = axis_idx

        R_ax = self.sim.body_rotation_matrix(axis_idx)
        agent.pre_pivot_pos_local = (R_ax.T @ (my_pos - axis_pos)).copy()
        agent.pre_pivot_lattice_ref = axis_idx

        target_pos_local = np.asarray(target_pos_local, dtype=float).reshape(3)
        target_world = self.sim.target_world_from_local(
            lattice_ref_body_idx, target_pos_local)

        if pivot_type == "lateral" and handoff_idx is not None:
            attract_body = handoff_idx
            attract_conn = self.sim.lateral_handoff_attract_connector(
                axis_idx, agent.body_idx, handoff_idx)
        else:
            attract_body = axis_idx
            attract_conn = self.sim.nearest_connector(
                axis_idx, target_world - pos[axis_idx])

        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n in neighbors:
            if n != axis_idx:
                self.sim.remove_bond(agent.body_idx, n)

        if pivot_type == "corner":
            if not self.sim.USE_SPRING_BONDS:
                self.sim.remove_bond(agent.body_idx, axis_idx)

        r_vec = my_pos - axis_pos
        rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
        r_target = target_world - axis_pos
        cos_angle = np.clip(
            np.dot(r_vec, r_target) / (np.linalg.norm(r_vec) *
                                        np.linalg.norm(r_target) + 1e-12),
            -1, 1)
        angle = np.arccos(cos_angle)
        kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

        self.sim.start_pivot(
            agent.body_idx, axis_idx, rot_axis, angle,
            kp, kd, duration=12.0,
            lattice_ref_body_idx=lattice_ref_body_idx,
            target_pos_local=target_pos_local,
            attract_body_idx=attract_body,
            attract_connector=attract_conn,
            pivot_type=pivot_type)

        agent.state = ModuleState.PIVOTING
        agent.target_pos = target_world.copy()
        agent.target_pos_local = target_pos_local.copy()
        agent.lattice_ref_body_idx = lattice_ref_body_idx
        agent.attract_body_idx = attract_body
        agent.attract_connector = attract_conn
        agent.pivot_axis_idx = axis_idx
        agent.pivot_type = pivot_type
        agent.handoff_idx = handoff_idx
        agent.handoff_done = False
        self.total_moves += 1

        self.move_log.append({
            "module": agent.module_id,
            "from": my_pos.tolist(),
            "to": target_world.tolist(),
            "axis": self._idx_to_mid.get(axis_idx, f"idx_{axis_idx}"),
            "sim_time": self.sim.sim_time,
            "pivot_type": pivot_type,
        })
        logger.info("Module {} starting {} pivot to {} (phase 2{})",
                     agent.module_id, pivot_type, target_world,
                     f", handoff={self._idx_to_mid.get(handoff_idx, handoff_idx)}"
                     if handoff_idx is not None else "")

    def _start_reversal(self, agent: ModuleAgent):
        """Pivot the module back to its pre-pivot lattice position (phase 2).

        See DecentralizedCoagulation._start_reversal for lateral reversal logic.
        """
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]

        neighbors = self.get_physical_neighbors(agent.body_idx)
        if not neighbors:
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            self.sim.create_bond(agent.body_idx, nearest)
            neighbors = [nearest]

        ref = agent.pre_pivot_lattice_ref
        target_local = agent.pre_pivot_pos_local
        if target_local is None:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None
            return

        original_axis = ref if ref is not None else None
        is_lateral_reversal = (
            agent.pivot_type == "lateral"
            and agent.handoff_done
            and original_axis is not None
            and original_axis != agent.pivot_axis_idx
        )

        best_ax = min(neighbors,
                      key=lambda n: np.linalg.norm(pos[n] - my_pos))
        if ref is None:
            ref = best_ax
        target_world = self.sim.target_world_from_local(ref, target_local)
        axis_pos = pos[best_ax]

        for n in neighbors:
            if n != best_ax:
                self.sim.remove_bond(agent.body_idx, n)

        if is_lateral_reversal:
            attract_conn = self.sim.lateral_handoff_attract_connector(
                best_ax, agent.body_idx, original_axis)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=original_axis,
                attract_connector=attract_conn,
                pivot_type="lateral",
                repel_idx=-1)

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = original_axis
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "lateral"
            agent.handoff_idx = original_axis
            agent.handoff_done = False
            agent.reversal_handoff_idx = original_axis
        else:
            attract_conn = self.sim.nearest_connector(
                best_ax, target_world - axis_pos)

            r_vec = my_pos - axis_pos
            rot_axis = self.sim.get_rotation_axis(my_pos, axis_pos, target_world)
            r_target = target_world - axis_pos
            cos_a = np.clip(
                np.dot(r_vec, r_target) / (
                    np.linalg.norm(r_vec) * np.linalg.norm(r_target) + 1e-12),
                -1, 1)
            angle = np.arccos(cos_a)
            kp, kd = self.sim.compute_pd_gains(r_vec, duration=12.0)

            self.sim.start_pivot(
                agent.body_idx, best_ax, rot_axis, angle, kp, kd,
                duration=12.0,
                lattice_ref_body_idx=ref,
                target_pos_local=target_local,
                attract_body_idx=best_ax,
                attract_connector=attract_conn,
                pivot_type="corner")

            agent.state = ModuleState.REVERSING
            agent.target_pos = target_world.copy()
            agent.target_pos_local = target_local.copy()
            agent.lattice_ref_body_idx = ref
            agent.attract_body_idx = best_ax
            agent.attract_connector = attract_conn
            agent.pivot_axis_idx = best_ax
            agent.pivot_type = "corner"
            agent.handoff_idx = None
            agent.handoff_done = False
            agent.reversal_handoff_idx = None

    def _emergency_snap_to_lattice(self, agent: ModuleAgent):
        """Ensure module is bonded and pivot to the nearest empty lattice site.

        Called when a reversal pivot fails (timeout/collision) during phase 2.
        Guarantees the module keeps at least one bond and attempts a corner
        pivot to the closest unoccupied lattice cell.
        """
        neighbors = self.get_physical_neighbors(agent.body_idx)
        pos = self.sim.get_positions()

        if not neighbors:
            nearest = min(
                (j for j in range(self.sim.N) if j != agent.body_idx),
                key=lambda j: np.linalg.norm(
                    pos[j] - pos[agent.body_idx]))
            self.sim.create_bond(agent.body_idx, nearest)
            neighbors = [nearest]

        axis_idx = min(neighbors,
                       key=lambda n: np.linalg.norm(
                           pos[n] - pos[agent.body_idx]))

        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)
        my_pos = pos[agent.body_idx]

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is not None:
            self._start_pivot(agent, best_target_local, axis_idx, "corner",
                              None)
            logger.info("Module {} emergency snap pivot (phase 2) around {}",
                        agent.module_id,
                        self._idx_to_mid.get(axis_idx, axis_idx))
        else:
            agent.state = ModuleState.IDLE
            agent.token = None
            agent.pending_retry = None

    def _start_corrective_pivot(self, agent: ModuleAgent, axis_idx: int):
        """Corner pivot to nearest empty lattice site on *axis_idx* to fix a triangular bond."""
        pos = self.sim.get_positions()
        my_pos = pos[agent.body_idx]
        R = self.sim.body_rotation_matrix(axis_idx)
        p_ref = pos[axis_idx]
        nom = float(self.sim.NOMINAL_DIST)

        occupied: set = set()
        for i in range(self.sim.N):
            u = R.T @ (pos[i] - p_ref) / nom
            occupied.add(tuple(np.round(u).astype(int)))

        arm = my_pos - p_ref
        best_target_local = None
        best_dist = float("inf")
        for delta in _LATTICE_DELTAS:
            dw = nom * np.array(delta, dtype=float)
            target_world = p_ref + R @ dw
            target_cell = tuple(np.round(dw / nom).astype(int))
            if target_cell in occupied:
                continue
            if not _lattice_delta_perpendicular_to_arm(delta, arm, R):
                continue
            d = float(np.linalg.norm(target_world - my_pos))
            if d < best_dist:
                best_dist = d
                best_target_local = R.T @ (target_world - p_ref)

        if best_target_local is None:
            return

        self._start_pivot(agent, best_target_local, axis_idx, "corner", None)
        logger.info("Module {} corrective pivot (triangle fix, phase 2) around {}",
                    agent.module_id, self._idx_to_mid.get(axis_idx, axis_idx))

    def _forward_token(self, agent):
        if agent.token is None:
            return
        pos = self.sim.get_positions()
        neighbors = self.get_physical_neighbors(agent.body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid is None or n_mid not in self.agents:
                continue
            n_agent = self.agents[n_mid]
            if n_agent.state in (ModuleState.PIVOTING, ModuleState.REVERSING):
                continue
            my_pos = pos[agent.body_idx]
            nbr_pos = pos[n_idx]
            R_me = self.sim.body_rotation_matrix(agent.body_idx)
            R_nbr = self.sim.body_rotation_matrix(n_idx)
            world_dir = R_me @ agent.token.direction + (my_pos - nbr_pos)
            propagated_dir = R_nbr.T @ world_dir
            n_agent.incoming_tokens.append(
                Token(direction=propagated_dir, source_id=agent.module_id))

    def _reconnect_bonds(self, body_idx):
        """Reconnect bonds dictated by the policy after a pivot.

        Only creates bonds to the pivot axis module and to modules that
        occupy lattice-adjacent positions of the target, provided they
        are within the distance threshold.  No blind proximity scan.
        """
        agent = None
        for a in self.agents.values():
            if a.body_idx == body_idx:
                agent = a
                break
        if agent is None:
            return

        pos = self.sim.get_positions()
        my_pos = pos[body_idx]
        if (agent.lattice_ref_body_idx is not None
                and agent.target_pos_local is not None):
            target = self.sim.target_world_from_local(
                agent.lattice_ref_body_idx, agent.target_pos_local)
        else:
            target = agent.target_pos if agent.target_pos is not None else my_pos

        expected_neighbors: Set[int] = set()
        if agent.pivot_axis_idx is not None:
            expected_neighbors.add(agent.pivot_axis_idx)

        if agent.lattice_ref_body_idx is not None:
            ref = agent.lattice_ref_body_idx
            R = self.sim.body_rotation_matrix(ref)
            p_ref = pos[ref]
            nom = float(self.sim.NOMINAL_DIST)
            goal_cell = np.round(
                agent.target_pos_local / nom).astype(int)  # type: ignore[union-attr]
            for delta in _LATTICE_DELTAS:
                nbr_cell = (int(goal_cell[0] + delta[0]),
                            int(goal_cell[1] + delta[1]),
                            int(goal_cell[2] + delta[2]))
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    j_cell = tuple(np.round(
                        R.T @ (pos[j] - p_ref) / nom).astype(int))
                    if j_cell == nbr_cell:
                        expected_neighbors.add(j)
        else:
            target_key = tuple(np.round(target).astype(int))
            for delta in _LATTICE_DELTAS:
                nbr_key = (target_key[0] + delta[0],
                           target_key[1] + delta[1],
                           target_key[2] + delta[2])
                for j in range(self.sim.N):
                    if j == body_idx:
                        continue
                    if tuple(np.round(pos[j]).astype(int)) == nbr_key:
                        expected_neighbors.add(j)

        for j in expected_neighbors:
            dist = np.linalg.norm(my_pos - pos[j])
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

        fault_idxs = self._get_fault_idxs()
        for j in range(self.sim.N):
            if j == body_idx:
                continue
            if (fault_idxs and j in fault_idxs
                    and not self.ALLOW_FAULT_AS_PIVOT_NEIGHBOR):
                continue
            dist = float(np.linalg.norm(my_pos - pos[j]))
            if dist < self.BOND_THRESHOLD:
                self.sim.create_bond(body_idx, j)

    def _has_pivoting_neighbor_of_neighbor(self, body_idx):
        _moving = (ModuleState.PIVOTING, ModuleState.REVERSING)
        neighbors = self.get_physical_neighbors(body_idx)
        for n_idx in neighbors:
            n_mid = self._idx_to_mid.get(n_idx)
            if n_mid and n_mid in self.agents:
                if self.agents[n_mid].state in _moving:
                    return True
            for nn_idx in self.get_physical_neighbors(n_idx):
                nn_mid = self._idx_to_mid.get(nn_idx)
                if nn_mid and nn_mid in self.agents:
                    if self.agents[nn_mid].state in _moving:
                        return True
        return False


class DisplacementRestructuring(DecentralizedRestructuring):
    """Displacement-guided restructuring (phase 2 alternative).

    Instead of propagating rendezvous tokens from non-movers' empty slots,
    each displaced module computes its displacement from its pre-damage
    position and uses that as a synthetic token direction.  No inter-module
    token propagation is needed -- each agent independently knows where it
    should return to.

    Modules are driven by largest-displacement-first priority (the greedy
    ordering from the graph-based ``restructuring_displacement``), adapted
    to the async PyBullet tick loop.
    """

    DISPLACEMENT_THRESHOLD = 0.5

    def __init__(self, sim, module_ids: List[str],
                 body_indices: Dict[str, int],
                 coag_moved: Set[str],
                 original_positions: Dict[str, np.ndarray],
                 pre_damage_neighbor_slots: Optional[Dict[str, List[np.ndarray]]] = None):
        super().__init__(
            sim=sim,
            module_ids=module_ids,
            body_indices=body_indices,
            coag_moved=coag_moved,
            pre_damage_neighbor_slots=pre_damage_neighbor_slots or {},
            token_strategy="furthest",
        )
        self.original_positions = original_positions

    def generate_initial_tokens(self):
        """Inject displacement-based tokens for all displaced coag-movers."""
        self._inject_displacement_tokens()

    def _inject_displacement_tokens(self):
        """For each displaced coag-mover, set a synthetic token pointing home."""
        pos = self.sim.get_positions()
        for mid in self.coag_moved:
            if mid not in self.agents:
                continue
            if mid not in self.original_positions:
                continue
            agent = self.agents[mid]
            if agent.state != ModuleState.IDLE:
                continue
            disp = self.original_positions[mid] - pos[agent.body_idx]
            dist = float(np.linalg.norm(disp))
            if dist < self.DISPLACEMENT_THRESHOLD:
                continue
            R = self.sim.body_rotation_matrix(agent.body_idx)
            direction = R.T @ disp
            agent.incoming_tokens.append(
                Token(direction=direction.copy(), source_id=mid))

    def _origin_world_for_pick_target(
            self, agent: ModuleAgent, pos: np.ndarray) -> Optional[np.ndarray]:
        """Scoring origin = the module's original (pre-damage) position."""
        if agent.module_id in self.original_positions:
            return self.original_positions[agent.module_id].copy()
        if agent.token is None:
            return None
        R = self.sim.body_rotation_matrix(agent.body_idx)
        return pos[agent.body_idx] + R @ agent.token.direction
