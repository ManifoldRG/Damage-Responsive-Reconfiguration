import numpy as np
from typing import Dict, List, Set, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import uuid
from loguru import logger

class Direction(Enum):
    """Cubic lattice directions for modular connections."""
    POS_X = (1, 0, 0)
    NEG_X = (-1, 0, 0)
    POS_Y = (0, 1, 0)
    NEG_Y = (0, -1, 0)
    POS_Z = (0, 0, 1)
    NEG_Z = (0, 0, -1)
    
    @property
    def vector(self) -> np.ndarray:
        return np.array(self.value)

@dataclass
class ConnectionPort:
    """Represents a connection port on a module."""
    direction: Direction
    is_connected: bool = False
    connected_module: Optional[str] = None  # Module ID
    is_dockable: bool = True
    
    def connect_to(self, module_id: str):
        """Connect this port to another module."""
        self.is_connected = True
        self.connected_module = module_id
    
    def disconnect(self):
        """Disconnect this port."""
        self.is_connected = False
        self.connected_module = None

@dataclass
class Message:
    """Message passed between modules."""
    sender_id: str
    receiver_id: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=lambda: 0.0)

class Module:
    """
    Represents a single module in the modular system.
    
    Each module maintains its own state, connections, and can process messages.
    """
    
    def __init__(self, module_id: str = None, position: np.ndarray = None):
        self.id = module_id or str(uuid.uuid4())[:8]
        self.position = position if position is not None else np.zeros(3)
        logger.info(f"Module {self.id} initialized at position {self.position}")
        
        # Activity state
        self.is_active = True
        self.is_pivotable = True
        
        # Attitude
        self.orientation = np.array([0.0, 0.0, 0.0])
        
        # Connection ports for all 6 directions
        self.ports: Dict[Direction, ConnectionPort] = {
            direction: ConnectionPort(direction) for direction in Direction
        }
        
        # Message handling
        self.message_queue: List[Message] = []
        self.message_handlers: Dict[str, callable] = {}
        
        # Neighbor tracking
        self.neighbors_1hop: Set[str] = set()
        self.neighbors_2hop: Set[str] = set()
        
        # Module-specific data
        self.module_data: Dict[str, Any] = {}
        
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Setup default message handlers."""
        self.message_handlers.update({
            'neighbor_discovery': self._handle_neighbor_discovery,
            'status_request': self._handle_status_request,
            'status_response': self._handle_status_response,
            'damage_notification': self._handle_damage_notification,
        })
    
    def get_connected_directions(self) -> List[Direction]:
        """Get all directions where this module has connections."""
        return [port.direction for port in self.ports.values() if port.is_connected]
    
    def get_available_ports(self) -> List[Direction]:
        """Get all available (unconnected but dockable) ports."""
        return [port.direction for port in self.ports.values() 
                if not port.is_connected and port.is_dockable]
    
    def connect_to_module(self, direction: Direction, other_module_id: str) -> bool:
        """
        Connect this module to another in the specified direction.
        
        Args:
            direction: Direction to connect in
            other_module_id: ID of the module to connect to
            
        Returns:
            bool: True if connection successful
        """
        if direction not in self.ports:
            return False
            
        port = self.ports[direction]
        if port.is_connected or not port.is_dockable:
            return False
        
        port.connect_to(other_module_id)
        self.neighbors_1hop.add(other_module_id)
        logger.info(f"Module {self.id} connected to {other_module_id} in direction {direction.name}")
        return True
    
    def disconnect_from_direction(self, direction: Direction) -> bool:
        """Disconnect from the specified direction."""
        if direction not in self.ports:
            return False
            
        port = self.ports[direction]
        if not port.is_connected:
            return False
        
        disconnected_module = port.connected_module
        port.disconnect()
        
        if disconnected_module in self.neighbors_1hop:
            self.neighbors_1hop.remove(disconnected_module)
        
        logger.info(f"Module {self.id} disconnected from {disconnected_module} in direction {direction.name}")
        return True
    
    def set_damage(self, damaged: bool = True):
        """Set the damage state of this module."""
        self.is_active = not damaged
        self.is_pivotable = not damaged
        
        if damaged:
            logger.warning(f"Module {self.id} marked as damaged")
            # When damaged, module becomes unpivotable but connections remain
            self.send_message_to_neighbors('damage_notification', {
                'module_id': self.id,
                'status': 'damaged'
            })
        else:
            logger.info(f"Module {self.id} damage state cleared")
    
    def can_pivot(self) -> bool:
        """Check if this module can perform a pivot operation."""
        if not self.is_active or not self.is_pivotable:
            return False
        
        # All neighbors must be active for pivot capability (Simplified for now)
        return True
    
    def can_be_pivot_axis(self) -> bool:
        """Check if this module can serve as a pivot axis."""
        return self.is_active
    
    def send_message(self, message: Message):
        """Send a message to another module (handled by the system)."""
        # This will be handled by the ModularSystem class
        pass
    
    def send_message_to_neighbors(self, message_type: str, payload: Dict[str, Any]):
        """Send a message to all 1-hop neighbors."""
        logger.debug(f"Module {self.id} sending {message_type} to {len(self.neighbors_1hop)} neighbors")
        for neighbor_id in self.neighbors_1hop:
            message = Message(
                sender_id=self.id,
                receiver_id=neighbor_id,
                message_type=message_type,
                payload=payload
            )
            self.send_message(message)
    
    def receive_message(self, message: Message):
        """Receive and queue a message."""
        logger.debug(f"Module {self.id} received {message.message_type} from {message.sender_id}")
        self.message_queue.append(message)
    
    def process_messages(self):
        """Process all queued messages."""
        processed_count = 0
        while self.message_queue:
            message = self.message_queue.pop(0)
            handler = self.message_handlers.get(message.message_type)
            if handler:
                handler(message)
                processed_count += 1
            else:
                logger.warning(f"Module {self.id} has no handler for message type: {message.message_type}")
        
        if processed_count > 0:
            logger.debug(f"Module {self.id} processed {processed_count} messages")
    
    def _handle_neighbor_discovery(self, message: Message):
        """Handle neighbor discovery messages."""
        sender_id = message.sender_id
        payload = message.payload
        
        # Update neighbor information
        if payload.get('hop_count', 1) == 1:
            self.neighbors_1hop.add(sender_id)
        elif payload.get('hop_count', 1) == 2:
            self.neighbors_2hop.add(sender_id)
    
    def _handle_status_request(self, message: Message):
        """Handle status request messages."""
        response = Message(
            sender_id=self.id,
            receiver_id=message.sender_id,
            message_type='status_response',
            payload={
                'is_active': self.is_active,
                'is_pivotable': self.is_pivotable,
                'position': self.position.tolist(),
                'connected_directions': [d.name for d in self.get_connected_directions()],
                'available_ports': [d.name for d in self.get_available_ports()]
            }
        )
        self.send_message(response)
    
    def _handle_status_response(self, message: Message):
        """Handle status response messages."""
        sender_id = message.sender_id
        payload = message.payload
        logger.debug(f"Module {self.id} received status from {sender_id}: active={payload.get('is_active')}")
        
        # Store neighbor status information
        if 'neighbor_status' not in self.module_data:
            self.module_data['neighbor_status'] = {}
        
        self.module_data['neighbor_status'][sender_id] = {
            'is_active': payload.get('is_active'),
            'is_pivotable': payload.get('is_pivotable'),
            'position': payload.get('position'),
            'connected_directions': payload.get('connected_directions'),
            'available_ports': payload.get('available_ports')
        }
    
    def _handle_damage_notification(self, message: Message):
        """Handle damage notification from neighbors."""
        damaged_module = message.payload.get('module_id')
        if damaged_module in self.neighbors_1hop:
            logger.warning(f"Module {self.id} detected damage in neighbor {damaged_module}")
            # React to neighbor damage - could trigger reconfiguration
            self.module_data['neighbor_damage_detected'] = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of this module."""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'is_active': self.is_active,
            'is_pivotable': self.is_pivotable,
            'orientation': self.orientation.tolist(),
            'connected_directions': [d.name for d in self.get_connected_directions()],
            'available_ports': [d.name for d in self.get_available_ports()],
            'neighbors_1hop': list(self.neighbors_1hop),
            'neighbors_2hop': list(self.neighbors_2hop),
            'message_queue_size': len(self.message_queue)
        }
    
    def __repr__(self):
        return f"Module(id={self.id}, active={self.is_active}, pos={self.position})"

# Complex example demonstrating a larger modular system with damage scenarios
def complex_example():
    """More complex example demonstrating a larger modular system with damage scenarios."""
    logger.info("=" * 60)
    logger.info("COMPLEX EXAMPLE: 3x3 Grid with Damage Scenarios")
    logger.info("=" * 60)
    
    # Create a 3x3 grid of modules
    modules = {}
    for x in range(3):
        for y in range(3):
            module_id = f"M{x}{y}"
            position = np.array([x, y, 0])
            modules[module_id] = Module(module_id, position)
    
    logger.info(f"Created {len(modules)} modules in 3x3 grid")
    
    # Connect adjacent modules in grid pattern
    connections_made = 0
    for x in range(3):
        for y in range(3):
            current_id = f"M{x}{y}"
            current_module = modules[current_id]
            
            # Connect to right neighbor
            if x < 2:
                neighbor_id = f"M{x+1}{y}"
                if current_module.connect_to_module(Direction.POS_X, neighbor_id):
                    modules[neighbor_id].connect_to_module(Direction.NEG_X, current_id)
                    connections_made += 1
            
            # Connect to top neighbor
            if y < 2:
                neighbor_id = f"M{x}{y+1}"
                if current_module.connect_to_module(Direction.POS_Y, neighbor_id):
                    modules[neighbor_id].connect_to_module(Direction.NEG_Y, current_id)
                    connections_made += 1
    
    logger.info(f"Established {connections_made} bidirectional connections")
    
    # Test neighbor discovery messages
    logger.info("\n--- Testing Neighbor Discovery ---")
    for module_id, module in modules.items():
        neighbor_count = len(module.neighbors_1hop)
        logger.info(f"Module {module_id}: {neighbor_count} neighbors - {list(module.neighbors_1hop)}")
    
    # Test status request broadcast
    logger.info("\n--- Testing Status Request Broadcast ---")
    center_module = modules["M11"]  # Center module
    center_module.send_message_to_neighbors("status_request", {"requester": "M11"})
    
    # Simulate message delivery and processing
    for neighbor_id in center_module.neighbors_1hop:
        neighbor_module = modules[neighbor_id]
        # Create response message
        status_message = Message(
            sender_id=neighbor_id,
            receiver_id="M11",
            message_type="status_response",
            payload=neighbor_module.get_status()
        )
        center_module.receive_message(status_message)
    
    center_module.process_messages()
    
    # Introduce multiple damage scenarios
    logger.info("\n--- Damage Scenario 1: Corner Module Damage ---")
    corner_module = modules["M00"]
    corner_module.set_damage(True)
    
    # Process damage notifications
    for neighbor_id in corner_module.neighbors_1hop:
        neighbor_module = modules[neighbor_id]
        damage_msg = Message(
            sender_id="M00",
            receiver_id=neighbor_id,
            message_type="damage_notification",
            payload={"module_id": "M00", "status": "damaged"}
        )
        neighbor_module.receive_message(damage_msg)
        neighbor_module.process_messages()
    
    logger.info("\n--- Damage Scenario 2: Critical Hub Damage ---")
    # Damage the center module (most connected)
    center_module.set_damage(True)
    
    # Notify all neighbors
    for neighbor_id in list(center_module.neighbors_1hop):  # Copy list to avoid modification during iteration
        neighbor_module = modules[neighbor_id]
        damage_msg = Message(
            sender_id="M11",
            receiver_id=neighbor_id,
            message_type="damage_notification",
            payload={"module_id": "M11", "status": "damaged"}
        )
        neighbor_module.receive_message(damage_msg)
        neighbor_module.process_messages()
    
    # Test pivot capabilities after damage
    logger.info("\n--- Testing Pivot Capabilities After Damage ---")
    active_modules = [m for m in modules.values() if m.is_active]
    pivotable_modules = [m for m in active_modules if m.can_pivot()]
    pivot_axis_modules = [m for m in active_modules if m.can_be_pivot_axis()]
    
    logger.info(f"Active modules: {len(active_modules)}/{len(modules)}")
    logger.info(f"Pivotable modules: {len(pivotable_modules)}")
    logger.info(f"Pivot axis capable: {len(pivot_axis_modules)}")
    
    # Test message queue handling under load
    logger.info("\n--- Stress Testing Message Queues ---")
    test_module = modules["M22"]  # Corner module still active
    
    # Send multiple message types
    message_types = ["neighbor_discovery", "status_request", "damage_notification", "custom_test"]
    for i in range(10):
        for msg_type in message_types:
            test_msg = Message(
                sender_id=f"EXTERNAL_{i}",
                receiver_id="M22",
                message_type=msg_type,
                payload={"test_data": i, "batch": "stress_test"}
            )
            test_module.receive_message(test_msg)
    
    logger.info(f"Queued {len(test_module.message_queue)} messages for processing")
    test_module.process_messages()
    
    # Final system state report
    logger.info("\n--- Final System State Report ---")
    active_count = sum(1 for m in modules.values() if m.is_active)
    damaged_count = len(modules) - active_count
    total_connections = sum(len(m.neighbors_1hop) for m in modules.values()) // 2  # Bidirectional
    
    logger.info(f"System Health: {active_count}/{len(modules)} modules active ({damaged_count} damaged)")
    logger.info(f"Network Connectivity: {total_connections} connections maintained")
    
    # Test disconnection due to damage
    logger.info("\n--- Testing Disconnection Scenarios ---")
    if modules["M01"].is_active and modules["M02"].is_active:
        # Test manual disconnection
        before_neighbors = len(modules["M01"].neighbors_1hop)
        modules["M01"].disconnect_from_direction(Direction.POS_X)
        after_neighbors = len(modules["M01"].neighbors_1hop)
        logger.info(f"M01 neighbors: {before_neighbors} -> {after_neighbors} after disconnection")
    
    logger.info("Complex example completed successfully")


if __name__ == "__main__":
    # Run simple example first
    logger.info("Starting modular system example")
    
    # Create a few modules
    module_a = Module("A", np.array([0, 0, 0]))
    module_b = Module("B", np.array([1, 0, 0]))
    module_c = Module("C", np.array([0, 1, 0]))
    
    # Connect modules
    module_a.connect_to_module(Direction.POS_X, "B")
    module_b.connect_to_module(Direction.NEG_X, "A")
    
    module_a.connect_to_module(Direction.POS_Y, "C")
    module_c.connect_to_module(Direction.NEG_Y, "A")
    
    # Test status
    logger.info("Module A status:")
    logger.info(module_a.get_status())
    
    # Test damage
    module_b.set_damage(True)
    logger.info(f"Module B after damage: active={module_b.is_active}")
    
    # Test message passing
    message = Message("A", "B", "status_request", {})
    module_b.receive_message(message)
    module_b.process_messages()
    
    logger.info("Simple example completed")
    
    # Run complex example
    complex_example()