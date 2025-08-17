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

# Example usage and testing
if __name__ == "__main__":
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
    
    logger.info("Example completed")