import numpy as np
from typing import Tuple

class UnitDualQuaternion:
    """
    Simple Unit Dual Quaternion for pure translations in 3D lattice.
    
    For pure translation: q̂ = 1 + (ε/2) * (0, t_x, t_y, t_z)
    where ε² = 0 and t is the translation vector.
    """
    
    def __init__(self, translation: np.ndarray):
        """Initialize with translation vector."""
        self.translation = np.array(translation, dtype=float)
        
    @property
    def inverse(self) -> 'UnitDualQuaternion':
        """Return the inverse: q̂⁻¹ = 1 - (ε/2) * (0, t)"""
        return UnitDualQuaternion(-self.translation)
    
    def __mul__(self, other: 'UnitDualQuaternion') -> 'UnitDualQuaternion':
        """Compose two dual quaternions (translation addition for pure translations)."""
        return UnitDualQuaternion(self.translation + other.translation)
    
    def to_translation(self) -> np.ndarray:
        """Extract translation vector."""
        return self.translation.copy()
    
    def __repr__(self):
        return f"UDQ(t={self.translation})"

def create_edge_gain(direction: Tuple[int, int, int]) -> UnitDualQuaternion:
    """Create edge gain for lattice direction."""
    return UnitDualQuaternion(np.array(direction, dtype=float))

# Lattice directions for cubic grid
LATTICE_DIRECTIONS = {
    'POS_X': (1, 0, 0),
    'NEG_X': (-1, 0, 0),
    'POS_Y': (0, 1, 0),
    'NEG_Y': (0, -1, 0),
    'POS_Z': (0, 0, 1),
    'NEG_Z': (0, 0, -1)
}