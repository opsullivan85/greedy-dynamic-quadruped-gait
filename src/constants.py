import numpy as np
from src import get_logger
from dataclasses import dataclass, field

logger = get_logger()


@dataclass(frozen=True)
class _ContactNet:
    grid_resolution: float = 0.075
    """Grid resolution used in contact-net training data generation"""
    grid_size: np.ndarray = field(default_factory=lambda: np.asarray((5, 5), dtype=int))
    """Grid size used in contact-net training data generation"""

    def __post_init__(self):
        self.grid_size.setflags(write=False)


contact_net = _ContactNet()
"""ContactNet constants"""


_footstep_scanner_scale: int = 5


@dataclass(frozen=True)
class _FootstepScanner:
    grid_resolution: float = contact_net.grid_resolution / _footstep_scanner_scale
    """Grid resolution used in footstep scanner observations"""
    grid_size: np.ndarray = field(
        default_factory=lambda: contact_net.grid_size * _footstep_scanner_scale
    )
    """Grid size used in footstep scanner observations"""
    cspace_dialation: int = 2
    """Number of times to apply max-pooling to the height scan to simulate c-space dialation"""
    upscale_costmap_noise: float = 0.03
    """Amount of noise (+/-) to add to the costmap during upscale"""

    def __post_init__(self):
        self.grid_size.setflags(write=False)


footstep_scanner = _FootstepScanner()
"""Footstep scanner constants"""


##### Checks

assert contact_net.grid_size.shape == (2,), "ContactNet grid size must be 2D"
assert footstep_scanner.grid_size.shape == (2,), "Footstep scanner grid size must be 2D"

assert np.all(
    contact_net.grid_resolution * contact_net.grid_size
    == footstep_scanner.grid_resolution * footstep_scanner.grid_size
), "ContactNet grid and footstep scanner grid must cover the same area"

if np.any(np.mod(contact_net.grid_size, footstep_scanner.grid_size) != 0):
    logger.warning(
        "ContactNet and Footstep scanner grid sizes should probably be multiples of each other"
    )
