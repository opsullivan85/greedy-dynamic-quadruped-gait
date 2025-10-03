"""Holds constants for footstep scanner configuration

Does not require starting IsaacGym, so can be imported in other places
"""
import torch
from src.constants import _contact_net_grid_resolution, _contact_net_grid_size

_grid_resolution: float = _contact_net_grid_resolution
"""Distance between rays in the grid and overall grid size"""
# TODO: decouple this from the contactnet grid size
_grid_size = _contact_net_grid_size
"""Odd numbers will be centered on the _stable_footstep_offset"""

def idx_to_xy(indices: torch.Tensor) -> torch.Tensor:
    """Convert from grid indices to (x, y) coordinates in meters.

    Args:
        indices (torch.Tensor): Indices of shape (..., 2/3) where the last
            dimension is (idx, idy) or (foot, idx, idy).

    Returns:
        torch.Tensor: (..., 2/3) tensor of (foot?, x, y) coordinates in meters.
    """
    # assert shape is correct
    assert indices.shape[-1] in [2, 3]
    half_size_x = (_grid_size[0] - 1) * _grid_resolution / 2
    half_size_y = (_grid_size[1] - 1) * _grid_resolution / 2
    x_locations = torch.linspace(-half_size_x, half_size_x, _grid_size[0], device=indices.device)
    y_locations = torch.linspace(-half_size_y, half_size_y, _grid_size[1], device=indices.device)

    x = x_locations[indices[..., -2]]
    y = y_locations[indices[..., -1]]

    if indices.shape[-1] == 2:
        return torch.stack((x, y), dim=-1)
    else:
        foot = indices[..., 0]
        return torch.stack((foot, x, y), dim=-1)
