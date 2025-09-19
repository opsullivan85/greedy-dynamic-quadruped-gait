"""Holds constants for footstep scanner configuration

Does not require starting IsaacGym, so can be imported in other places
"""

grid_resolution = 0.075
"""Distance between rays in the grid and overall grid size"""
grid_size = (5, 5)
"""Odd numbers will be centered on the _stable_footstep_offset"""