from typing import Callable

from isaaclab.sensors import RayCasterCfg, patterns  # type: ignore


_hip_names = ["FR_hip", "FL_hip", "RL_hip", "RR_hip"]

# Distance between rays in the grid and overall grid size
grid_resolution = 0.075
# Odd numbers will be centered on the _stable_footstep_offset
grid_size = (5, 5)

# Configure offset positions for the raycasters relative to each hip
# note that all axes are in the same orientation as the robot body frame
_stable_footstep_offsets = {
    _hip_names[0]: [0.1, -0.1],
    _hip_names[1]: [0.1, 0.1],
    _hip_names[2]: [-0.1, 0.1],
    _hip_names[3]: [-0.1, -0.1],
}
_height_scanner_offsets = {
    hip_name: RayCasterCfg.OffsetCfg(pos=(*_stable_footstep_offsets[hip_name], 20.0))
    for hip_name in _hip_names
}


# Configure other raycaster settings
_height_scanner_settings = {
    hip_name: {
        "prim_path": f"{{ENV_REGEX_NS}}/Robot/{hip_name}",
        "offset": _height_scanner_offsets[hip_name],
        "update_period": 0.00,  # every sim step
        "ray_alignment": "yaw",
        "pattern_cfg": patterns.GridPatternCfg(
            resolution=grid_resolution,
            size=[
                (grid_size[0] - 1) * grid_resolution,
                (grid_size[1] - 1) * grid_resolution,
            ],
        ),
        "debug_vis": True,
        "mesh_prim_paths": ["/World/ground"],
    }
    for hip_name in _hip_names
}

FR_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["FR_hip"]
)
FL_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["FL_hip"]
)
RL_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["RL_hip"]
)
RR_FootstepScannerCfg: Callable[[], RayCasterCfg] = lambda: RayCasterCfg(
    **_height_scanner_settings["RR_hip"]
)
