# import torch
# import isaaclab.sim as sim_utils
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils import configclass
# import isaaclab.assets.unitree as unitree_assets

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.flat_env_cfg import UnitreeGo1FlatEnvCfg as QuadrupedEnvCfg # pyright: ignore[reportMissingImports]

# @configclass
# class QuadrupedEnvCfg(DirectRLEnvCfg):
#     """Configuration for quadruped environment"""
    
#     # Environment settings
#     episode_length_s = 20.0
#     decimation = 4  # Control frequency: 1000Hz / 4 = 250Hz
#     num_envs = 1
#     num_observations = 436  # heightmap (400) + robot_state (36)
#     num_actions = 12
    
#     # Simulation settings
#     sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
#         dt=1.0/1000.0,  # 1000Hz physics
#         render_interval=decimation,
#     )
    
#     # Scene configuration
#     scene: InteractiveSceneCfg = InteractiveSceneCfg(
#         num_envs=num_envs,
#         env_spacing=2.5,
#         replicate_physics=True,
#     )
    
#     # Robot configuration (A1)
#     robot: ArticulationCfg = unitree_assets.UNITREE_A1_CFG.replace(
#         prim_path="/World/envs/env_.*/Robot",
#         spawn=sim_utils.SpawnCfg(
#             activate_contact_sensors=True,
#         ),
#     )
    
#     # Terrain configuration
#     terrain = TerrainImporterCfg(
#         prim_path="/World/ground",
#         terrain_type="plane",
#         collision_group=-1,
#         physics_material=sim_utils.RigidBodyMaterialCfg(
#             friction_combine_mode="multiply",
#             restitution_combine_mode="multiply",
#             static_friction=1.0,
#             dynamic_friction=1.0,
#         ),
#     )
    
#     # Height scanner for terrain perception
#     height_scanner: RayCasterCfg = RayCasterCfg(
#         prim_path="/World/envs/env_.*/Robot/base",
#         mesh_prim_paths=["/World/ground"],
#         pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.0]),
#         attach_yaw_only=True,
#         max_distance=3.0,
#         drift_range=(-0.5, 0.5),
#     )
    
#     # Contact sensors for feet
#     contact_sensor: ContactSensorCfg = ContactSensorCfg(
#         prim_path="/World/envs/env_.*/Robot/.*_foot",
#         track_pose=False,
#         track_air_time=True,
#     )