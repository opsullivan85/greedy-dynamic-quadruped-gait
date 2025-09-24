from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from src import sim2real
from src.simulation.cfg.scene import SceneCfg
import numpy as np

from src import get_logger
import env_cfg
from src.util import VectorPool

logger = get_logger()


@configclass
class GaitNetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(env_spacing=2.5)  # type: ignore

    # Basic settings
    observations: env_cfg.ObservationsCfg = env_cfg.ObservationsCfg()  # type: ignore
    actions: env_cfg.ActionsCfg = env_cfg.ActionsCfg()  # type: ignore
    events: env_cfg.EventsCfg = env_cfg.EventsCfg()  # type: ignore
    terminations: env_cfg.TerminationsCfg = env_cfg.TerminationsCfg()  # type: ignore
    rewards: env_cfg.RewardsCfg = env_cfg.RewardsCfg()  # type: ignore

    robot_controllers: VectorPool[sim2real.Sim2RealInterface] = None  # type: ignore to be set later

    # ['trunk', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
    hip_indices = np.asarray([1, 2, 3, 4], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    thigh_indices = np.asarray([5, 6, 7, 8], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    calf_indices = np.asarray([9, 10, 11, 12], dtype=np.int32)
    """In order: FL, FR, RL, RR"""
    foot_indices = np.asarray([13, 14, 15, 16], dtype=np.int32)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10  # env decimation -> 25 Hz footstep planning
        self.render_interval = (
            self.decimation / 2
        )  # render faster than footstep planning rate
        # simulation settings
        self.sim.dt = 0.004  # simulation timestep -> 250 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material

        self.episode_length_s = 5


def get_gaitnet_env_cfg(num_envs: int, device: str, iterations_between_mpc: int) -> GaitNetEnvCfg:
    """Get the gaitnet environment configuration.

    Args:
        num_envs (int): Number of environments.
        device (str): Device to use.

    Returns:
        GaitNetEnvCfg: The environment configuration.
    """
    cfg = GaitNetEnvCfg()
    controllers = VectorPool(
        instances=num_envs,
        cls=sim2real.SimInterface,
        dt=cfg.decimation * cfg.sim.dt,  # 500 Hz leg PD control
        iterations_between_mpc=iterations_between_mpc,
        debug_logging=False,
    )
    cfg.robot_controllers = controllers  # type: ignore
    cfg.actions.footstep_controller.robot_controllers = controllers  # type: ignore
    cfg.actions.mpc_controller.robot_controllers = controllers  # type: ignore


    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    # update sensor update periods
    # we tick all the sensors based on the smallest update period (physics update period)
    for scanner in [
        cfg.scene.FR_foot_scanner,
        cfg.scene.FL_foot_scanner,
        cfg.scene.RL_foot_scanner,
        cfg.scene.RR_foot_scanner,
        cfg.scene.contact_forces,
    ]:
        scanner.update_period = cfg.decimation * cfg.sim.dt  # control rate
    return cfg
