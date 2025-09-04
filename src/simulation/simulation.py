import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging

import numpy as np
import torch
from isaaclab.utils import configclass  # type: ignore
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # type: ignore

from src.simulation.cfg.manager_components import ActionsCfg, EventsCfg, ObservationsCfg, RewardsCfg, TerminationsCfg
from src.simulation.cfg.scene import SceneCfg
from src.sim2real import SimInterface, VectObjectPool
from src.simulation.util import (
    interface_to_isaac_torques,
    isaac_body_to_interface,
    isaac_joints_to_interface,
)

logger = logging.getLogger(__name__)


@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device

        self.episode_length_s = 5
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        for scanner in [
            self.scene.FR_foot_scanner,
            self.scene.FL_foot_scanner,
            self.scene.RL_foot_scanner,
            self.scene.RR_foot_scanner,
        ]:
            scanner.update_period = self.decimation * self.sim.dt  # 50 Hz


def main():
    """Main function."""
    # create environment configuration
    env_cfg = QuadrupedEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            # # reset
            # if count % 300 == 0:
            #     count = 0
            #     env.reset()
            #     print("-" * 80)
            #     print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn_like(env.action_manager.action)
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)
            # update counter
            count += 1

    # close the environment
    env.close()