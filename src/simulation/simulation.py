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

import isaaclab.sim as sim_utils  # type: ignore
import numpy as np
import torch
from isaaclab.scene import InteractiveScene  # type: ignore
from isaaclab.utils import configclass  # type: ignore
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg  # type: ignore

from src.simulation.cfg.manager_components import ActionsCfg, EventCfg, ObservationsCfg
from src.simulation.cfg.scene import SceneCfg
from src.sim2real import SimInterface, VectSim2Real
from src.simulation.util import (
    interface_to_isaac_torques,
    isaac_body_to_interface,
    isaac_joints_to_interface,
)

logger = logging.getLogger(__name__)


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device
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
    # setup base environment
    env_cfg = QuadrupedEnvCfg()
    env = ManagerBasedEnv(cfg=env_cfg)

    # load level policy
    # policy_path = ISAACLAB_NUCLEUS_DIR + "/Policies/ANYmal-C/HeightScan/policy.pt"
    # # check if policy file exists
    # if not check_file_path(policy_path):
    #     raise FileNotFoundError(f"Policy file '{policy_path}' does not exist.")
    # file_bytes = read_file(policy_path)
    # # jit load the policy
    # policy = torch.jit.load(file_bytes).to(env.device).eval()

    # simulate physics
    count = 0
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 1000 == 0:
                obs, _ = env.reset()
                count = 0
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # infer action
            # action = policy(obs["policy"])
            # make action all zeros for now
            action = torch.zeros(
                (env.num_envs, 12), device=env.device
            )
            # step env
            obs, _ = env.step(action)
            # update counter
            count += 1

    # close the environment
    env.close()