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
from isaaclab.scene import InteractiveSceneCfg

from src.simulation.cfg.manager_components import (
    ActionsCfg,
    EventsCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)
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


def controls_to_joint_efforts(
    controls: np.ndarray, controllers: VectObjectPool, scene: InteractiveSceneCfg
) -> torch.Tensor:
    joint_pos = scene["robot"].data.joint_pos.cpu().numpy()
    joint_vel = scene["robot"].data.joint_vel.cpu().numpy()
    joint_states = isaac_joints_to_interface(joint_pos, joint_vel)

    body_state = scene["robot"].data.root_state_w.cpu().numpy()
    body_state = isaac_body_to_interface(body_state)

    torques_interface = controllers.call(
        function=SimInterface.get_torques,
        mask=None,
        joint_states=joint_states,
        body_state=body_state,
        command=controls,
    )
    torques_isaac_np = interface_to_isaac_torques(torques_interface)
    torques_isaac = torch.from_numpy(torques_isaac_np).to(scene.device)
    return torques_isaac


def walk_in_place(
    count: int, step_state: bool, control_interface: VectObjectPool
) -> bool:
    # step in place every 200ms
    if count % 40 == 0:
        if step_state:
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([0]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([0.1, 0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([3]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([-0.1, -0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
        else:
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([1]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([0.1, -0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([2]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([-0.1, 0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
        return not step_state
    return step_state


def main():
    """Main function."""
    # create environment configuration
    env_cfg = QuadrupedEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    # setup RL environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    make_controllers = lambda: VectObjectPool(
        instances=args_cli.num_envs,
        cls=SimInterface,
        dt=env_cfg.sim.dt,
        # dt=env_cfg.sim.dt * env_cfg.decimation,
        debug_logging=False,
    )
    step_state = False

    # simulate physics
    count = 0
    with make_controllers() as controllers:
        while simulation_app.is_running():
            with torch.inference_mode():

                step_state = walk_in_place(count, step_state, controllers)
                command = np.zeros((args_cli.num_envs, 3), dtype=np.float32)
                command[:, 0] = 0.3
                command[:, 2] = 0.2
                joint_efforts = controls_to_joint_efforts(command, controllers, env.scene)

                # step the environment
                obs, rew, terminated, truncated, info = env.step(joint_efforts)

                # update counter
                count += 1

    # close the environment
    env.close()
