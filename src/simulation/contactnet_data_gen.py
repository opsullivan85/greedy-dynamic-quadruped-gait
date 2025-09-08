import argparse
import signal

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
# parser.add_argument(
#     "--num_envs", type=int, default=1, help="Number of environments to spawn."
# )
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import multiprocessing

import numpy as np
import torch
from dataclasses import dataclass
from typing import TypeAlias

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from src.sim2real import SimInterface, VectorPool
from src.simulation.util import controls_to_joint_efforts, reset_all_to
from src.util.data_logging import data_logger
from src.simulation.cfg.quadrupedenv import QuadrupedEnvCfg, get_quadruped_env_cfg
import src.simulation.cfg.footstep_scanner as fs

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    if multiprocessing.current_process().name == "MainProcess":
        signal_name = signal.Signals(sig).name
        logger.info(f"Signal {signal_name} received in main process, shutting down...")
    shutdown_requested = True


# Set up the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


@dataclass
class IsaacState:
    """Class for keeping track of an Isaac state."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_state: torch.Tensor


StepCostMap: TypeAlias = np.ndarray
"""(4, n, m) where n and m are the number of footstep positions"""


def get_step_locations_hip() -> np.ndarray:
    """Get the footstep locations relative to the hip.

    Returns:
        np.ndarray: Array of footstep locations relative to the hip.
            (4, n*m, 2) where n and m are the number of footstep positions.
            in FL, FR, RL, RR order.
    """
    locations = []
    for _ in range(4):
        half_size_x = (fs.grid_size[0] - 1) * fs.grid_resolution / 2
        half_size_y = (fs.grid_size[1] - 1) * fs.grid_resolution / 2
        x_locations = np.linspace(-half_size_x, half_size_x, fs.grid_size[0])
        y_locations = np.linspace(-half_size_y, half_size_y, fs.grid_size[1])
        locations.append([[x, y] for x in x_locations for y in y_locations])
    locations = np.asarray(locations, dtype=np.float32)
    return locations


def get_step_cost_map(
    env: ManagerBasedEnv,
    control: np.ndarray,
    state: IsaacState,
) -> StepCostMap:
    """Evaluates an instance

    An instance consists of a starting state and a control input.
    Then we run a simulation and try every footstep position.
    And return an array of costs.

    Args:
        env (ManagerBasedEnv): The environment to run the simulation in.
            expected to have an instance for every footstep position. (4*n*m)
        control (np.ndarray): Control input for the instance.
            (3,) i.e. not (num_envs, 3) since they all have the same control.
        state (IsaacState): State to initialize with.

    Returns:
        StepCostMap: Costs for each footstep position.
    """
    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    footstep_locations_hip = get_step_locations_hip()

    # reset the environment to the desired state
    reset_all_to(
        env,
        state.joint_pos,
        state.joint_vel,
        state.body_state,
    )
    

    # send all footstep commands
    controllers.call(
        SimInterface.initiate_footstep,
        legs,
        locations,
        durations,
    )
        

    # with torch.inference_mode():
    #     while True:
    #         command = np.zeros((args_cli.num_envs, 3), dtype=np.float32)
    #         command[:, 0] = 0.3
    #         command[:, 2] = 0.2
    #         joint_efforts = controls_to_joint_efforts(command, controllers, env.scene)

    #         # step the environment
    #         obs, _ = env.step(joint_efforts)  # type: ignore
    #         obs: dict[str, dict[str, torch.Tensor]] = obs


def main():
    """Main function."""
    # 4 since there are 4 feet
    num_envs = 4 * fs.grid_size[0] * fs.grid_size[1]
    # create environment configuration
    env_cfg: QuadrupedEnvCfg = get_quadruped_env_cfg(num_envs, args_cli.device)
    # setup RL environment
    env = ManagerBasedEnv(cfg=env_cfg)
    iterations_between_mpc = 2  # 50 Hz MPC
    controllers = VectorPool(
        instances=num_envs,
        cls=SimInterface,
        dt=env_cfg.sim.dt * env_cfg.decimation,  # 100 Hz leg PD control
        iterations_between_mpc=iterations_between_mpc,
        debug_logging=False,
    )

    # simulate physics
    with controllers, torch.inference_mode():
        env_cfg.controllers = controllers
        while simulation_app.is_running() and not shutdown_requested:  # Add flag check
            state: IsaacState = IsaacState(
                joint_pos=torch.zeros((num_envs, 12), device=args_cli.device),
                joint_vel=torch.zeros((num_envs, 12), device=args_cli.device),
                body_state=torch.zeros((num_envs, 13), device=args_cli.device),
            )
            get_step_cost_map(
                env,
                control=np.zeros((num_envs, 3), dtype=np.float32),
                state=state,
            )

    env_cfg.controllers = None
    del controllers

    # close the environment
    env.close()


if __name__ == "__main__":
    from src.util import log_exceptions

    with log_exceptions(logger):
        main()
