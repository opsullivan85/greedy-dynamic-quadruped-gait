import argparse
import signal
from tracemalloc import start

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
from typing import Any, TypeAlias

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from src.sim2real import SimInterface, VectorPool
from src.simulation.util import controls_to_joint_efforts, reset_all_to
from src.util.data_logging import data_logger
from src.simulation.cfg.quadrupedenv import QuadrupedEnvCfg, get_quadruped_env_cfg
import src.simulation.cfg.footstep_scanner as fs
from nptyping import Float32, NDArray, Shape, Bool

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


# frozen allows for hashing
@dataclass(frozen=True)
class IsaacStateCPU:
    """Class for keeping track of an Isaac state."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_state: np.ndarray

    def to_torch(self, device: Any) -> "IsaacStateTorch":
        """Convert to torch tensors on the specified device."""
        return IsaacStateTorch(
            joint_pos=torch.from_numpy(self.joint_pos).to(device),
            joint_vel=torch.from_numpy(self.joint_vel).to(device),
            body_state=torch.from_numpy(self.body_state).to(device),
        )

@dataclass()
class IsaacStateTorch:
    """Class for keeping track of an Isaac state."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_state: torch.Tensor

    def to_numpy(self) -> IsaacStateCPU:
        """Convert to numpy arrays."""
        return IsaacStateCPU(
            joint_pos=self.joint_pos.cpu().numpy(),
            joint_vel=self.joint_vel.cpu().numpy(),
            body_state=self.body_state.cpu().numpy(),
        )


StepCostMap: TypeAlias = np.ndarray
"""(4 * n * m,) where n and m are the number of footstep positions"""


def get_step_locations_hip() -> NDArray[Shape["4, N, M ,2"], Float32]:
    """Get the footstep locations relative to the hip.

    Returns:
        np.ndarray: Array of footstep locations relative to the hip.
            (4, N, M, 2) where n and m are the number of footstep positions.
            in FL, FR, RL, RR order.
    """
    N, M = fs.grid_size
    leg = np.empty((N, M, 2), dtype=np.float32)
    half_size_x = (fs.grid_size[0] - 1) * fs.grid_resolution / 2
    half_size_y = (fs.grid_size[1] - 1) * fs.grid_resolution / 2
    x_locations = np.linspace(-half_size_x, half_size_x, fs.grid_size[0])
    y_locations = np.linspace(-half_size_y, half_size_y, fs.grid_size[1])
    for i, x in enumerate(x_locations):
        for j, y in enumerate(y_locations):
            leg[i, j] = [x, y]
    legs = np.tile(leg, (4, 1, 1, 1))
    return legs.astype(np.float32)


def check_dones(
    env: ManagerBasedEnv, obs: dict[str, dict[str, torch.Tensor]]
) -> tuple[NDArray[Shape["*"], Bool], np.ndarray]:
    """Check which footstep positions are done.

    Args:
        env (ManagerBasedEnv): The environment to check the dones from.
        obs (dict): The observation from the environment step.

    Returns:
        np.ndarray: Array of booleans indicating which footstep positions are done.
            (4*N*M) where n and m are the number of footstep positions.
            in FL, FR, RL, RR order.
        np.ndarray: State of the done robots
    """
    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    contact_states = controllers.call(
        SimInterface.get_contact_state,
        mask=None,
    )
    done_state = np.asarray((1, 1, 1, 1), dtype=bool)
    dones = np.all(contact_states == done_state, axis=1)

    states = np.full((env.num_envs,), None, dtype=object)
    indices = np.argwhere(dones)
    for index in indices:
        i = index[0]
        states[i] = IsaacStateTorch(
            env.scene["robot"].data.joint_pos[i],
            env.scene["robot"].data.joint_vel[i],
            env.scene["robot"].data.root_state_w[i],
        ).to_numpy()
    return dones, states

    # # TODO: implement
    # N, M = fs.grid_size
    # return np.zeros((4 * N * M,), dtype=bool)


def get_cost(an_obs: dict[str, torch.Tensor]) -> float:
    """Get the cost for a footstep position.

    Args:
        an_obs (dict): The observation from the environment step.

    Returns:
        float: The cost for the footstep position.
    """
    # TODO: implement
    return -1.0


def update_costs(
    step_cost_map: StepCostMap,
    dones: NDArray[Shape["*"], Bool],
    done_states: np.ndarray,
    env: ManagerBasedEnv,
    obs: dict[str, dict[str, torch.Tensor]],
) -> tuple[StepCostMap, NDArray[Shape["*"], Bool], np.ndarray]:
    """Update the costs for the footstep positions that are done.

    Args:
        step_cost_map (StepCostMap): Current cost map.
        dones (NDArray[Shape["*"], Bool]): Current done map.
        done_states (np.ndarray): Current done states.
        env (ManagerBasedEnv): The environment to get the rewards from.
        obs (dict): The observations from the environment step.

    Returns:
        StepCostMap: Updated cost map.
        NDArray[Shape["*"], Bool]: Updated done map.
    """
    currently_done, currently_done_states = check_dones(env, obs)
    newly_done = np.logical_and(currently_done, np.logical_not(dones))
    for idx in np.argwhere(newly_done):
        an_obs = {k: v[idx] for k, v in obs["policy"].items()}
        cost = get_cost(an_obs)
        step_cost_map[idx] = cost
    all_dones = np.logical_or(dones, currently_done)
    done_states[newly_done] = currently_done_states[newly_done]
    return step_cost_map, all_dones, done_states


def get_step_cost_map(
    env: ManagerBasedEnv,
    control: np.ndarray,
    state: IsaacStateTorch,
) -> tuple[StepCostMap, np.ndarray]:
    """Evaluates an instance

    An instance consists of a starting state and a control input.
    Then we run a simulation and try every footstep position.
    And return an array of costs.

    Args:
        env (ManagerBasedEnv): The environment to run the simulation in.
            expected to have an instance for every footstep position. (4*n*m)
        control (np.ndarray): Control input for the instance.
            (3,) i.e. not (num_envs, 3) since they all have the same control.
        state (IsaacStateTorch): State to initialize with.

    Returns:
        StepCostMap: Costs for each footstep position.
        np.ndarray: States for each footstep position if done, else None.
            Note: I'm pretty sure these will never be none, not 100% though
    """
    N, M = fs.grid_size
    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    footstep_locations_hip = get_step_locations_hip()

    # reset the environment to the desired state
    reset_all_to(
        env,
        state.joint_pos,
        state.joint_vel,
        state.body_state,
    )

    # array of n*m zeros, n*m ones, n*m twos, etc
    legs = np.repeat(np.arange(4, dtype=np.int32), N * M)
    durations = np.full((4 * N * M,), 0.2, dtype=np.float32)
    locations = footstep_locations_hip.reshape((4 * N * M, 2))

    # send all footstep commands
    controllers.call(
        SimInterface.initiate_footstep,
        mask=None,
        leg=legs,
        location_hip=locations,
        duration=durations,
    )

    control_vect = np.tile(control, (env.num_envs, 1))

    step_cost_map = np.zeros((4 * N * M,), dtype=np.float32)
    dones = np.zeros((4 * N * M,), dtype=bool)
    done_states = np.full((4 * N * M,), None, dtype=object)

    while simulation_app.is_running() and not shutdown_requested:  # Add flag check
        joint_efforts = controls_to_joint_efforts(control_vect, controllers, env.scene)

        # step the environment
        obs, _ = env.step(joint_efforts)  # type: ignore
        obs: dict[str, dict[str, torch.Tensor]] = obs

        step_cost_map, dones, done_states = update_costs(
            step_cost_map, dones, done_states, env, obs
        )
        if np.all(dones):
            break

    step_cost_map = step_cost_map.reshape((4, N, M))
    done_states = done_states.reshape((4, N, M))
    return step_cost_map, done_states


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

    state_cost_map: list[tuple[IsaacStateCPU, StepCostMap]] = []

    start_state: IsaacStateTorch = IsaacStateTorch(
        env.scene["robot"].data.joint_pos[0],
        env.scene["robot"].data.joint_vel[0],
        env.scene["robot"].data.root_state_w[0],
    )

    # simulate physics
    with controllers, torch.inference_mode():
        env_cfg.controllers = controllers
        while simulation_app.is_running() and not shutdown_requested:  # Add flag check
            # grab initial state from robot 0
            cost_map, terminal_states = get_step_cost_map(
                env,
                control=np.zeros((3,), dtype=np.float32),
                state=start_state,
            )
            state_cost_map.append((start_state.to_numpy(), cost_map))

            # pick new start state from one of the 10 lowest cost terminal states
            flat_cost_map = cost_map.flatten()
            # lowest_indices = np.argpartition(flat_cost_map, 10)[:10]
            lowest_indices = np.argpartition(flat_cost_map, 99)[:99]
            chosen_index = np.random.choice(lowest_indices)
            # the terminal states array is technically of IsaacStates
            state: IsaacStateCPU = terminal_states.flatten()[chosen_index]  # type: ignore
            start_state = state.to_torch(env.scene.device)

    env_cfg.controllers = None
    del controllers

    # close the environment
    env.close()


if __name__ == "__main__":
    from src.util import log_exceptions

    with log_exceptions(logger):
        main()
