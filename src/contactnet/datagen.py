import argparse
import signal
from tracemalloc import start

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
# parser.add_argument(
#     "--num_envs", type=int, default=1, help="Number of environments to spawn."
# )
parser.add_argument("--debug", action="store_true", help="Enable debug views.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args, unused_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import multiprocessing

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import rewards
import itertools

import src.contactnet.costs as cn_costs
from src.sim2real import SimInterface
from src.simulation.util import controls_to_joint_efforts, reset_all_to
from src.util import VectorPool
from src.simulation.cfg.quadrupedenv import QuadrupedEnvCfg, get_quadruped_env_cfg
import src.simulation.cfg.footstep_scanner as fs
from src.contactnet.debug import (
    view_footstep_cost_map,
    view_multiple_footstep_cost_maps,
)
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


def get_step_locations_hip() -> NDArray[Shape["4, N, M ,2"], Float32]:
    """Get the footstep locations relative to the hip.

    Returns:
        np.ndarray: Array of footstep locations relative to the hip.
            (4, N, M, 2) where n and m are the number of footstep positions.
            in FR, FL, RR, RL order.
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


def _controller_dones(
    controllers: VectorPool[SimInterface],
) -> NDArray[Shape["*"], Bool]:
    """Check which controllers are done.

    Args:
        controllers (VectorPool[SimInterface]): The controllers to check.

    Returns:
        np.ndarray: Array of booleans indicating which controllers are done.
    """
    done_state = np.asarray((1, 1, 1, 1), dtype=bool)
    contact_states = controllers.call(
        SimInterface.get_contact_state,
        mask=None,
    )
    dones = np.all(contact_states == done_state, axis=1)
    return dones


def _contact_sensor_dones(
    env: ManagerBasedEnv,
) -> NDArray[Shape["*"], Bool]:
    """Check which robots are done based on contact sensors.

    Args:
        env (ManagerBasedEnv): The environment to check.

    Returns:
        np.ndarray: Array of booleans indicating which robots are done.
    """
    contact_forces = env.scene["contact_forces"].data.net_forces_w
    # robot is done if all feet have contact force above threshold
    foot_contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    dones = np.all(foot_contacts.cpu().numpy(), axis=1)
    return dones


def check_dones(env: ManagerBasedEnv) -> tuple[NDArray[Shape["*"], Bool], np.ndarray]:
    """Check which footstep positions are done.

    Args:
        env (ManagerBasedEnv): The environment to check the dones from.
        obs (dict): The observation from the environment step.

    Returns:
        np.ndarray: Array of booleans indicating which footstep positions are done.
            (4*N*M) where n and m are the number of footstep positions.
            in FR, FL, RR, RL order.
        np.ndarray: State of the done robots
    """
    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    controller_dones = _controller_dones(controllers)
    dones = np.zeros((env.num_envs,), dtype=bool)
    if np.any(controller_dones):
        contact_sensor_dones = _contact_sensor_dones(env)
        dones = np.logical_and(controller_dones, contact_sensor_dones)

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


class CostManager:
    def __init__(
        self,
        env: ManagerBasedEnv,
        running_costs: list[cn_costs.RunningCost],
        terminal_costs: list[cn_costs.TerminalCost],
    ):
        self.running_costs = running_costs
        self.terminal_costs = terminal_costs
        self.dones = np.zeros((env.num_envs,), dtype=bool)
        self.done_states = np.full((env.num_envs,), None, dtype=object)
        self.penalties = np.zeros((env.num_envs,), dtype=np.float32)
        self.costs: np.ndarray = np.zeros((env.num_envs,), dtype=np.float32)

    def update(self, env: ManagerBasedEnv, new_dones: NDArray[Shape["*"], Bool]):
        for cost in self.running_costs:
            cost.update_running_cost(env)

        if np.any(new_dones):
            costs = np.zeros((env.num_envs,), dtype=np.float32)
            for func in itertools.chain(self.terminal_costs, self.running_costs):
                costs += func.terminal_cost(env).cpu().numpy()
            self.costs[new_dones] = costs[new_dones]

    def get_cost_map(self, env: ManagerBasedEnv) -> NDArray[Shape["*"], Float32]:
        """Get the cost map for the environment.

        Returns:
            NDArray[Shape["*"], Float32]: The cost map for the environment.
        """
        return self.costs + self.penalties

    def get_dones(
        self, env: ManagerBasedEnv
    ) -> tuple[NDArray[Shape["*"], Bool], NDArray[Shape["*"], Bool]]:
        """Get the done states for the environment.

        Args:
            env (ManagerBasedEnv): The environment to get the done states from.

        Returns:
            NDArray[Shape["*"], Bool]: Array of booleans indicating which robots have finished.
            NDArray[Shape["*"], Bool]: Array of booleans indicating which robots are newly done.
        """
        currently_done, currently_done_states = check_dones(env)
        newly_done = np.logical_and(currently_done, np.logical_not(self.dones))
        self.dones = np.logical_or(self.dones, currently_done)
        self.done_states[newly_done] = currently_done_states[newly_done]
        return self.dones, newly_done

    def get_done_states(self) -> np.ndarray:
        """Get the done states for the environment.

        Returns:
            np.ndarray[None | IsaacStateCPU]: Array of done states for each robot.
                None: if the robot is not done.
                IsaacStateCPU: if the robot is done.
        """
        return self.done_states

    def apply_penalty(self, mask: NDArray[Shape["*"], Bool], penalty: float = 1.0):
        """Apply a penalty to the done states.

        Args:
            mask (NDArray[Shape["*"], Bool]): Boolean mask of which robots to apply the penalty to.
            penalty (float, optional): Penalty to apply. Defaults to 1.0.
        """
        self.penalties[mask] += penalty

    def debug_plot(self, env: ManagerBasedEnv):
        cost_maps = []
        titles = []
        for cost in itertools.chain(self.running_costs, self.terminal_costs):
            cost_maps.append(
                cost.terminal_cost(env)
                .cpu()
                .numpy()
                .reshape((4, fs.grid_size[0], fs.grid_size[1]))
            )
            titles.append(cost.name)
        view_multiple_footstep_cost_maps(cost_maps, titles=titles)


def get_step_cost_map(
    env: ManagerBasedEnv,
    control: np.ndarray,
    state: IsaacStateTorch,
) -> tuple[NDArray[Shape["4, N, M"], Float32], NDArray[Shape["4, N, M"], Any]]:
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
        np.ndarray: Costs for each footstep position.
            (4, N, M) where n and m are the number of footstep positions
        np.ndarray: States for each footstep position if done, else None.
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
    control_gpu = torch.from_numpy(control_vect).to(env.scene.device)

    max_time_s = 0.25  # max time to simulate
    elapsed_time_s = 0.0

    cost_manager = CostManager(
        env,
        running_costs=[
            # cn_costs.SimpleIntegrator(0.8, rewards.lin_vel_z_l2),  # type: ignore
            cn_costs.BallanceFootCosts(
                cn_costs.SimpleIntegrator(0.06, rewards.ang_vel_xy_l2)  # type: ignore
            ),
            # cn_costs.SimpleIntegrator(4e-4, rewards.joint_torques_l2),  # type: ignore
            # cn_costs.SimpleIntegrator(5e-8, rewards.joint_acc_l2),  # type: ignore
            cn_costs.BallanceFootCosts(
                cn_costs.ControlErrorCost(1.0, control_gpu, env)
            ),
        ],
        terminal_costs=[
            # cn_costs.SimpleTerminalCost(0.5, cn_costs.controller_swing_error),
            # cn_costs.SimpleTerminalCost(-1.5, cn_costs.support_polygon_area),
            cn_costs.SimpleTerminalCost(-1.5, cn_costs.inscribed_circle_radius),
            cn_costs.SimpleTerminalCost(1.0, cn_costs.foot_hip_distance),
        ],
    )

    while simulation_app.is_running() and not shutdown_requested:  # Add flag check
        joint_efforts = controls_to_joint_efforts(control_vect, controllers, env.scene)

        # step the environment
        obs, _ = env.step(joint_efforts)  # type: ignore
        obs: dict[str, dict[str, torch.Tensor]] = obs

        dones, new_dones = cost_manager.get_dones(env)
        cost_manager.update(env, new_dones)

        elapsed_time_s += env.step_dt

        if np.all(dones) or elapsed_time_s >= max_time_s:
            # apply a cost penalty for not finishing
            cost_manager.apply_penalty(~dones, float('inf'))
            break

    if args.debug:
        cost_manager.debug_plot(env)
    step_cost_map = cost_manager.get_cost_map(env).reshape((4, N, M))
    done_states = cost_manager.get_done_states().reshape((4, N, M))
    return step_cost_map, done_states


def main():
    """Main function."""
    # 4 since there are 4 feet
    num_envs = 4 * fs.grid_size[0] * fs.grid_size[1]
    # create environment configuration
    env_cfg: QuadrupedEnvCfg = get_quadruped_env_cfg(num_envs, args.device)
    # setup RL environment
    env = ManagerBasedEnv(cfg=env_cfg)
    iterations_between_mpc = 10  # 50 Hz MPC
    controllers = VectorPool(
        instances=num_envs,
        cls=SimInterface,
        dt=env_cfg.sim.dt * env_cfg.decimation,  # 500 Hz leg PD control
        iterations_between_mpc=iterations_between_mpc,
        debug_logging=False,
    )

    state_cost_map: list[tuple[IsaacStateCPU, NDArray[Shape["4, N, M"], Float32]]] = []

    start_state: IsaacStateTorch = IsaacStateTorch(
        env.scene["robot"].data.joint_pos[0],
        env.scene["robot"].data.joint_vel[0],
        env.scene["robot"].data.root_state_w[0],
    )

    control = np.array([0.1, 0.0, 0.0], dtype=np.float32)

    # simulate physics
    with controllers, torch.inference_mode():
        env_cfg.controllers = controllers

        for i in itertools.count():
            logger.info(f"Iteration {i}")
            if not (simulation_app.is_running() and not shutdown_requested):
                break
            # grab initial state from robot 0
            cost_map, terminal_states = get_step_cost_map(
                env,
                control=control,
                state=start_state,
            )
            state_cost_map.append((start_state.to_numpy(), cost_map))

            # pick new start state from one of the n lowest cost terminal states
            flat_cost_map = cost_map.flatten()
            pick_from_best_n = 5
            lowest_indices = np.argpartition(flat_cost_map, pick_from_best_n)[
                :pick_from_best_n
            ]
            chosen_index = np.random.choice(lowest_indices)
            # the terminal states array is technically of IsaacStates
            state: IsaacStateCPU = terminal_states.flatten()[chosen_index]  # type: ignore
            start_state = state.to_torch(env.scene.device)

            if args.debug:
                view_footstep_cost_map(
                    cost_map,
                    np.unravel_index(chosen_index, cost_map.shape),
                    title="Footstep Cost Map",
                )

    env_cfg.controllers = None
    del controllers

    # close the environment
    env.close()


if __name__ == "__main__":
    from src.util import log_exceptions

    with log_exceptions(logger):
        main()
