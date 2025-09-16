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
from typing import Any, TypeAlias
from itertools import count

from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp import rewards

import src.contactnet.rewards as cn_rewards
from src.sim2real import SimInterface
from src.simulation.util import controls_to_joint_efforts, reset_all_to
from src.util import VectorPool
from src.simulation.cfg.quadrupedenv import QuadrupedEnvCfg, get_quadruped_env_cfg
import src.simulation.cfg.footstep_scanner as fs
from src.contactnet.debug import view_footstep_cost_map
from nptyping import Float32, Int32, NDArray, Shape, Bool

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


def get_costs(
    env: ManagerBasedEnv,
    control: torch.Tensor,
    mask: NDArray[Shape["*"], Bool] | None = None,
) -> NDArray[Shape["*"], Float32]:
    """Get the cost for a footstep position.

    Args:
        env (ManagerBasedEnv): The environment to get the costs from.
        mask (NDArray[Shape["*"], Bool]): Boolean mask of which footstep positions to get costs for.
            Only used in debug visualization.
        control (NDArray[Shape["*"], Float32]): Control input for the instance.

    Returns:
        np.ndarray: Array of costs for each footstep position.
    """
    # TODO: add in:
    # - foot position errors (to avoid slipping)
    # - support polygon stability (to avoid falling)
    costs = torch.zeros((env.num_envs,), device=env.scene.device)

    # lin_vel_z_l2_cost = 1.0 * rewards.lin_vel_z_l2(env) / 6  # type: ignore
    # costs += lin_vel_z_l2_cost

    # ang_vel_xy_l2_cost = 0.05 * rewards.ang_vel_xy_l2(env) / 6  # type: ignore
    # costs += ang_vel_xy_l2_cost

    # joint_torques_l2_cost = 1.0e-5 * rewards.joint_torques_l2(env) * 5  # type: ignore
    # costs += joint_torques_l2_cost

    # joint_acc_l2_cost = 1.0e-7 * rewards.joint_acc_l2(env) / 4  # type: ignore
    # costs += joint_acc_l2_cost

    swing_error_cost = 1.0 * cn_rewards.controller_real_swing_error(env)
    costs += swing_error_cost

    support_polygon_cost = -1.5 * cn_rewards.support_polygon_area(env)
    costs += support_polygon_cost

    inscribed_circle_radius = -1.5 * cn_rewards.inscribed_circle_radius(env)
    costs += inscribed_circle_radius

    foot_distance = 1.0 * cn_rewards.foot_hip_distance(env)
    costs += foot_distance

    control_alignment = 0.75 * cn_rewards.control_velocity_alignment(env, control)
    costs += control_alignment

    if args.debug:
        # skip if less than 10% of the envs are being visualized
        if mask is not None and np.sum(mask) < 0.5 * env.num_envs:
            return costs.cpu().numpy()
        all = torch.stack(
            [
                # lin_vel_z_l2_cost,
                # ang_vel_xy_l2_cost,
                # joint_torques_l2_cost,
                # joint_acc_l2_cost,
                swing_error_cost,
                support_polygon_cost,
                inscribed_circle_radius,
                foot_distance,
                control_alignment,
            ]
        )
        if mask is not None:
            vmin = float(torch.min(all[:, mask]))  # type: ignore
            vmax = float(torch.max(all[:, mask]))  # type: ignore
        else:
            vmin = float(torch.min(all))
            vmax = float(torch.max(all))
        label_map = {
            # "lin_vel_z_l2_cost": lin_vel_z_l2_cost,
            # "ang_vel_xy_l2_cost": ang_vel_xy_l2_cost,
            # "joint_torques_l2_cost": joint_torques_l2_cost,
            # "joint_acc_l2_cost": joint_acc_l2_cost,
            "swing_error_cost": swing_error_cost,
            "support_polygon_cost": support_polygon_cost,
            "inscribed_circle_radius": inscribed_circle_radius,
            "foot_distance": foot_distance,
            "control_alignment": control_alignment,
        }
        for label, cost in label_map.items():
            cost = cost.cpu().numpy()
            if mask is not None:
                cost = np.where(mask, cost, np.nan)
            view_footstep_cost_map(
                cost.reshape((4, fs.grid_size[0], fs.grid_size[1])),
                title=label,
                # vmin=vmin,
                # vmax=vmax,
            )

    # move costs to cpu
    costs = costs.cpu().numpy()
    return costs


def update_costs(
    step_cost_map: NDArray[Shape["*"], Float32],
    dones: NDArray[Shape["*"], Bool],
    done_states: np.ndarray,
    env: ManagerBasedEnv,
    obs: dict[str, dict[str, torch.Tensor]],
    control: torch.Tensor,
) -> tuple[NDArray[Shape["4, N, M"], Float32], NDArray[Shape["*"], Bool], np.ndarray]:
    """Update the costs for the footstep positions that are done.

    Args:
        step_cost_map (NDArray[Shape["*"], Float32]): Current cost map.
        dones (NDArray[Shape["*"], Bool]): Current done map.
        done_states (np.ndarray): Current done states.
        env (ManagerBasedEnv): The environment to get the rewards from.
        obs (dict): The observations from the environment step.
        control (NDArray[Shape["*"], Float32]): Control input for the instance.

    Returns:
        NDArray[Shape["*"], Float32]: Updated cost map.
        NDArray[Shape["*"], Bool]: Updated done map.
    """
    currently_done, currently_done_states = check_dones(env, obs)
    newly_done = np.logical_and(currently_done, np.logical_not(dones))
    if np.any(newly_done):
        costs = get_costs(env, mask=newly_done, control=control)
        step_cost_map[newly_done] = costs[newly_done]
    all_dones = np.logical_or(dones, currently_done)
    done_states[newly_done] = currently_done_states[newly_done]
    return step_cost_map, all_dones, done_states


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
    control_gpu = torch.from_numpy(control_vect).to(env.scene.device)

    step_cost_map = np.zeros((4 * N * M,), dtype=np.float32)
    dones = np.zeros((4 * N * M,), dtype=bool)
    done_states = np.full((4 * N * M,), None, dtype=object)
    max_time_s = 0.25  # max time to simulate
    elapsed_time_s = 0.0

    while simulation_app.is_running() and not shutdown_requested:  # Add flag check
        joint_efforts = controls_to_joint_efforts(control_vect, controllers, env.scene)

        # step the environment
        obs, _ = env.step(joint_efforts)  # type: ignore
        obs: dict[str, dict[str, torch.Tensor]] = obs

        step_cost_map, dones, done_states = update_costs(
            step_cost_map, dones, done_states, env, obs, control=control_gpu
        )
        elapsed_time_s += env.cfg.sim.dt * env.cfg.decimation
        if np.all(dones):
            break
        if elapsed_time_s >= max_time_s:
            # apply a cost penalty for not finishing
            step_cost_map[~dones] += 1.0
            break

    step_cost_map = step_cost_map.reshape((4, N, M))
    done_states = done_states.reshape((4, N, M))
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

        for i in count():
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

            # pick new start state from one of the 10 lowest cost terminal states
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
