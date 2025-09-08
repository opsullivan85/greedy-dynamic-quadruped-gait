import numpy as np
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.scene import InteractiveScene
from src.sim2real import SimInterface
from src.util.vectorpool import VectorPool


def isaac_joints_to_interface(
    joint_pos_isaac: np.ndarray, joint_vel_isaac: np.ndarray
) -> np.ndarray:
    """
    Convert Isaac Gym joint positions and velocities to the interface format.

    Parameters:
    - joint_pos_isaac: np.ndarray of shape (num_envs, 12)
    - joint_vel_isaac: np.ndarray of shape (num_envs, 12)

    Returns:
    - joint_states: np.ndarray of shape (num_envs, 4, 3, 2)
    """
    # Stack positions and velocities along the last axis
    joint_pos_vel = np.stack(
        [joint_pos_isaac, joint_vel_isaac], axis=-1
    )  # shape (:, 12, 2)

    # Reshape to (num_envs, 4, 3, 2) with Fortran-like index order
    joint_states_interface = joint_pos_vel.reshape(-1, 4, 3, 2, order="F")

    return joint_states_interface


def isaac_body_to_interface(body_state_isaac: np.ndarray) -> np.ndarray:
    """
    Convert Isaac Gym body position, orientation (quaternion), and velocity to the interface format.

    Parameters:
    - body_state_isaac: np.ndarray of shape (num_envs, 13)
        [
            pos_x, pos_y, pos_z,
            quat_w, quat_x, quat_y, quat_z,
            vel_x, vel_y, vel_z,
            omega_x, omega_y, omega_z
        ]

    Returns:
    - body_states_interface: np.ndarray of shape (num_envs, 13)
        [
            pos_x, pos_y, pos_z,
            quat_x, quat_y, quat_z, quat_w
            vel_x, vel_y, vel_z,
            omega_x, omega_y, omega_z
        ]
    """
    # move quatw to end and shift xyz to left
    body_states_interface = np.concatenate(
        [
            body_state_isaac[:, :3],  # pos_x, pos_y, pos_z
            body_state_isaac[:, 4:7],  # quat_x, quat_y, quat_z
            body_state_isaac[:, 3:4],  # quat_w
            body_state_isaac[:, 7:],  # rest
        ],
        axis=1,
    )

    return body_states_interface


def interface_to_isaac_torques(torques_interface: np.ndarray) -> np.ndarray:
    """
    Convert torques from the interface format back to Isaac Gym format.

    Parameters:
    - torques_interface: np.ndarray of shape (num_envs, 4, 3)

    Returns:
    - torques_isaac: np.ndarray of shape (num_envs, 12)
    """
    # Reshape to (num_envs, 12) with Fortran-like index order
    torques_isaac = torques_interface.reshape(-1, 12, order="F")

    return torques_isaac


def controls_to_joint_efforts(
    controls: np.ndarray, controllers: VectorPool, scene: InteractiveScene
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


def reset_all_to(
    env: ManagerBasedEnv,
    joint_pos_isaac: np.ndarray,
    joint_vel_isaac: np.ndarray,
    body_state_isaac: np.ndarray,
):
    """
    Reset all environments in the ManagerBasedEnv to the specified joint positions, velocities, and body states.

    Parameters:
    - env: ManagerBasedEnv instance
    - joint_pos_isaac: np.ndarray of shape (num_envs, 12), or if shape (12,), will be broadcasted
    - joint_vel_isaac: np.ndarray of shape (num_envs, 12), or if shape (12,), will be broadcasted
    - body_state_isaac: np.ndarray of shape (num_envs, 13), or if shape (13,), will be broadcasted
    """
    # broadcast vars
    num_envs = env.num_envs
    vars = {
        "joint_pos_isaac": (12, joint_pos_isaac),
        "joint_vel_isaac": (12, joint_vel_isaac),
        "body_state_isaac": (13, body_state_isaac),
    }
    for var_name, (expected_dim, var_value) in vars.items():
        if var_value.shape == (expected_dim,):
            var_value = np.tile(var_value, (num_envs, 1))
        elif var_value.shape != (num_envs, expected_dim):
            raise ValueError(
                f"{var_name} must have shape ({expected_dim},) or ({num_envs}, {expected_dim})"
            )

    # Reset the environments
    env.reset()
    # TODO: I'm 50% sure that we can just reset the controllers without
    # manually initializing the state. There may be issues with the contact history in the state estimator.
    controllers: VectorPool[SimInterface] = env.scene["robot"].controllers
    controllers.call(
        SimInterface.reset,
        mask=None,
    )

    # update base state
    env.scene["robot"].write_root_pose_to_sim(body_state_isaac[:, :7])
    env.scene["robot"].write_root_velocity_to_sim(body_state_isaac[:, 7:])

    # update joint state
    env.scene["robot"].write_joint_state_to_sim(joint_pos_isaac, joint_vel_isaac)
