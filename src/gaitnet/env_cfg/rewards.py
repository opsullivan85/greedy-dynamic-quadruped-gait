import numpy as np
import torch
from isaaclab.assets import ArticulationData
from isaaclab.envs import ManagerBasedRLEnv, mdp
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp  # type: ignore
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from src.sim2real import Sim2RealInterface


def a_foot_in_swing(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    controllers: VectorPool[Sim2RealInterface] = env.cfg.robot_controllers  # type: ignore

    contacts: np.ndarray = controllers.call(
        Sim2RealInterface.get_contact_state, mask=None
    )
    # note we don't need to fix leg ordering here since we only care about the number of legs in swing
    num_legs_in_swing = np.sum(~contacts, axis=1)
    # atleast 1 leg in swing
    reward = (num_legs_in_swing >= 1).astype(np.float32)
    return torch.from_numpy(reward).to(env.device)


def height_below_minimum(
    env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    data: ArticulationData = env.scene[asset_cfg.name].data
    # properly index the body positions for the specified body IDs
    body_pos = data.body_link_pos_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 3)
    heights = body_pos[:, :, 2]  # (num_envs, num_bodies)
    # if any body is below the minimum height, incur a cost
    bodies_below = torch.sum((heights < minimum_height).long(), dim=1)
    return bodies_below


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # rewards
    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    a_foot_in_swing = RewTerm(func=a_foot_in_swing, weight=1.0)
    xy_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
        },
    )
    yaw_tracking = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
        },
    )

    # penalties
    terminating = RewTerm(func=mdp.is_terminated, weight=-300.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
    # foot_too_low = RewTerm(
    #     func=height_below_minimum,
    #     weight=-1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "minimum_height": 0.0,
    #     },
    # )
