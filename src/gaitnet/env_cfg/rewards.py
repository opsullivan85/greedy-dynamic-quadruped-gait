import numpy as np
import torch
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # rewards
    alive = RewTerm(func=mdp.is_alive, weight=0.2)
    a_foot_in_swing = RewTerm(func=a_foot_in_swing, weight=0.25)
    xy_tracking = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.5,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
        },
    )
    yaw_tracking = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={
            "std": 0.5,
            "command_name": "base_velocity",
        },
    )

    # penalties
    terminating = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5)
    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )
