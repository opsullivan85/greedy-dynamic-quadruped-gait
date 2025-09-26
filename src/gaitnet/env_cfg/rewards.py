import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from src.sim2real import Sim2RealInterface


def a_foot_in_swing(
    env: ManagerBasedRLEnv
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    controllers: VectorPool[Sim2RealInterface] = env.cfg.robot_controllers  # type: ignore
    
    contacts: np.ndarray = controllers.call(
        Sim2RealInterface.get_contact_state,
        mask=None
    )
    # note we don't need to fix leg ordering here since we only care about the number of legs in swing
    num_legs_in_swing = np.sum(~contacts, axis=1)
    # atleast 1 leg in swing
    reward = (num_legs_in_swing >= 1).astype(np.float32)
    return torch.from_numpy(reward).to(env.device)

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.2)

    terminating = RewTerm(func=mdp.is_terminated, weight=-5.0)

    xy_tracking = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=0.4, params={
        "std": 2,
        "command_name": "base_velocity",
    })

    yaw_tracking = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.4, params={
        "std": 2,
        "command_name": "base_velocity",
    })

    a_foot_in_swing = RewTerm(func=a_foot_in_swing, weight=0.4)
