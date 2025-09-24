from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    xy_tracking = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=0.4, params={
        "std": 2,
        "command_name": "base_velocity",
    })

    yaw_tracking = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.4, params={
        "std": 2,
        "command_name": "base_yaw",
    })
