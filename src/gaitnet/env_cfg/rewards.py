import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp  # type: ignore
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
