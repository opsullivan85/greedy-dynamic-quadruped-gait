import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp  # type: ignore
from isaaclab.utils import configclass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0
    )
