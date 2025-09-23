import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp  # type: ignore
from isaaclab.utils import configclass
from src.gaitnet.actions.footstep_action import FSCActionCfg
from src.gaitnet.actions.hierarchical_action import HierarchicalActionCfg
from src.gaitnet.actions.mpc_action import MPCActionCfg


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    footstep_controller = FSCActionCfg(
        asset_name="robot",
        robot_controllers=None,  # to be set later
    )

    # trick type hinting system into giving us better hints by faking
    # type to be MPCActionCfg
    mpc_controller: MPCActionCfg = HierarchicalActionCfg(  # type: ignore
        action_cfg=MPCActionCfg(
            robot_controllers=None,  # to be set later
            asset_name="robot",
            joint_names=[".*"],  # all joints
        ),
        skip=1,  # run at full speed
        asset_name="robot",
    )

