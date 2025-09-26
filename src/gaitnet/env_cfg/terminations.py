from isaaclab.envs import mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "limit_angle": math.radians(20),
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
        },
    )
    bad_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.2,
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
        },
    )
