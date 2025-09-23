"""Action for running low level control with an MPC
"""

from dataclasses import MISSING
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

class MPCActionTerm(ActionTerm):
    """MPC Action Term
    """
    def __init__(self, cfg: "MPCActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)


@configclass
class MPCActionCfg(ActionTermCfg):
    """Configuration for the MPC Action Term
    """

    class_type: type[ActionTerm] = MPCActionTerm
