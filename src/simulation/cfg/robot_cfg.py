import copy
import logging

import isaaclab.sim as sim_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # type: ignore
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # type: ignore
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # type: ignore
from isaaclab.utils import configclass  # type: ignore
from isaaclab.actuators import DCMotorCfg  # type: ignore
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # type: ignore


# override the default ActuatorNetMLPCfg motors so we can usetorque control
ROBOT_CFG = copy.deepcopy(UNITREE_GO1_CFG)
ROBOT_CFG.actuators["base_legs"] = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=23.7,
    saturation_effort=23.7,
    velocity_limit=30.0,
    stiffness=0.0,
    damping=0.0,
)