# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to add and simulate on-board sensors for a robot.

We add the following sensors on the quadruped robot, ANYmal-C (ANYbotics):

* Height Scanner: This is a height scanner sensor that is attached to the robot's base.
* Contact Sensor: This is a contact sensor that is attached to the robot's feet.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/04_sensors/add_sensors_on_robot.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
from distutils import command

from isaaclab.app import AppLauncher  # type: ignore

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils  # type: ignore
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # type: ignore
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # type: ignore
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # type: ignore
from isaaclab.utils import configclass  # type: ignore
import numpy as np

import src.robotinterface.siminterface as SimInterface
from src.robotinterface.interface import RobotInterfaceVect
import logging
logger = logging.getLogger(__name__)

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG as ROBOT_CFG  # type: ignore


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    # friction to simulate rubber on concrete
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.8),
    )
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=ground)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        update_period=0.02,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/defaultGroundPlane"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=0.0, history_length=6, debug_vis=True
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    control_interface = RobotInterfaceVect(dt=sim.get_physics_dt(), instances=args_cli.num_envs, cls=SimInterface.SimInterface, debug_logging=False)
    # make the first one log
    control_interface.interfaces[0].logger = SimInterface.logger

    sim_dt = sim.get_physics_dt()
    logger.debug(f"dt: {sim_dt}")
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            logger.info("resetting the simulation")

            for interface in control_interface.interfaces:
                interface.reset()

            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = (
                scene["robot"].data.default_joint_pos.clone(),
                scene["robot"].data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
        # Apply default actions to the robot
        # -- generate actions/commands
        # targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        # scene["robot"].set_joint_position_target(targets)

        joint_pos = scene["robot"].data.joint_pos.cpu().numpy()
        joint_vel = scene["robot"].data.joint_vel.cpu().numpy()
        joint_pos_vel = np.stack([joint_pos, joint_vel], axis=-1)  # shape (:, 12, 2)
        # order F makes the indices correspond to the expected format for the interface
        joint_states = joint_pos_vel.reshape(-1, 4, 3, 2, order="F")

        body_pos = scene["robot"].data.root_pos_w.cpu().numpy()
        body_quat = scene["robot"].data.root_quat_w.cpu().numpy()
        # re-order body quat from wxyz to xyzw
        body_quat = np.concatenate([body_quat[..., 3:4], body_quat[..., :3]], axis=-1)
        # this is both linear and angular velocity
        body_vel = scene["robot"].data.root_vel_w.cpu().numpy()
        body_state = np.concatenate([body_pos, body_quat, body_vel], axis=-1)  # shape (:, 13)

        # logger.critical(scene["robot"].data.joint_names)
        # exit(0)

        command = np.zeros((args_cli.num_envs, 3), dtype=np.float32)

        torques = control_interface.get_torques(
            joint_states=joint_states,
            body_states=body_state,
            commands=command,
        )
        # order F makes the indices correspond to the expected format for the simulator
        torques = torques.reshape((-1, 12), order="F")
        torques = torch.from_numpy(torques).to(scene.device)

        scene["robot"].set_joint_effort_target(torques)


        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # print information from the sensors
        # print("-------------------------------")
        # print(scene["height_scanner"])
        # print("Received max height value: ", torch.max(scene["height_scanner"].data.ray_hits_w[..., -1]).item())
        # print("-------------------------------")
        # print(scene["contact_forces"])
        # print("Received max contact force of: ", torch.max(scene["contact_forces"].data.net_forces_w).item())


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    logger.info("setup complete")
    # Run the simulator
    run_simulator(sim, scene)

    simulation_app.close()