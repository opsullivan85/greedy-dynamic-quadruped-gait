import argparse

from isaaclab.app import AppLauncher  # type: ignore

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import logging

import isaaclab.sim as sim_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # type: ignore
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # type: ignore
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns  # type: ignore
from isaaclab.utils import configclass  # type: ignore
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg  # type: ignore
from isaaclab.terrains.height_field import hf_terrains_cfg  # type: ignore

from src.sim2real import VectSim2Real, SimInterface
from src.simulation.util import (
    interface_to_isaac_torques,
    isaac_body_to_interface,
    isaac_joints_to_interface,
)

logger = logging.getLogger(__name__)

from isaaclab.actuators import DCMotorCfg  # type: ignore
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # type: ignore

# override the default ActuatorNetMLPCfg motors so we can usetorque control
ROBOT_CFG_TORQUE = copy.deepcopy(UNITREE_GO1_CFG)
ROBOT_CFG_TORQUE.actuators["base_legs"] = DCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit=23.7,
    saturation_effort=23.7,
    velocity_limit=30.0,
    stiffness=0.0,
    damping=0.0,
)


# @configclass
# class Terrain(hf_terrains_cfg.HfSteppingStonesTerrainCfg):
#     def __post_init__(self):



@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # Replace ground plane with stepping stones terrain
    ground = TerrainImporterCfg(
        prim_path="/World/defaultGroundPlane",
        terrain_type="generator",
        # https://isaac-sim.github.io/IsaacLab/v2.1.0/source/api/lab/isaaclab.terrains.html#isaaclab.terrains.TerrainGeneratorCfg
        terrain_generator=TerrainGeneratorCfg(
            size=(10,10),
            difficulty_range=(0.5, 1.0),
            horizontal_scale = 0.01,
            vertical_scale = 0.5,
            # color_scheme="height",

            # https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.terrains.html#isaaclab.terrains.height_field.hf_terrains_cfg.HfSteppingStonesTerrainCfg
            sub_terrains={"stepping_stones": hf_terrains_cfg.HfSteppingStonesTerrainCfg(
                stone_height_max = 0.9,
                stone_width_range = (0.05, 0.1),
                stone_distance_range = (0.0, 0.3),
                # holes_depth = -0.5,
            )},
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.8,
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    robot: ArticulationCfg = ROBOT_CFG_TORQUE.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    control_interface = VectSim2Real(
        dt=sim.get_physics_dt(),
        instances=args_cli.num_envs,
        cls=SimInterface,
        debug_logging=False,
    )
    # make the first one log
    # control_interface.interfaces[0].logger = SimInterface.logger

    sim_dt = sim.get_physics_dt()
    logger.debug(f"dt: {sim_dt}")
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            logger.info("resetting the simulation")

            # for interface in control_interface.interfaces:
            #     interface.reset()

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
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            control_interface.call(
                function=SimInterface.reset,
                mask=None,
            )

        joint_pos = scene["robot"].data.joint_pos.cpu().numpy()
        joint_vel = scene["robot"].data.joint_vel.cpu().numpy()
        joint_states = isaac_joints_to_interface(joint_pos, joint_vel)

        body_state = scene["robot"].data.root_state_w.cpu().numpy()
        body_state = isaac_body_to_interface(body_state)

        command = np.zeros((args_cli.num_envs, 3), dtype=np.float32)

        torques_interface = control_interface.call(
            function=SimInterface.get_torques,
            mask=None,
            joint_states=joint_states,
            body_state=body_state,
            command=command,
        )
        torques_isaac_np = interface_to_isaac_torques(torques_interface)
        torques_isaac = torch.from_numpy(torques_isaac_np).to(scene.device)

        scene["robot"].set_joint_effort_target(torques_isaac)
        # fake_torques = np.asarray([[20.0,]*12,]*args_cli.num_envs)
        # fake_torques = torch.from_numpy(fake_torques).to(scene.device)
        # scene["robot"].set_joint_effort_target(fake_torques)

        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)


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
