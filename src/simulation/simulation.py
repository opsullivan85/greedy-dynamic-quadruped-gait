import argparse

from isaaclab.app import AppLauncher
from src.simulation.cfg.manager_components import ActionsCfg, EventCfg, ObservationsCfg
from src.simulation.cfg.scene import SceneCfg

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

import logging

import isaaclab.sim as sim_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # type: ignore
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # type: ignore
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg  # type: ignore
from isaaclab.utils import configclass  # type: ignore

from src.sim2real import SimInterface, VectSim2Real
from src.simulation.cfg.footstep_scanner import (
    FL_FootstepScannerCfg,
    FR_FootstepScannerCfg,
    RL_FootstepScannerCfg,
    RR_FootstepScannerCfg,
)
from src.simulation.cfg.robot import ROBOT_CFG
from src.simulation.cfg.terrain import VoidTerrainImporterCfg
from src.simulation.util import (
    interface_to_isaac_torques,
    isaac_body_to_interface,
    isaac_joints_to_interface,
)

logger = logging.getLogger(__name__)


@configclass
class QuadrupedEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # env decimation -> 50 Hz control
        # simulation settings
        self.sim.dt = 0.005  # simulation timestep -> 200 Hz physics
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = args_cli.device
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt  # 50 Hz


def reset_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    control_interface: VectSim2Real,
):
    logger.info("resetting the simulation")
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


def set_torques(
    command: np.ndarray, scene: InteractiveScene, control_interface: VectSim2Real
):
    joint_pos = scene["robot"].data.joint_pos.cpu().numpy()
    joint_vel = scene["robot"].data.joint_vel.cpu().numpy()
    joint_states = isaac_joints_to_interface(joint_pos, joint_vel)

    body_state = scene["robot"].data.root_state_w.cpu().numpy()
    body_state = isaac_body_to_interface(body_state)

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


def walk_in_place(
    count: int, step_state: bool, control_interface: VectSim2Real
) -> bool:
    # step in place every 200ms
    if count % 40 == 0:
        if step_state:
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([0]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([0.1, 0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([3]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([-0.1, -0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
        else:
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([1]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([0.1, -0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
            control_interface.call(
                SimInterface.initiate_footstep,
                mask=None,
                leg=np.repeat(np.array([2]), args_cli.num_envs),
                location_hip=np.repeat(
                    np.asarray([-0.1, 0.1])[None, :], args_cli.num_envs, axis=0
                ),
                duration=np.repeat(np.array([0.2]), args_cli.num_envs),
            )
        return not step_state
    return step_state


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    control_interface = VectSim2Real(
        dt=sim.get_physics_dt(),
        instances=args_cli.num_envs,
        cls=SimInterface,
        debug_logging=False,
    )

    sim_dt = sim.get_physics_dt()
    logger.debug(f"dt: {sim_dt}")
    sim_time = 0.0
    count = 0
    step_state = True

    # Simulate physics
    while simulation_app.is_running():
        # Reset
        if count % 5000 == 0:
            # reset counter
            count = 0
            reset_simulator(sim, scene, control_interface)

        step_state = walk_in_place(count, step_state, control_interface)

        command = np.zeros((args_cli.num_envs, 3), dtype=np.float32)
        command[:, 0] = 0.3
        command[:, 2] = 0.2
        set_torques(command, scene, control_interface)

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
