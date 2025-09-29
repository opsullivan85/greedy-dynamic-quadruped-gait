import numpy as np
import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from src.gaitnet.actions.mpc_action import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_shape,
)
import src.simulation.cfg.footstep_scanner_constants as fs
from src.simulation.cfg.footstep_scanner import real_grid_size
import torch.nn.functional as F
from src.util.vectorpool import VectorPool
from src.sim2real.abstractinterface import Sim2RealInterface
from src import get_logger

logger = get_logger()


@generic_io_descriptor(
    units="m",
    axes=["X", "Y", "Z"],
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def foot_position_xy_b(
    env: ManagerBasedEnv,
    transform_name: SceneEntityCfg = SceneEntityCfg("foot_transforms"),
    flatten: bool = False,
) -> torch.Tensor:
    """Get foot xy positions in the base frame.
        Assumes transform_name sensor exists in the scene, ex:

    .. code-block:: python

        foot_transforms = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/trunk",
            target_frames=[
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FL_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/FR_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RL_foot"),
                FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/RR_foot"),
            ],
            debug_vis=False,
        )

    Args:
        env: The environment instance.
        transform_name: The name of the FrameTransformer sensor in the scene.
        flatten: Whether to flatten the output to (N, 8) instead of (N, 4, 2).
    """
    foot_positions_b = env.scene[transform_name.name].data.target_pos_source[:, :, :2]
    if flatten:
        foot_positions_b = foot_positions_b.reshape(foot_positions_b.shape[0], -1)
    return foot_positions_b


@generic_io_descriptor(
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def contact_state_sensors(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Get the contact state from a sensor."""
    contact_forces = env.scene[sensor_cfg.name].data.net_forces_w
    contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    return contacts


@generic_io_descriptor(
    observation_type="RootState",
    on_inspect=[record_shape, record_dtype],
)
def contact_state_controller(env: ManagerBasedEnv) -> torch.Tensor:
    """Get the contact state from the controller."""
    controllers: VectorPool[Sim2RealInterface] = env.cfg.robot_controllers  # type: ignore

    # controllers won't be initilized when the on_inspect is called, in that case return fake data.
    # this should only happen once in the beginning of training when the env is created
    if controllers is None:
        logger.warning(
            "Controllers are not initialized, returning fake data. Normal 1 time only."
        )
        return torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.bool)

    contacts: np.ndarray = controllers.call(
        Sim2RealInterface.get_contact_state, mask=None
    )
    # FR, FL, RR, RL to FL, FR, RL, RR
    contacts = contacts[:, [1, 0, 3, 2]]
    # logger.info(f"contact: {contacts[0]}")
    contacts_gpu = torch.from_numpy(contacts).to(env.device)
    return contacts_gpu


def cspace_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    assumes all sensor_cfgs point to RayCaster sensors with the same grid size

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    height_scan = mdp.height_scan(env=env, sensor_cfg=sensor_cfg, offset=offset)
    height_scan = height_scan.reshape((-1, *real_grid_size))

    # pool the height scan to match the desired grid size
    kernel_size = 2 * fs.upscale_factor - 1
    stride = fs.upscale_factor
    heights_pooled = F.max_pool2d(height_scan, kernel_size=kernel_size, stride=stride)
    heights_pooled = heights_pooled.reshape(heights_pooled.shape[0], -1)
    return heights_pooled


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        foot_position_xy_b = ObsTerm(
            func=foot_position_xy_b,
            noise=Unoise(n_min=-0.01, n_max=0.01),  # roughly 1/20 of the typical range
            params={"flatten": True},
        )

        base_pos_z = ObsTerm(
            func=mdp.base_pos_z,
            noise=Unoise(n_min=-0.01, n_max=0.01),  # just made up this number
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(
                n_min=-0.01, n_max=0.01
            ),  # roughly 1/10 of the max control input
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(
                n_min=-0.02, n_max=0.02
            ),  # roughly 1/10 of the max control input
        )

        control = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        # contact_state_sensor = ObsTerm(
        #     func=contact_state_sensors,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        # )

        contact_state_controller = ObsTerm(
            func=contact_state_controller,
            params={},
        )

        FR_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("FR_foot_scanner")},
        )

        FL_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("FL_foot_scanner")},
        )

        RL_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("RL_foot_scanner")},
        )

        RR_foot_scanner = ObsTerm(
            func=cspace_height_scan,
            params={"sensor_cfg": SceneEntityCfg("RR_foot_scanner")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            # self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


def get_terrain_mask(
    valid_height_range: tuple[float, float], obs: torch.Tensor
) -> torch.Tensor:
    """Get a mask for the terrain observations.

    0 indicates invalid terrain (too high or too low)
    1 indicates valid terrain
    """
    terrain_terms = fs.grid_size[0] * fs.grid_size[1] * 4
    terrain_obs = obs[:, -terrain_terms:]
    # reshape to (N, 4, H, W)
    terrain_obs = terrain_obs.reshape(
        terrain_obs.shape[0], 4, fs.grid_size[0], fs.grid_size[1]
    )
    # mask out values outside of allowed height range
    max_height, min_height = valid_height_range
    terrain_mask = (terrain_obs > min_height) & (terrain_obs < max_height)
    return terrain_mask
