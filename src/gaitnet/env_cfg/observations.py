import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from src.gaitnet.actions.mpc_action import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_shape,
)


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
def contact_state(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Get the contact state from a sensor."""
    contact_forces = env.scene[sensor_cfg.name].data.net_forces_w
    contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    return contacts


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

        contact_state = ObsTerm(
            func=contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            # self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class TerrainCfg(ObsGroup):
        """Observations for terrain group."""

        FR_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FR_foot_scanner")},
        )
        FL_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("FL_foot_scanner")},
        )
        RL_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RL_foot_scanner")},
        )
        RR_foot_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("RR_foot_scanner")},
        )

    terrain: TerrainCfg = TerrainCfg()
