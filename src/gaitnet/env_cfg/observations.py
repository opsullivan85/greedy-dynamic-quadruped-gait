import torch
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from src.gaitnet.actions.mpc_action import ManagerBasedEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_shape,
)


@generic_io_descriptor(
    units="m", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def foot_position_xy_b(env: ManagerBasedEnv, transform_name: SceneEntityCfg = SceneEntityCfg("foot_transforms")) -> torch.Tensor:
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
    """
    foot_positions_b = env.scene[transform_name.name].data.target_pos_source[:, :, :2]
    return foot_positions_b


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_pos_z = ObsTerm(
            func=mdp.base_pos_z, noise=Unoise(n_min=-0.01, n_max=0.01)  # just made up this number
        )

        foot_position_xy_b = ObsTerm(
            func=foot_position_xy_b, noise=Unoise(n_min=-0.01, n_max=0.01)  # roughly 1/20 of the typical range
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01)  # roughly 1/10 of the max control input
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.02, n_max=0.02)  # roughly 1/10 of the max control input
        )

        def __post_init__(self):
            self.enable_corruption = False
            # self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
