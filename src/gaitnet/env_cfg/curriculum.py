from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vmdp  # type: ignore
from isaaclab.envs import mdp


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # in the beginning we reward always moving a foot to teach
    # the model when to use/not the no-op action. Once it has
    # learned this, we remove the reward.
    remove_foot_in_swing_reward = CurrTerm(
        func=mdp.modify_reward_weight,  # type: ignore
        params={
            "term_name": "a_foot_in_swing",
            "weight": 0,
            "num_steps": 50,
        },
    )

    terrain_levels = CurrTerm(func=vmdp.terrain_levels_vel)
