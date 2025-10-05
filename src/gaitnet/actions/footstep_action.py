"""Action for running footstep controller"""

from dataclasses import MISSING
from typing import Any, Sequence
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import torch

from src import sim2real
from src.util import VectorPool
from src.simulation.util import controls_to_joint_efforts
from src.gaitnet.env_cfg.footstep_options_manager import FootstepObservationManager
import numpy as np
import src.constants as const
from src import get_logger

logger = get_logger()

NO_STEP = -1  # special value for no step

class FSCActionTerm(ActionTerm):
    """Footstep Controller (FSC) Action Term"""

    def __init__(self, cfg: "FSCActionCfg", env: ManagerBasedEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """
        super().__init__(cfg, env)
        # for type hinting
        self.cfg: "FSCActionCfg"
        self.env_cfg = env.cfg
        self._asset: Articulation  # type: ignore
        self.footstep_option_manager: FootstepObservationManager = env.observation_manager  # type: ignore
        assert isinstance(
            self.footstep_option_manager, FootstepObservationManager
        ), "FSCActionTerm requires FootstepObservationManager to be used as the observation manager"

        self._raw_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device
        )
        self._processed_actions = self._raw_actions

    @property
    def action_dim(self) -> int:
        """Dimension of the action term.
        """
        return const.gait_net.num_footstep_options * 4  # 4 legs

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    @staticmethod
    def footstep_kwargs(processed_actions: np.ndarray) -> dict[str, np.ndarray]:
        """Generate the kwargs for the footstep initiation call.

        Args:
            processed_actions: The processed actions (on cpu).

        Returns:
            The kwargs for the footstep initiation call.
        """
        # convert legs from [FL, FR, RL, RR] order to [FR, FL, RR, RL] order
        # this is needed to match the order expected by the Sim2RealInterface
        legs = processed_actions[:, 0].astype(np.int32)
        legs_processed = np.copy(legs)
        legs_processed[legs == 0] = 1  # FL -> FR
        legs_processed[legs == 1] = 0  # FR -> FL
        legs_processed[legs == 2] = 3  # RL -> RR
        legs_processed[legs == 3] = 2  # RR -> RL
        return {
            "leg": legs_processed,
            "location_hip": processed_actions[:, 1:3],
            "duration": processed_actions[:, 3],
        }
    
    def action_probs_to_actions(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Convert the action probabilities to some number of footstep actions.

        Args:
            action_probs: The raw action probabilities. (num_envs, action_dim)

        Returns:
            A list of footstep actions.
            (num_envs, 2, 4) where each action is (leg, x, y, duration)
            the second index is the two actions chosen.
        """
        # get the best action each leg can take
        # also consider the no-op action's value (last index)
        # considering these 5 actions and values, pick the 2 best actions
        #     because of how we considered above, these are garanteed to pretain to different legs (or no-op)
        # in the case that the no-op is the best of the two, replace the second best with no-op
        # return the two actions as a list [(num_envs, 4), (num_envs, 4)]
        # where each action is (leg, x, y, duration)

        # TODO: make sure this behemoth of tensor operations is correct

        # set of actions action_probs correspond to
        all_actions = self.footstep_option_manager.footstep_actions
        # we are assuming no-op is the last action
        no_op_prob = action_probs[:, -1]
        no_op_index = torch.full((self.num_envs, 1), -1, device=self.device, dtype=torch.int64)
        # reshape to (num_envs, num_legs, num_options)
        leg_action_probs = action_probs[:, :-1].view(  # (num_envs, num_legs, num_options)
            self.num_envs, const.robot.num_legs, const.gait_net.num_footstep_options
        )

        # get the best action_probs and indices for each leg
        best_leg_probs, best_leg_indices = torch.max(leg_action_probs, dim=2)
        # add in the no-op action
        best_leg_probs = torch.cat((best_leg_probs, no_op_prob.unsqueeze(1)), dim=1)  # (num_envs, num_legs + 1)
        best_leg_indices = torch.cat((best_leg_indices, no_op_index), dim=1)  # (num_envs, num_legs + 1)

        # unravel the best_leg_indices. This corresponds to indices of the best action for each leg in all_actions
        best_leg_action_indices = torch.unravel_index(
            best_leg_indices, (const.robot.num_legs, const.gait_net.num_footstep_options)
        )[1]  # (num_envs, num_legs)

        # get the top 2 actions
        # at this point we don't care about the probabilities anymore
        _, top2_indices = torch.topk(best_leg_probs, k=2, dim=1, sorted=True)  # (num_envs, 2)
        top2_action_indices = torch.gather(best_leg_action_indices, 1, top2_indices)  # (num_envs, 2)
        top2_actions = all_actions[top2_action_indices]  # (num_envs, 2, 4)

        # if the leg of the top action is no-op (-1), replace the second action no-op as well
        top2_actions[:, 1, 0] = torch.where(
            top2_actions[:, 0, 0] == NO_STEP, NO_STEP, top2_actions[:, 1, 0]
        )
        # zero out the x, y, duration of no-op actions
        # we only need to do this for the second action, because the first action would never be coerced to no-op
        top2_actions[:, 1, 1:4] = torch.where(
            top2_actions[:, 1, 0].unsqueeze(1) == NO_STEP,
            torch.zeros_like(top2_actions[:, 1, 1:4]),
            top2_actions[:, 1, 1:4],
        )
        return top2_actions  # (num_envs, 2, 4)

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        # initiate the footstep if specified by the actions
        self._raw_actions = actions
        self._processed_actions = self.action_probs_to_actions(actions)
        processed_actions_cpu = self.processed_actions.cpu().numpy()

        # mask out invalid steps
        mask = processed_actions_cpu[:, 0] != NO_STEP
        footstep_parameters = self.footstep_kwargs(processed_actions_cpu)

        # logger.info(f"actions: {actions[0].cpu().numpy().tolist()}")

        # initiate the footsteps
        robot_controllers: VectorPool[sim2real.Sim2RealInterface] = self.env_cfg.robot_controllers  # type: ignore
        robot_controllers.call(
            function=sim2real.Sim2RealInterface.initiate_footstep,
            mask=mask,
            **footstep_parameters,
        )

    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        # we don't do anything here
        pass
    
    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the action term.

        Args:
            env_ids: The environment IDs to reset.
        """
        super().reset(env_ids)

@configclass
class FSCActionCfg(ActionTermCfg):
    """Configuration for the Footstep Controller (FSC) Action Term"""

    class_type: type[ActionTerm] = FSCActionTerm
