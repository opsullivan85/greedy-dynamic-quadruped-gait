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
import numpy as np
import src.constants as const
from src import get_logger

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.gaitnet.components.gaitnet_observation_manager import GaitNetObservationManager

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

        self._raw_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device
        )
        self._processed_actions = self._raw_actions
    
    def _get_option_manager(self) -> "GaitNetObservationManager":
        """Get the footstep option manager.

        note that it isn't initilized until after the action term is initialized

        Returns:
            The footstep option manager.
        """
        footstep_option_manager: "GaitNetObservationManager" = self._env.observation_manager  # type: ignore
        return footstep_option_manager

    @property
    def action_dim(self) -> int:
        """Dimension of the action term.
        
        Returns 2: action index (0-16) and duration value.
        """
        return 2  # Action index + duration

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

    def action_indices_to_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert actions (index + duration) to footstep actions.

        Args:
            actions: The actions from the policy. (num_envs, 2) where:
                     - Column 0: action index (0-16)
                     - Column 1: duration value

        Returns:
            Footstep actions.
            (num_envs, 4) where each action is (leg, x, y, duration)
        """
        # Extract action indices and durations
        action_indices = actions[:, 0].long()  # (num_envs,)
        durations = actions[:, 1]  # (num_envs,)
        
        # Get the footstep options from the observation manager
        footstep_option_manager: "GaitNetObservationManager" = (
            self._get_option_manager()
        )
        all_options = footstep_option_manager.footstep_options  # (num_envs, 17, 4) - last column is cost
        
        # Use proper indexing to select the options (leg, x, y, cost)
        batch_size = action_indices.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device)
        
        # Gather the selected options (leg, x, y, cost)
        selected_options = all_options[batch_indices, action_indices]  # (num_envs, 4)
        
        # Replace the cost (column 3) with the duration from the policy
        selected_actions = selected_options.clone()
        selected_actions[:, 3] = durations
        
        return selected_actions  # (num_envs, 4) - (leg, x, y, duration)

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions from the policy (num_envs, 2) where:
                     - Column 0: action index (0-16)
                     - Column 1: duration value
        """
        # Store raw actions
        self._raw_actions = actions
        
        # Convert actions (index + duration) to footstep actions (leg, x, y, duration)
        self._processed_actions = self.action_indices_to_actions(actions)
        processed_actions_cpu = self.processed_actions.cpu().numpy()

        # processed_actions is now (num_envs, 4) - single action per env
        action = processed_actions_cpu  # (num_envs, 4)

        # ablate swing duration if specified
        if const.experiments.ablate_swing_duration:
            action[:, 3] = const.experiments.constant_swing_duration
        
        # mask out invalid steps
        mask = action[:, 0] != NO_STEP
        footstep_parameters = self.footstep_kwargs(action)

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
