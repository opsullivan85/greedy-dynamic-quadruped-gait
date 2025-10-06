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
    from src.gaitnet.env_cfg.footstep_options_manager import FootstepObservationManager

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
        
        # Store reference to actor wrapper for duration retrieval
        self._actor_wrapper = None

    def set_actor_wrapper(self, actor_wrapper):
        """Set the actor wrapper reference for duration retrieval.
        
        Args:
            actor_wrapper: The GaitnetActorWrapper instance
        """
        self._actor_wrapper = actor_wrapper
    
    def _get_option_manager(self) -> "FootstepObservationManager":
        """Get the footstep option manager.

        note that it isn't initilized until after the action term is initialized

        Returns:
            The footstep option manager.
        """
        footstep_option_manager: "FootstepObservationManager" = self._env.observation_manager  # type: ignore
        return footstep_option_manager

    @property
    def action_dim(self) -> int:
        """Dimension of the action term.
        
        Returns the number of action indices we select (k=2 for top-2 selection),
        not the number of available options (17).
        """
        return 2  # We select top-2 action indices

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

    def action_indices_to_actions(self, action_indices: torch.Tensor) -> torch.Tensor:
        """Convert action indices to footstep actions.

        Args:
            action_indices: The action indices from the policy. (num_envs, k) where k=2

        Returns:
            Footstep actions.
            (num_envs, 2, 4) where each action is (leg, x, y, duration)
        """
        # Get the footstep options and actions from the observation manager
        footstep_option_manager: "FootstepObservationManager" = (
            self._get_option_manager()
        )
        all_actions = footstep_option_manager.footstep_actions  # (num_envs, 17, 4)
        
        # action_indices shape: (num_envs, 2)
        # all_actions shape: (num_envs, 17, 4)
        
        # Use proper indexing to select the actions
        batch_size, k = action_indices.shape
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, k)
        
        # Gather the selected actions
        selected_actions = all_actions[batch_indices, action_indices]  # (num_envs, 2, 4)
        
        # Handle no-op logic: if first action is no-op, make second one no-op too
        selected_actions[:, 1, 0] = torch.where(
            selected_actions[:, 0, 0] == NO_STEP, 
            NO_STEP, 
            selected_actions[:, 1, 0]
        )
        # Zero out the x, y, duration of no-op actions for the second action
        selected_actions[:, 1, 1:4] = torch.where(
            selected_actions[:, 1, 0].unsqueeze(1) == NO_STEP,
            torch.zeros_like(selected_actions[:, 1, 1:4]),
            selected_actions[:, 1, 1:4],
        )
        
        return selected_actions  # (num_envs, 2, 4)

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The action indices from the policy (num_envs, k)
        """
        # Store raw actions
        self._raw_actions = actions
        
        # Get durations from the actor wrapper if available
        if self._actor_wrapper is not None and hasattr(self._actor_wrapper, '_cached_durations'):
            durations = self._actor_wrapper.get_durations_for_actions(actions)
            footstep_option_manager = self._get_option_manager()
            footstep_option_manager.set_footstep_actions(actions, durations)
        
        # Convert action indices to footstep actions
        self._processed_actions = self.action_indices_to_actions(actions)
        processed_actions_cpu = self.processed_actions.cpu().numpy()

        # iterate over the second axis of processed_actions
        for i in range(processed_actions_cpu.shape[1]):
            action = processed_actions_cpu[:, i, :]  # (num_envs, 4)
            
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
