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
        
        Returns a single integer representing one action index (0-16).
        """
        return 1  # We select 1 action index

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
            action_indices: The action indices from the policy. (num_envs,) or (num_envs, 1) with integer values 0-16

        Returns:
            Footstep actions.
            (num_envs, 4) where each action is (leg, x, y, duration)
        """
        # Get the footstep options and actions from the observation manager
        footstep_option_manager: "FootstepObservationManager" = (
            self._get_option_manager()
        )
        all_actions = footstep_option_manager.footstep_actions  # (num_envs, 17, 4)
        
        # Flatten action_indices if it has shape (num_envs, 1)
        if action_indices.dim() > 1:
            action_indices = action_indices.squeeze(-1)
        
        # action_indices shape: (num_envs,)
        # all_actions shape: (num_envs, 17, 4)
        
        # Use proper indexing to select the actions
        batch_size = action_indices.shape[0]
        batch_indices = torch.arange(batch_size, device=self.device)
        
        # Gather the selected actions
        selected_actions = all_actions[batch_indices, action_indices]  # (num_envs, 4)
        
        return selected_actions  # (num_envs, 4)

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The action indices from the policy (num_envs,) or (num_envs, 1)
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

        # processed_actions is now (num_envs, 4) - single action per env
        action = processed_actions_cpu  # (num_envs, 4)
        
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
