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
        self._asset: Articulation  # type: ignore

        # setup parallel robot controllers
        if self.cfg.robot_controllers is None:
            raise ValueError("robot_controllers must be set in the cfg before initialization.")
        self.robot_controllers = self.cfg.robot_controllers

        self._raw_actions = torch.zeros(
            (self.num_envs, self.action_dim), device=self.device
        )
        self._processed_actions = self._raw_actions

    def __del__(self):
        super().__del__()
        del self.robot_controllers

    @property
    def action_dim(self) -> int:
        """Dimension of the action term.
        
        [:, 0] = leg index (0-3) or no step (-1)
        [:, 1:3] = x, y location relative to hip
        [:, 3] = duration of the step in seconds
        """
        return 4

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
        return {
            "leg": processed_actions[:, 0].astype(np.int32),
            "location_hip": processed_actions[:, 1:3],
            "duration": processed_actions[:, 3],
        }

    def process_actions(self, actions: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        # initiate the footstep if specified by the actions
        self._raw_actions = actions
        self._processed_actions = self._raw_actions  # no processing needed
        processed_actions_cpu = self.processed_actions.cpu().numpy()

        # mask out invalid steps
        mask = processed_actions_cpu[:, 0] != NO_STEP
        footstep_parameters = self.footstep_kwargs(processed_actions_cpu)

        # initiate the footsteps
        self.robot_controllers.call(
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

    robot_controllers: VectorPool[sim2real.Sim2RealInterface] | None = None
    """Pre-initialized robot controllers to use.
    Needs to be set to non-None value before initializing the action term.
    """

    footstep_duration: float = 0.2
    """Duration of each footstep in seconds."""
