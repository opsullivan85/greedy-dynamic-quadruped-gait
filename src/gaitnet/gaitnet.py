from typing import Callable
from rsl_rl.modules import ActorCritic
import torch
import torch.nn as nn
from skrl.models.torch import Model, DeterministicMixin
from skrl.agents.torch.dqn import DDQN

from isaaclab.envs import ManagerBasedRLEnv
from src.gaitnet.actions.footstep_action import NO_STEP
from src import get_logger
from src.contactnet.contactnet import CostMapGenerator
from src.gaitnet.env_cfg.observations import get_terrain_mask
from src.simulation.cfg.footstep_scanner_constants import idx_to_xy

from src.contactnet.debug import view_footstep_cost_map

logger = get_logger()


class FootstepOptionGenerator:
    def __init__(self, env: ManagerBasedRLEnv, num_options: int = 5):
        self.env = env
        self.num_options = num_options
        self.cost_map_generator = CostMapGenerator(device=env.device)

    def _filter_cost_map(
        self, cost_map: torch.Tensor, obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Filter the cost map to remove invalid footstep options based on terrain and robot state.

        Args:
            cost_map: (num_envs, 4, H, W) raw cost maps for each leg
            obs: (num_envs, obs_dim) observation tensor containing robot state

        Returns:
            filtered_cost_map: (num_envs, 4, H, W) cost maps with invalid options set to inf
        """
        # remove options with invalid terrain
        valid_height_range = self.env.cfg.valid_height_range  # type: ignore
        terrain_mask = get_terrain_mask(valid_height_range, obs)  # (num_envs, 4, H, W)
        masked_cost_maps = torch.where(
            terrain_mask, cost_map, torch.tensor(float("inf"), device=cost_map.device)
        )

        # remove options for legs in swing state
        # here 18:22 are the contact states for the 4 legs
        contact_states = obs[:, 18:22].bool()  # (num_envs, 4)
        swing_states = ~contact_states  # (num_envs, 4)
        masked_cost_maps = torch.where(
            swing_states.unsqueeze(-1).unsqueeze(-1),
            torch.tensor(float("inf"), device=cost_map.device),
            masked_cost_maps,
        )

        # require atleast 1 leg to be in contact
        # by setting all costs to inf if only 1 leg is in contact
        num_legs_in_contact = contact_states.sum(dim=1)  # (num_envs,)
        only_one_contact = num_legs_in_contact <= 2  # (num_envs,)
        masked_cost_maps = torch.where(
            only_one_contact.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
            torch.tensor(float("inf"), device=cost_map.device),
            masked_cost_maps,
        )

        return masked_cost_maps

    @staticmethod
    def _overall_best_options(
        cost_map: torch.Tensor, num_options: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the overall best footstep options across all legs.

        Args:
            cost_map: (num_envs, 4, H, W) filtered cost maps for each leg

        Returns:
            best_options: (num_envs, num_options, 3) best options as (leg_idx, dx, dy)
            best_values: (num_envs, num_options) corresponding costs
        """
        # get the best options
        flat_cost_map = cost_map.flatten()
        flat_cost_map = flat_cost_map.reshape(
            cost_map.shape[0], -1
        )  # (num_envs, 4*H*W)
        topk_values, topk_flat_indices = torch.topk(
            flat_cost_map, num_options, largest=False, sorted=False
        )  # (num_envs, num_options)

        # unravel indices to (leg, idx, idy)
        topk_indices = torch.unravel_index(topk_flat_indices, cost_map.shape[1:])
        # to (num_envs, num_options, 3)
        topk_indices = torch.stack(topk_indices, dim=-1)

        # convert to (leg, x, y) to (leg, dx, dy)
        topk_pos = idx_to_xy(topk_indices)  # (num_envs, num_options, 3)

        return topk_pos, topk_values

    @staticmethod
    def _best_options_per_leg(
        cost_map: torch.Tensor, num_options: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the best footstep options for each leg.

        Args:
            cost_map: (num_envs, 4, H, W) filtered cost maps for each leg

        Returns:
            best_options: (num_envs, num_options, 3) best options as (leg_idx, dx, dy)
            best_values: (num_envs, num_options) corresponding costs
        """

        best_options = []
        best_values = []
        for leg in range(4):
            leg_cost_map = cost_map[:, leg, :, :].unsqueeze(1)  # (num_envs, 1, H, W)
            topk_pos, topk_values = FootstepOptionGenerator._overall_best_options(
                leg_cost_map, num_options // 4
            )
            # manually set leg index
            topk_pos[:, :, 0] = leg
            best_options.append(topk_pos)
            best_values.append(topk_values)

        best_options = torch.cat(best_options, dim=1)  # (num_envs, num_options, 3)
        best_values = torch.cat(best_values, dim=1)  # (num_envs, num_options)

        return best_options, best_values

    def get_footstep_options(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate footstep options based on the current environment state.

        Returns:
            torch.Tensor: Footstep options of shape (num_envs, num_options, 3)
                          Each option is represented as (leg_index, x_offset, y_offset)
                          total shape is (num_options, 3)
            torch.Tensor: Corresponding costs of shape (num_envs, num_options)
        """
        with torch.inference_mode():
            cost_maps = self.cost_map_generator.predict(obs)  # (num_envs, 4, H, W)
            # switch from (FL, FR, RL, RR) to (FR, FL, RR, RL)
            cost_maps = cost_maps[:, [1, 0, 3, 2], :, :]

        masked_cost_maps = self._filter_cost_map(cost_maps, obs)  # (num_envs, 4, H, W)

        # best_options, best_values = self._overall_best_options(masked_cost_maps, self.num_options)
        best_options, best_values = self._best_options_per_leg(
            masked_cost_maps, self.num_options
        )
        return best_options, best_values


class Gaitnet(nn.Module):
    """
    Value-based network for scoring individual footstep options and predicting swing duration.
    This network learns to evaluate each footstep option independently.
    """

    def __init__(
        self,
        get_footstep_options: Callable[
            [
                torch.Tensor,
            ],
            tuple[torch.Tensor, torch.Tensor],
        ],
        robot_state_dim: int = 22,
        shared_encoder_dim: int = 128,
        footstep_encoder_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.get_footstep_options = get_footstep_options

        self.robot_state_dim = robot_state_dim

        # Shared encoder for robot state
        self.robot_state_encoder = nn.Sequential(
            nn.Linear(robot_state_dim, shared_encoder_dim),
            nn.LayerNorm(shared_encoder_dim),
            nn.ReLU(),
            nn.Linear(shared_encoder_dim, shared_encoder_dim),
            nn.LayerNorm(shared_encoder_dim),
            nn.ReLU(),
        )
        logger.info(f"robot state encoder\n{self.robot_state_encoder}")

        # Footstep option encoder (processes each option)
        # Input: [leg_idx_one_hot(4), x_offset, y_offset, cost]
        self.footstep_encoder = nn.Sequential(
            nn.Linear(
                4 + 2 + 1, footstep_encoder_dim
            ),  # 4 for one-hot leg, 2 for x,y, 1 for cost
            nn.LayerNorm(footstep_encoder_dim),
            nn.ReLU(),
            nn.Linear(footstep_encoder_dim, footstep_encoder_dim),
            nn.LayerNorm(footstep_encoder_dim),
            nn.ReLU(),
        )
        logger.info(f"footstep encoder\n{self.footstep_encoder}")

        # Shared trunk for both outputs
        self.shared_trunk = nn.Sequential(
            nn.Linear(shared_encoder_dim + footstep_encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )
        logger.info(f"shared trunk\n{self.shared_trunk}")

        # Value head: outputs reward value
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        logger.info(f"value head\n{self.value_head}")

        # Duration head: outputs swing duration
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Normalize to [0, 1], scale later to actual duration range
        )
        logger.info(f"duration head\n{self.duration_head}")

        # Special embedding for "no action" option
        self.no_action_embedding = nn.Parameter(torch.randn(footstep_encoder_dim))

        # Duration range parameters (in seconds)
        self.min_swing_duration = 0.1
        self.max_swing_duration = 0.3

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute values and swing durations for all footstep options.

        Args:
            obs: (batch_size, 122) robot state information

        Returns:
            footstep_options: (batch_size, num_options + 1, 3), including no-action option: [NO_STEP, 0, 0, 0]
                last idx is (leg, dx, dy)
            values: (batch_size, num_options + 1) values for each option + no-action
            durations: (batch_size, num_options + 1) swing durations for each option
                no-action duration is 0
        """
        robot_state = obs[:, : self.robot_state_dim]
        # (batch_size, num_options, 3)
        # [leg_idx, dx, dy]
        batch_size = robot_state.shape[0]
        device = robot_state.device
        # add numeric representation for no-action to footstep options
        no_action_option = torch.zeros((batch_size, 1, 3), device=device)
        no_action_option[:, :, 0] = NO_STEP  # leg_idx = NO_STEP

        # prepare footstep options
        footstep_options, footstep_costs = self.get_footstep_options(obs)
        # add no-action option to the list
        footstep_options = torch.cat(
            [footstep_options, no_action_option],
            dim=1,
        )  # (batch, num_options + 1, 3)
        footstep_costs = torch.cat(
            [footstep_costs, torch.full((batch_size, 1), float("inf"), device=device)],
            dim=1,
        )  # (batch, num_options + 1)
        # set invalid options to no-action
        inf_cost_mask = torch.isinf(footstep_costs)  # (batch, num_options)
        footstep_options[inf_cost_mask] = no_action_option.expand_as(footstep_options)[
            inf_cost_mask
        ]

        # Encode robot state (shared for all options)
        robot_features = self.robot_state_encoder(
            robot_state
        )  # (batch, shared_encoder_dim)

        # Process each footstep option
        option_values = []
        option_durations = []

        num_options = footstep_options.shape[1]

        for i in range(num_options):
            # Extract option
            option = footstep_options[:, i, :]  # (batch, 3)
            non_inf_mask_i = ~inf_cost_mask[:, i]

            # Convert leg index to one-hot
            leg_idx = option[:, 0].long()
            leg_one_hot = torch.zeros((batch_size, 4), device=device)
            leg_one_hot[non_inf_mask_i] = nn.functional.one_hot(
                leg_idx[non_inf_mask_i], num_classes=4
            ).float()  # (batch, 4)

            dx = option[:, 1]  # (batch,)
            dy = option[:, 2]  # (batch,)

            cost = footstep_costs[:, i]  # (batch,)

            # Combine footstep features
            footstep_input = torch.cat(
                [leg_one_hot, dx.unsqueeze(-1), dy.unsqueeze(-1), cost.unsqueeze(-1)],
                dim=-1,
            )  # (batch, 4+2+1)

            # Encode footstep
            # default to no-action embedding, calculate embedding for valid options
            footstep_features = (
                self.no_action_embedding.unsqueeze(0).expand(batch_size, -1).clone()
            )
            footstep_features[non_inf_mask_i] = self.footstep_encoder(
                footstep_input[non_inf_mask_i]
            )

            # Combine with robot state and compute outputs
            combined = torch.cat([robot_features, footstep_features], dim=-1)
            trunk_features = self.shared_trunk(combined)

            # Get value and duration
            value = self.value_head(trunk_features)  # (batch, 1)
            duration = torch.zeros(
                (batch_size, 1), device=device
            )  # default to 0 for no-action
            duration[non_inf_mask_i] = self.duration_head(
                trunk_features[non_inf_mask_i]
            )  # (batch, 1), in [0, 1]

            # Scale duration to actual range
            duration[non_inf_mask_i] = self.min_swing_duration + duration[
                non_inf_mask_i
            ] * (self.max_swing_duration - self.min_swing_duration)

            option_values.append(value)
            option_durations.append(duration)

        # Stack all values and durations
        all_values = torch.cat(option_values, dim=-1)  # (batch, num_options + 1)
        all_durations = torch.cat(option_durations, dim=-1)  # (batch, num_options + 1)

        # logger.info(f"cost: {footstep_costs[0].detach().cpu().numpy().tolist()}")
        # # if all of footstep_costs[0] are inf
        # if torch.all(torch.isinf(footstep_costs[0])):
        #     pass
        # logger.info(f"value: {all_values[0].detach().cpu().numpy().tolist()}")

        return footstep_options, all_values, all_durations


class GaitNetDDQN(Model, DeterministicMixin):
    """
    DDQN-compatible wrapper for Gaitnet that outputs Q-values for discrete footstep actions.
    This wrapper adapts the existing Gaitnet architecture to work with skrl's DDQN agent.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        get_footstep_options: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        robot_state_dim=22,
        shared_encoder_dim=128,
        footstep_encoder_dim=64,
        hidden_dim=256,
        device=None,
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)

        self.robot_state_dim = robot_state_dim

        # Core Gaitnet for Q-value computation - this is the only thing we actually need!
        self.gaitnet = Gaitnet(
            get_footstep_options=get_footstep_options,
            robot_state_dim=robot_state_dim,
            shared_encoder_dim=shared_encoder_dim,
            footstep_encoder_dim=footstep_encoder_dim,
            hidden_dim=hidden_dim,
        )

    def compute(self, inputs, role=""):
        """
        Compute Q-values for all possible footstep actions.

        Args:
            inputs: Dictionary containing "states" key with observations
            role: Role identifier (not used in this implementation)

        Returns:
            Q-values for each discrete action option
        """
        states = inputs["states"]

        # Get footstep options and their Q-values from Gaitnet
        # We DON'T throw away footstep_options and swing_durations - they're used implicitly
        # The Gaitnet already handles the mapping from discrete actions to continuous actions
        # through its forward pass which considers all possible footstep options
        footstep_options, action_values, swing_durations = self.gaitnet(states)

        # Return Q-values for DDQN (these represent the value of each discrete action)
        return action_values


class GaitNetDDQNAgent(DDQN):
    """
    Custom DDQN agent that handles discrete-to-continuous action conversion for GaitNet.
    """
    
    def __init__(self, footstep_generator, robot_state_dim=71, **kwargs):
        super().__init__(**kwargs)
        self.footstep_generator = footstep_generator
        self.robot_state_dim = robot_state_dim
        
    def act(self, states, timestep, timesteps):
        """Override act to convert discrete actions to continuous."""
        # Get discrete actions from parent DDQN
        discrete_actions = super().act(states, timestep, timesteps)
        
        # Convert to continuous actions
        continuous_actions = self._discrete_to_continuous(discrete_actions, states)
        
        return continuous_actions
    
    def _discrete_to_continuous(self, discrete_actions, states):
        """Convert discrete action indices to continuous footstep actions."""
        # Use the footstep generator to get options for current state
        # This is more direct than going through the network
        batch_size = discrete_actions.shape[0]
        device = discrete_actions.device
        
        # Generate footstep options for the current states  
        footstep_options = []
        swing_durations = []
        
        for i in range(batch_size):
            # Get robot state for this batch item
            robot_state = states[i, :self.robot_state_dim]
            
            # Generate footstep options for this state
            options = self.footstep_generator.get_footstep_options(robot_state.cpu().numpy())
            
            # Convert to tensors
            options_tensor = torch.tensor(options['footstep_options'], device=device, dtype=torch.float32)
            durations_tensor = torch.tensor(options['swing_durations'], device=device, dtype=torch.float32)
            
            footstep_options.append(options_tensor)
            swing_durations.append(durations_tensor)
            
        footstep_options = torch.stack(footstep_options)  # [batch_size, num_actions, 4, 3]
        swing_durations = torch.stack(swing_durations)    # [batch_size, num_actions]
        
        # Select the chosen footstep options based on discrete actions
        selected_footsteps = torch.zeros((batch_size, 4, 3), device=device)
        selected_durations = torch.zeros((batch_size,), device=device)
        
        for b in range(batch_size):
            action_idx = discrete_actions[b].long().item()
            if action_idx < footstep_options.shape[1]:
                selected_footsteps[b] = footstep_options[b, action_idx]
                selected_durations[b] = swing_durations[b, action_idx]
            # else: keep zero values (no-action)
        
        # Flatten footsteps to [batch_size, 12] and concatenate with durations to get [batch_size, 13]
        continuous_actions = torch.cat([
            selected_footsteps.view(batch_size, -1),  # flatten to [batch_size, 12]
            selected_durations.unsqueeze(-1)          # add duration as [batch_size, 1]
        ], dim=-1)  # final shape: [batch_size, 13]
        
        return continuous_actions
