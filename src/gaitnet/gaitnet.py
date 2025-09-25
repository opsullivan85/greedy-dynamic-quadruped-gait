from typing import Callable
from rsl_rl.modules import ActorCritic
import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnv
from src.gaitnet.actions.footstep_action import NO_STEP
from src import get_logger
from src.contactnet.contactnet import CostMapGenerator
from src.gaitnet.env_cfg.observations import get_terrain_mask
from src.simulation.cfg.footstep_scanner_constants import idx_to_xy

logger = get_logger()


class FootstepOptionGenerator:
    def __init__(self, env: ManagerBasedRLEnv, num_options: int = 5):
        self.env = env
        self.num_options = num_options
        self.cost_map_generator = CostMapGenerator(device=env.device)

    def get_footstep_options(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate footstep options based on the current environment state.

        Returns:
            torch.Tensor: Footstep options of shape (num_envs, num_options, 3)
                          Each option is represented as (leg_index, x_offset, y_offset)
                          total shape is (num_options, 3)
            torch.Tensor: Corresponding costs of shape (num_envs, num_options)
        """
        cost_maps = self.cost_map_generator.predict(obs)  # (num_envs, 4, H, W)

        # remove options with invalid terrain
        valid_height_range = self.env.cfg.valid_height_range  # type: ignore
        terrain_mask = get_terrain_mask(valid_height_range, obs)  # (num_envs, 4, H, W)
        masked_cost_maps = torch.where(
            terrain_mask, cost_maps, torch.tensor(float("inf"), device=cost_maps.device)
        )

        # remove options for legs in swing state
        # here 18:22 are the contact states for the 4 legs
        contact_states = obs[:, 18:22].bool()  # (num_envs, 4)
        swing_states = ~contact_states  # (num_envs, 4)
        swing_states = swing_states.unsqueeze(-1).unsqueeze(-1)  # (num_envs, 4, 1, 1)
        masked_cost_maps = torch.where(
            swing_states, torch.tensor(float("inf"), device=cost_maps.device), masked_cost_maps
        )

        # get the best options
        flat_cost_map = masked_cost_maps.flatten()
        flat_cost_map = flat_cost_map.reshape(masked_cost_maps.shape[0], -1)  # (num_envs, 4*H*W)
        topk_values, topk_flat_indices = torch.topk(
            flat_cost_map, self.num_options, largest=False, sorted=False
        )  # (num_envs, num_options)

        # unravel indices to (leg, idx, idy)
        topk_indices = torch.unravel_index(
            topk_flat_indices, masked_cost_maps.shape[1:]
        )
        # to (num_envs, num_options, 3)
        topk_indices = torch.stack(topk_indices, dim=-1)

        # convert to (leg, x, y) to (leg, dx, dy)
        topk_pos = idx_to_xy(topk_indices)  # (num_envs, num_options, 3)

        return topk_pos, topk_values


class Gaitnet(nn.Module):
    """
    Value-based network for scoring individual footstep options and predicting swing duration.
    This network learns to evaluate each footstep option independently.
    """

    def __init__(
        self,
        get_footstep_options: Callable[[torch.Tensor,], tuple[torch.Tensor, torch.Tensor]],
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

        # Shared trunk for both outputs
        self.shared_trunk = nn.Sequential(
            nn.Linear(shared_encoder_dim + footstep_encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
        )

        # Value head: outputs reward value
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Duration head: outputs swing duration
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Normalize to [0, 1], scale later to actual duration range
        )

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
            [footstep_costs, torch.full((batch_size, 1), float('inf'), device=device)],
            dim=1,
        )  # (batch, num_options + 1)
        # set invalid options to no-action
        inf_cost_mask = torch.isinf(footstep_costs)  # (batch, num_options)
        footstep_options[inf_cost_mask] = no_action_option.expand_as(footstep_options)[inf_cost_mask]
        
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
            footstep_features = self.no_action_embedding.unsqueeze(0).expand(batch_size, -1).clone()
            footstep_features[non_inf_mask_i] = self.footstep_encoder(footstep_input[non_inf_mask_i])

            # Combine with robot state and compute outputs
            combined = torch.cat([robot_features, footstep_features], dim=-1)
            trunk_features = self.shared_trunk(combined)

            # Get value and duration
            value = self.value_head(trunk_features)  # (batch, 1)
            duration = torch.zeros((batch_size, 1), device=device)  # default to 0 for no-action
            duration[non_inf_mask_i] = self.duration_head(trunk_features[non_inf_mask_i])  # (batch, 1), in [0, 1]

            # Scale duration to actual range
            duration[non_inf_mask_i] = self.min_swing_duration + duration[non_inf_mask_i] * (
                self.max_swing_duration - self.min_swing_duration
            )

            option_values.append(value)
            option_durations.append(duration)

        # Stack all values and durations
        all_values = torch.cat(option_values, dim=-1)  # (batch, num_options + 1)
        all_durations = torch.cat(option_durations, dim=-1)  # (batch, num_options + 1)

        return footstep_options, all_values, all_durations


class GaitNetActorCritic(ActorCritic):
    """
    Actor-Critic network for footstep selection using value-based approach.
    Compatible with RSL-RL.
    """

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        get_footstep_options: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        robot_state_dim=22,
        num_footstep_options=5,
        hidden_dims=[256, 256],
        activation="relu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        **kwargs
    ):
        """
        Initialize the GaitNetActorCritic.
        Args:
            robot_state_dim (int, optional): Dimension of the robot state representation. Defaults to 22.
            num_footstep_options (int, optional): Number of footstep options. Defaults to 5.
            hidden_dims (list, optional): Hidden dimensions for the network. Defaults to [256, 256].
        """
        self.robot_state_dim = robot_state_dim
        self.num_footstep_options = num_footstep_options
        self.hidden_dims = hidden_dims
        self.num_actions = num_actions

        # Distribution for RL compatibility
        self.distribution: torch.distributions.Categorical = None  # type: ignore

        # Call parent constructor
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=hidden_dims,
            critic_hidden_dims=hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs
        )

        # Override the actor with our custom value-based network
        self.actor = Gaitnet(
            get_footstep_options=get_footstep_options,
            robot_state_dim=self.robot_state_dim,
            shared_encoder_dim=hidden_dims[0] // 2,
            footstep_encoder_dim=hidden_dims[0] // 4,
            hidden_dim=hidden_dims[0],
        )

        # Replace the critic with our custom architecture
        self.critic = nn.Sequential(
            nn.Linear(num_critic_obs, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(
                hidden_dims[0],
                hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0],
            ),
            nn.LayerNorm(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1] if len(hidden_dims) > 1 else hidden_dims[0], 1),
        )

        # Cache for lazy evaluation
        self._cached_observations: torch.Tensor = None  # type: ignore
        self._cached_footstep_options: torch.Tensor = None  # type: ignore
        self._cached_action_values: torch.Tensor = None  # type: ignore
        self._cached_swing_durations: torch.Tensor = None  # type: ignore
        self._cached_action_probs: torch.Tensor = None  # type: ignore
        self._cached_selected_indices: torch.Tensor = None  # type: ignore

    def _ensure_forward_pass(self, observations: torch.Tensor):
        """
        Ensure forward pass has been computed for given observations.
        Uses caching to avoid redundant computation.
        """
        # Check if we need to compute or observations have changed
        if (self._cached_observations is None or 
            not torch.equal(self._cached_observations, observations)):
            
            # Perform forward pass
            footstep_options, action_values, swing_durations = self.actor(observations)
            
            # Convert to probabilities with softmax
            action_probs = nn.functional.softmax(action_values, dim=-1)
            
            # Cache results
            self._cached_observations = observations.clone()
            self._cached_footstep_options = footstep_options
            self._cached_action_values = action_values
            self._cached_swing_durations = swing_durations
            self._cached_action_probs = action_probs

    def update_distribution(self, observations: torch.Tensor):
        """
        Update the action distribution based on current observations.
        Uses lazy evaluation - actual computation happens on first access.
        """
        # Ensure forward pass is computed
        self._ensure_forward_pass(observations)
        
        # Create categorical distribution from cached probabilities
        self.distribution = torch.distributions.Categorical(self._cached_action_probs)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Select action based on current observations.

        Args:
            observations: (batch, num_obs) tensor of observations (robot state)

        Returns:
            actions: (batch, 4) tensor with [leg_idx, dx, dy, swing_duration]
                    For no-action: [NO_STEP, 0, 0, 0]
        """
        # Ensure distribution is updated
        self.update_distribution(observations)

        # Sample action index (or take argmax if deterministic)
        if kwargs.get("deterministic", False):
            selected_indices = torch.argmax(self._cached_action_probs, dim=-1)
        else:
            selected_indices = self.distribution.sample()

        # Cache the selected indices for get_actions_log_prob
        self._cached_selected_indices = selected_indices

        # Add swing durations as the 4th dimension
        actions_with_durations = torch.cat(
            [self._cached_footstep_options, self._cached_swing_durations.unsqueeze(-1)], 
            dim=-1
        )  # (batch, num_options + 1, 4)

        # Select the actions based on selected indices
        batch_size = observations.shape[0]
        batch_indices = torch.arange(batch_size, device=observations.device)
        selected_actions = actions_with_durations[batch_indices, selected_indices]

        return selected_actions

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Deterministic action selection for inference/evaluation.
        
        Args:
            observations: (batch, num_obs) tensor of observations
            
        Returns:
            actions: (batch, 4) tensor with [leg_idx, dx, dy, swing_duration]
        """
        actions = self.act(observations, deterministic=True)
        return actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute log probabilities of given actions.
        
        Args:
            actions: (batch, 4) tensor of actions [leg_idx, dx, dy, swing_duration]
        
        Returns:
            log_probs: (batch,) tensor of log probabilities
        """
        if self._cached_selected_indices is not None:
            # Use cached indices from the last act() call (most common case)
            return self.distribution.log_prob(self._cached_selected_indices)
        else:
            # Fallback: infer indices from actions (less reliable due to floating point precision)
            batch_size = actions.shape[0]
            device = actions.device
            
            # Add swing durations to cached footstep options for comparison
            actions_with_durations = torch.cat(
                [self._cached_footstep_options, self._cached_swing_durations.unsqueeze(-1)], 
                dim=-1
            )  # (batch, num_options + 1, 4)
            
            # Find which option matches each action
            inferred_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for b in range(batch_size):
                # Compare actions with all options for this batch element
                differences = torch.abs(actions_with_durations[b] - actions[b]).sum(dim=-1)
                inferred_indices[b] = torch.argmin(differences)
            
            return self.distribution.log_prob(inferred_indices)

    @property
    def action_mean(self):
        """
        Mean of the action distribution.
        For our value-based approach, return the expected action based on probabilities.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        
        # Compute expected action as weighted sum of all options
        batch_size = self._cached_footstep_options.shape[0]
        
        # Add swing durations as the 4th dimension
        actions_with_durations = torch.cat(
            [self._cached_footstep_options, self._cached_swing_durations.unsqueeze(-1)], 
            dim=-1
        )  # (batch, num_options + 1, 4)
        
        # Compute weighted average using probabilities
        probs = self.distribution.probs.unsqueeze(-1)  # (batch, num_options + 1, 1)
        expected_action = (actions_with_durations * probs).sum(dim=1)  # (batch, 4)
        
        return expected_action

    @property 
    def action_std(self):
        """
        Standard deviation of the action distribution.
        Compute std based on the variance of actions weighted by probabilities.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        
        # Get expected action
        mean_action = self.action_mean  # (batch, 4)
        
        # Add swing durations as the 4th dimension
        actions_with_durations = torch.cat(
            [self._cached_footstep_options, self._cached_swing_durations.unsqueeze(-1)], 
            dim=-1
        )  # (batch, num_options + 1, 4)
        
        # Compute variance
        probs = self.distribution.probs.unsqueeze(-1)  # (batch, num_options + 1, 1)
        mean_expanded = mean_action.unsqueeze(1)  # (batch, 1, 4)
        variance = (probs * (actions_with_durations - mean_expanded) ** 2).sum(dim=1)  # (batch, 4)
        
        # Return standard deviation
        return torch.sqrt(variance + 1e-8)  # Add small epsilon for numerical stability

    @property
    def entropy(self):
        """
        Entropy of the action distribution.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution() first.")
        return self.distribution.entropy()

    def reset(self, dones=None):
        """
        Reset the cached values. Called between episodes.
        """
        self._cached_observations = None  # type: ignore
        self._cached_footstep_options = None  # type: ignore
        self._cached_action_values = None  # type: ignore
        self._cached_swing_durations = None  # type: ignore
        self._cached_action_probs = None  # type: ignore
        self._cached_selected_indices = None  # type: ignore
        self.distribution = None  # type: ignore