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

    def get_footstep_options(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate footstep options based on the current environment state.

        Returns:
            torch.Tensor: Footstep options of shape (num_envs, num_options, 3)
                          Each option is represented as (leg_index, x_offset, y_offset)
                          total shape is (num_options, 3)
            torch.Tensor: Corresponding costs of shape (num_envs, num_options)
        """
        cost_maps = self.cost_map_generator.predict(self.env)  # (num_envs, 4, H, W)
        terrain_mask = get_terrain_mask(self.env)  # (num_envs, 4, H, W)

        # get the argmin of the cost maps where terrain is valid (1)
        masked_cost_maps = torch.where(
            terrain_mask, cost_maps, torch.tensor(float("inf"), device=cost_maps.device)
        )

        flat_cost_map = masked_cost_maps.flatten()
        topk_values, topk_flat_indices = torch.topk(
            flat_cost_map, self.num_options, largest=False, sorted=False
        )  # (num_envs, num_options)

        # unravel indices to (leg, idx, idy)
        topk_indices = torch.unravel_index(
            topk_flat_indices, masked_cost_maps.shape[1:]
        )
        # to (num_envs, num_options, 3)
        topk_indices = torch.stack(topk_indices, dim=-1).permute(1, 0, 2)

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
        get_footstep_options: Callable[..., tuple[torch.Tensor, torch.Tensor]],
        robot_state_dim: int = 22,
        shared_encoder_dim: int = 128,
        footstep_encoder_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.get_footstep_options = get_footstep_options

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
        robot_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute values and swing durations for all footstep options.

        Args:
            robot_state: (batch_size, 22) robot state information

        Returns:
            footstep_options: (batch_size, num_options + 1, 3), including no-action option: [NO_STEP, 0, 0, 0]
            values: (batch_size, num_options + 1) values for each option + no-action
            durations: (batch_size, num_options + 1) swing durations for each option
        """
        footstep_options, footstep_costs = self.get_footstep_options()
        # (batch_size, num_options, 3)
        # [leg_idx, dx, dy]
        batch_size = robot_state.shape[0]
        device = robot_state.device

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

            # Convert leg index to one-hot
            leg_idx = option[:, 0].long()
            leg_one_hot = nn.functional.one_hot(
                leg_idx, num_classes=4
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
            footstep_features = self.footstep_encoder(
                footstep_input
            )  # (batch, footstep_encoder_dim)

            # Combine with robot state and compute outputs
            combined = torch.cat([robot_features, footstep_features], dim=-1)
            trunk_features = self.shared_trunk(combined)

            # Get value and duration
            value = self.value_head(trunk_features)  # (batch, 1)
            duration_norm = self.duration_head(trunk_features)  # (batch, 1), in [0, 1]

            # Scale duration to actual range
            duration = self.min_swing_duration + duration_norm * (
                self.max_swing_duration - self.min_swing_duration
            )

            option_values.append(value)
            option_durations.append(duration)

        # Add no-action option
        no_action_features = self.no_action_embedding.unsqueeze(0).expand(
            batch_size, -1
        )
        no_action_combined = torch.cat([robot_features, no_action_features], dim=-1)
        no_action_trunk = self.shared_trunk(no_action_combined)
        no_action_value = self.value_head(no_action_trunk)
        no_action_duration = torch.zeros_like(
            no_action_value
        )  # No duration for no-action

        # add numeric representation for no-action to footstep options
        no_action_option = torch.zeros((batch_size, 1, 3), device=device)
        no_action_option[:, 0] = NO_STEP  # leg_idx = NO_STEP
        footstep_options = torch.cat(
            [footstep_options, no_action_option],
            dim=1,
        )  # (batch, num_options + 1, 3)

        option_values.append(no_action_value)
        option_durations.append(no_action_duration)

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
        get_footstep_options: Callable[..., torch.Tensor],
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

    def update_distribution(self, observations):
        """
        Update the action distribution for RL sampling.
        """
        robot_state = observations[:, : self.robot_state_dim]
        footstep_options_flat = observations[:, self.robot_state_dim :]
        batch_size = observations.shape[0]
        footstep_options = footstep_options_flat.reshape(
            batch_size, self.num_footstep_options, 3
        )
        action_values, _ = self.actor(robot_state, footstep_options)
        action_probs = nn.functional.softmax(action_values, dim=-1)
        self.distribution = torch.distributions.Categorical(action_probs)

    def get_actions_log_prob(self, actions):
        """
        Get log probabilities of actions for RL training.
        """
        # actions: (batch, 4) [leg_idx, dx, dy, swing_duration]
        # We map actions to their selected index (leg_idx, dx, dy, duration)
        # For compatibility, assume leg_idx is the index (for discrete selection)
        # If no-action, leg_idx == NO_STEP, which is last index
        batch_size = actions.shape[0]
        indices = actions[:, 0].long()
        # If leg_idx == NO_STEP, set to last index
        indices = torch.where(
            indices == NO_STEP,
            torch.tensor(self.num_footstep_options, device=actions.device),
            indices,
        )
        # Get log prob from distribution
        log_probs = self.distribution.log_prob(indices)
        return log_probs

    @property
    def action_mean(self):
        # Not meaningful for categorical, but required for compatibility
        return torch.zeros(self.num_actions, device=next(self.parameters()).device)

    @property
    def action_std(self):
        # Not meaningful for categorical, but required for compatibility
        return torch.ones(self.num_actions, device=next(self.parameters()).device)

    @property
    def entropy(self):
        return self.distribution.entropy()

    def act_inference(self, observations):
        """
        Deterministic action selection for inference.
        """
        return self.act(observations, deterministic=True)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Select action based on current observations.

        Args:
            observations: (batch, num_obs) tensor of observations
                [:, :robot_state_dim] - robot state
                [:, robot_state_dim:] - footstep options (leg_idx, dx, dy)*num_options flattened

        Returns:
            actions: (batch, 4) tensor with [leg_idx, dx, dy, swing_duration]
                    For no-action: [NO_STEP, 0, 0, 0]
        """
        # Get values and durations for all options
        footstep_options, action_values, swing_durations = self.actor(observations)

        # Convert to probabilities with softmax
        action_probs = nn.functional.softmax(action_values, dim=-1)

        # Sample action index (or take argmax if deterministic)
        if kwargs.get("deterministic", False):
            selected_indices = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            selected_indices = dist.sample()

        # Add swing durations as the 4th dimension
        swing_durations_expanded = swing_durations.unsqueeze(
            -1
        )  # (batch, num_options + 1, 1)
        actions_with_durations = torch.cat(
            [footstep_options, swing_durations_expanded], dim=-1
        )  # (batch, num_options + 1, 4)

        # Select the actions based on selected indices
        batch_size = observations.shape[0]
        batch_indices = torch.arange(batch_size, device=observations.device)
        selected_actions = actions_with_durations[batch_indices, selected_indices]

        return selected_actions

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute value estimate for observations.
        """
        return self.critic(critic_observations)
