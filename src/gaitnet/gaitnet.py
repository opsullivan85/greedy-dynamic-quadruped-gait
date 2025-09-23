from ast import Num
from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
import torch
import torch.nn as nn

from src.util import log_exceptions
from src import get_logger

logger = get_logger()


class Gaitnet(nn.Module):
    """
    Value-based network for scoring individual footstep options and predicting swing duration.
    This network learns to evaluate each footstep option independently.
    """

    def __init__(
        self,
        robot_state_dim: int = 22,
        shared_encoder_dim: int = 128,
        footstep_encoder_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

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
        # Input: [leg_idx_one_hot(4), x_offset, y_offset]
        self.footstep_encoder = nn.Sequential(
            nn.Linear(4 + 2, footstep_encoder_dim),  # 4 for one-hot leg, 2 for x,y
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
        footstep_options: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to compute values and swing durations for all footstep options.

        Args:
            robot_state: (batch_size, 22) robot state information
            footstep_options: (batch_size, num_options, 3) footstep options
                             [leg_idx, dx, dy]
            num_options: number of footstep options

        Returns:
            values: (batch_size, num_options + 1) values for each option + no-action
            durations: (batch_size, num_options + 1) swing durations for each option
        """
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

            # Combine footstep features
            footstep_input = torch.cat(
                [leg_one_hot, dx.unsqueeze(-1), dy.unsqueeze(-1)], dim=-1
            )  # (batch, 4+1+1)

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

        option_values.append(no_action_value)
        option_durations.append(no_action_duration)

        # Stack all values and durations
        all_values = torch.cat(option_values, dim=-1)  # (batch, num_options + 1)
        all_durations = torch.cat(option_durations, dim=-1)  # (batch, num_options + 1)

        return all_values, all_durations


class GaitNetActorCritic(ActorCritic):
    """
    Actor-Critic network for footstep selection using value-based approach.
    Compatible with RSL-RL.
    """

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        hidden_dims: list = [256, 256],
        activation: str = "relu",
        init_noise_std: float = 1.0,
        robot_state_dim: int = 22,
        num_footstep_options: int = 5,
        **kwargs
    ):
        """Initialize the GaitNetActorCritic.

        Args:
            num_obs (int): Number of observation dimensions.
            num_actions (int): Number of action dimensions.
            hidden_dims (list, optional): Hidden dimensions for the network. Defaults to [256, 256].
            activation (str, optional): Activation function to use. Defaults to "relu".
            init_noise_std (float, optional): Standard deviation for initial noise. Defaults to 1.0.
            robot_state_dim (int, optional): Dimension of the robot state representation. Defaults to 22.
            num_footstep_options (int, optional): Number of footstep options. Defaults to 5.
        """
        # For our case: num_actions = 6 (5 footstep options + 1 no-action)
        super().__init__(
            num_obs, num_actions, hidden_dims, activation, init_noise_std, **kwargs
        )

        # Override with our custom architecture
        self.robot_state_dim = robot_state_dim
        self.num_footstep_options = num_footstep_options

        # Replace the actor with our value-based network
        self.actor = Gaitnet(
            robot_state_dim=self.robot_state_dim,
            shared_encoder_dim=hidden_dims[0] // 2,
            footstep_encoder_dim=hidden_dims[0] // 4,
            hidden_dim=hidden_dims[0],
        )

        # Keep standard value network for critic
        self.critic = nn.Sequential(
            nn.Linear(num_obs, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Select action based on current observations.

        Args:
            observations: (batch, num_obs) tensor of observations
                [:, :robot_state_dim] - robot state
                [:, robot_state_dim:] - footstep options (leg_idx, horizontal_idx, vertical_idx)*num_options flattened

        Returns:
            actions: (batch, 4) tensor with [leg_idx, dx, dy, swing_duration]
                    For no-action: [-1, 0, 0, 0]
        """
        # Parse observations
        robot_state = observations[:, : self.robot_state_dim]
        footstep_options_flat = observations[:, self.robot_state_dim :]

        # Reshape footstep options
        batch_size = observations.shape[0]
        footstep_options = footstep_options_flat.reshape(
            batch_size, self.num_footstep_options, 3
        )

        # Get values and durations for all options
        action_values, swing_durations = self.actor(robot_state, footstep_options)

        # Convert to probabilities with softmax
        action_probs = nn.functional.softmax(action_values, dim=-1)

        # Sample action index (or take argmax if deterministic)
        if kwargs.get("deterministic", False):
            selected_indices = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            selected_indices = dist.sample()

        # Now convert selected indices to actual action parameters
        # get the associated footstep option or no-action for each selected index
        noop = torch.tensor([-1.0, 0.0, 0.0], device=observations.device)
        footstep_options_with_noop = torch.cat(
            [
                footstep_options,
                noop.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1),
            ],
            dim=1,
        )  # (batch, num_options + 1, 3)
        # add the swing duration to the last dimension
        swing_durations = swing_durations.unsqueeze(-1)  # (batch, num_options + 1, 1)
        footstep_options_with_noop = torch.cat(
            [footstep_options_with_noop, swing_durations], dim=-1
        )  # (batch, num_options + 1, 4)
        selected_options = footstep_options_with_noop[selected_indices]
        return selected_options

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute value estimate for observations.
        """
        return self.critic(critic_observations)


def main(): ...


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
