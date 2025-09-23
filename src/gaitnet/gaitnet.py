from rsl_rl.modules import ActorCritic
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
import torch
import torch.nn as nn

from src.util import log_exceptions
from src import get_logger

logger = get_logger()


class FootstepValueNetwork(nn.Module):
    """
    Value-based network for scoring individual footstep options.
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

        # Value head: combines robot state and footstep features
        self.value_head = nn.Sequential(
            nn.Linear(shared_encoder_dim + footstep_encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Output single value
        )

        # Special embedding for "no action" option
        self.no_action_embedding = nn.Parameter(torch.randn(footstep_encoder_dim))

    def forward(
        self,
        robot_state: torch.Tensor,
        footstep_options: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass to compute values for all footstep options.

        Args:
            robot_state: (batch_size, 22) robot state information
            footstep_options: (batch_size, num_options, 3) footstep options
                             [leg_idx, horizontal_idx, vertical_idx]

        Returns:
            values: (batch_size, num_options + 1) values for each option + no-action
        """
        batch_size = robot_state.shape[0]
        device = robot_state.device

        # Encode robot state (shared for all options)
        robot_features = self.robot_state_encoder(
            robot_state
        )  # (batch, shared_encoder_dim)

        # Process each footstep option
        option_values = []

        num_options = footstep_options.shape[1]

        for i in range(num_options):
            # Extract option
            option = footstep_options[:, i, :]  # (batch, 3)

            # Convert leg index to one-hot
            leg_idx = option[:, 0].long()
            leg_one_hot = nn.functional.one_hot(
                leg_idx, num_classes=4
            ).float()  # (batch, 4)

            # Convert grid indices to normalized offsets
            # Assuming 5x5 grid, normalize to [-1, 1]
            x_offset = (option[:, 1] - 2.0) / 2.0  # Center at 2, scale by 2
            y_offset = (option[:, 2] - 2.0) / 2.0

            # Combine footstep features
            footstep_input = torch.cat(
                [leg_one_hot, x_offset.unsqueeze(-1), y_offset.unsqueeze(-1)], dim=-1
            )  # (batch, 6)

            # Encode footstep
            footstep_features = self.footstep_encoder(
                footstep_input
            )  # (batch, footstep_encoder_dim)

            # Combine with robot state and compute value
            combined = torch.cat([robot_features, footstep_features], dim=-1)
            value = self.value_head(combined)  # (batch, 1)
            option_values.append(value)

        # Add no-action option
        no_action_features = self.no_action_embedding.unsqueeze(0).expand(
            batch_size, -1
        )
        no_action_combined = torch.cat([robot_features, no_action_features], dim=-1)
        no_action_value = self.value_head(no_action_combined)
        option_values.append(no_action_value)

        # Stack all values
        all_values = torch.cat(option_values, dim=-1)  # (batch, num_options + 1)

        return all_values


class FootstepActorCritic(ActorCritic):
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
        **kwargs
    ):
        # For our case: num_actions = 6 (5 footstep options + 1 no-action)
        super().__init__(
            num_obs, num_actions, hidden_dims, activation, init_noise_std, **kwargs
        )

        # Override with our custom architecture
        self.robot_state_dim = 22
        self.num_footstep_options = 5

        # Replace the actor with our value-based network
        self.actor = FootstepValueNetwork(
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
        """
        # Parse observations
        robot_state = observations[:, : self.robot_state_dim]
        footstep_options_flat = observations[:, self.robot_state_dim :]

        # Reshape footstep options
        batch_size = observations.shape[0]
        footstep_options = footstep_options_flat.reshape(
            batch_size, self.num_footstep_options, 3
        )

        # Get values for all options
        action_values = self.actor(robot_state, footstep_options)

        # Convert to probabilities with softmax
        action_probs = nn.functional.softmax(action_values, dim=-1)

        # Sample action (or take argmax if deterministic)
        if kwargs.get("deterministic", False):
            actions = torch.argmax(action_probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            actions = dist.sample()

        return actions

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute value estimate for observations.
        """
        return self.critic(critic_observations)


def main(): ...


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
