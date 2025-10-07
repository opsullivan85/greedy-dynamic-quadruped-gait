import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gaitnet.actions.footstep_action import Sequence
from rsl_rl.modules import ActorCritic
from src import get_logger
from src.gaitnet.env_cfg.footstep_options_env import FootstepOptionEnv
from torch.distributions import Normal, Categorical
import src.constants as const

logger = get_logger()


def make_mlp(
    input_size, hidden_sizes, output_size, activation=nn.ReLU, output_activation=None
) -> nn.Sequential:
    """
    Creates an MLP (multi-layer perceptron) in PyTorch.

    Args:
        input_size (int): Number of input features.
        hidden_sizes (list[int]): Sizes of hidden layers.
        output_size (int): Number of output features.
        activation (nn.Module): Activation class for hidden layers (default: ReLU).
        output_activation (nn.Module or None): Optional activation for output layer.

    Returns:
        nn.Sequential: The constructed MLP.
    """
    layers = []
    in_size = input_size
    for h in hidden_sizes:
        layers.append(nn.Linear(in_size, h))
        layers.append(activation())
        in_size = h
    layers.append(nn.Linear(in_size, output_size))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class GaitnetActor(nn.Module):
    def __init__(
        self,
        shared_state_dim: int,
        shared_layer_sizes: Sequence[int],
        unique_state_dim: int,
        unique_layer_sizes: Sequence[int],
        trunk_layer_sizes: Sequence[int],
    ):
        super().__init__()

        self.shared_state_size = 22
        self.unique_state_size = 8  # leg one-hot (5), dx, dy, cost

        logger.info("GaitnetActor initializing")

        self.shared_encoder = make_mlp(
            input_size=shared_state_dim,
            hidden_sizes=shared_layer_sizes[:-1],
            output_size=shared_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        logger.info(f"shared_encoder: {self.shared_encoder}")

        self.unique_encoder = make_mlp(
            input_size=unique_state_dim,
            hidden_sizes=unique_layer_sizes[:-1],
            output_size=unique_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        self.unique_embedding_size = unique_layer_sizes[-1]
        # random embedding to represent no-op
        self.no_op_embedding = nn.Parameter(torch.randn(unique_layer_sizes[-1]))
        logger.info(f"unique_encoder: {self.unique_encoder}")

        trunk_input_dim = shared_layer_sizes[-1] + unique_layer_sizes[-1]
        self.trunk = make_mlp(
            input_size=trunk_input_dim,
            hidden_sizes=trunk_layer_sizes[:-1],
            output_size=trunk_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        logger.info(f"trunk: {self.trunk}")

        self.value_head = make_mlp(
            input_size=trunk_layer_sizes[-1], hidden_sizes=[], output_size=1
        )
        logger.info(f"value_head: {self.value_head}")

        self.duration_head = make_mlp(
            input_size=trunk_layer_sizes[-1],
            hidden_sizes=[],
            output_size=1,
        )
        logger.info(f"duration_head: {self.duration_head}")

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the GaitNet actor.

        Args:
            obs (torch.Tensor): Input observations.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - logits: Action selection logits (num_envs, num_options)
                - durations: Duration predictions for each option (num_envs, num_options)
        """
        num_envs = obs.shape[0]
        shared_state = obs[:, : self.shared_state_size]

        remaining_obs_size = obs.shape[1] - self.shared_state_size
        unique_states_dim = remaining_obs_size / self.unique_state_size
        assert (
            unique_states_dim.is_integer()
        ), f"Expected unique_state_size ({self.unique_state_size}) to evenly divide the remaining observation size ({remaining_obs_size}), got {unique_states_dim}"
        unique_states_dim = int(unique_states_dim)
        unique_states = obs[:, self.shared_state_size :].view(
            num_envs, unique_states_dim, self.unique_state_size
        )
        unique_states_iter = torch.split(unique_states, 1, dim=1)

        shared_embedding: torch.Tensor = self.shared_encoder(shared_state)

        unique_embeddings = []
        for unique_state in unique_states_iter:
            # remove the extra dimension
            unique_state = unique_state.squeeze(1)
            # check if this is a no-op state
            no_op_mask = unique_state[:, 4] == 1
            # also treat high cost as no-op
            no_op_mask = no_op_mask | (unique_state[:, -1] >= 2.0)

            unique_embedding = torch.zeros(
                (num_envs, self.unique_embedding_size), device=obs.device
            )
            if (~no_op_mask).any():
                unique_embedding[~no_op_mask] = self.unique_encoder(
                    unique_state[~no_op_mask]
                )
            if no_op_mask.any():
                unique_embedding[no_op_mask] = self.no_op_embedding
            unique_embeddings.append(unique_embedding)

        unique_embeddings = torch.stack(unique_embeddings, dim=1)
        trunk_input = torch.cat(
            [
                shared_embedding.unsqueeze(dim=1).expand(-1, unique_states_dim, -1),
                unique_embeddings,
            ],
            dim=-1,
        )
        trunk_output = self.trunk(trunk_input)

        logits = self.value_head(trunk_output).squeeze(-1)  # (num_envs, num_options)
        duration = self.duration_head(trunk_output).squeeze(-1)  # (num_envs, num_options)
        
        # Scale durations to reasonable range (e.g., 0.1 to 0.5 seconds)
        # Using sigmoid to map to [0, 1], then scale to [0.1, 0.5]
        duration = torch.sigmoid(duration) * 0.4 + 0.1

        return logits, duration


class GaitnetCritic(nn.Module):
    def __init__(
        self,
        shared_state_dim: int,
        shared_layer_sizes: Sequence[int],
        num_unique_states: int,
        unique_state_dim: int,
        unique_layer_sizes: Sequence[int],
        trunk_layer_sizes: Sequence[int],
        trunk_combiner_head_sizes: Sequence[int] = [32, 32],
    ):
        super().__init__()

        self.shared_state_size = 22
        self.unique_state_size = 8  # leg one-hot (5), dx, dy, cost

        logger.info("GaitnetCritic initializing")

        self.shared_encoder = make_mlp(
            input_size=shared_state_dim,
            hidden_sizes=shared_layer_sizes[:-1],
            output_size=shared_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        logger.info(f"shared_encoder: {self.shared_encoder}")

        self.unique_encoder = make_mlp(
            input_size=unique_state_dim,
            hidden_sizes=unique_layer_sizes[:-1],
            output_size=unique_layer_sizes[-1],
            output_activation=nn.ReLU,
        )
        self.unique_embedding_size = unique_layer_sizes[-1]
        # random embedding to represent no-op
        self.no_op_embedding = nn.Parameter(torch.randn(unique_layer_sizes[-1]))
        logger.info(f"unique_encoder: {self.unique_encoder}")

        trunk_input_dim = shared_layer_sizes[-1] + unique_layer_sizes[-1]
        self.trunk = make_mlp(
            input_size=trunk_input_dim,
            hidden_sizes=trunk_layer_sizes,
            output_size=1,
        )

        self.trunk_combiner_head = make_mlp(
            input_size=num_unique_states,
            hidden_sizes=trunk_combiner_head_sizes,
            output_size=1,
        )
        logger.info(f"trunk: {self.trunk}")

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the MiMo network.

        Args:
            obs (torch.Tensor): Input observations.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Value and duration predictions.
        """
        num_envs = obs.shape[0]
        shared_state = obs[:, : self.shared_state_size]

        remaining_obs_size = obs.shape[1] - self.shared_state_size
        unique_states_dim = remaining_obs_size / self.unique_state_size
        assert (
            unique_states_dim.is_integer()
        ), f"Expected unique_state_size ({self.unique_state_size}) to evenly divide the remaining observation size ({remaining_obs_size}), got {unique_states_dim}"
        unique_states_dim = int(unique_states_dim)
        unique_states = obs[:, self.shared_state_size :].view(
            num_envs, unique_states_dim, self.unique_state_size
        )
        unique_states_iter = torch.split(unique_states, 1, dim=1)

        shared_embedding: torch.Tensor = self.shared_encoder(shared_state)

        unique_embeddings = []
        for unique_state in unique_states_iter:
            # remove the extra dimension
            unique_state = unique_state.squeeze(1)
            # check if this is a no-op state
            no_op_mask = unique_state[:, 4] == 1
            # also treat high cost as no-op
            no_op_mask = no_op_mask | (unique_state[:, -1] >= 2.0)

            unique_embedding = torch.zeros(
                (num_envs, self.unique_embedding_size), device=obs.device
            )
            if (~no_op_mask).any():
                unique_embedding[~no_op_mask] = self.unique_encoder(
                    unique_state[~no_op_mask]
                )
            if no_op_mask.any():
                unique_embedding[no_op_mask] = self.no_op_embedding
            unique_embeddings.append(unique_embedding)

        unique_embeddings = torch.stack(unique_embeddings, dim=1)
        trunk_input = torch.cat(
            [
                shared_embedding.unsqueeze(dim=1).expand(-1, unique_states_dim, -1),
                unique_embeddings,
            ],
            dim=-1,
        )
        values = self.trunk(trunk_input).squeeze(-1)  # (num_envs, num_unique_states)
        # collapse the values across the unique states to a single value
        # in theory this part should learn how we are using the output of the actor (single action selection)
        value = self.trunk_combiner_head(values)  # (num_envs, 1)

        return value


class GaitnetActorWrapper(nn.Module):
    """Wrap a GaitnetActor to handle duration storage for selected actions."""

    def __init__(self, actor: GaitnetActor, env: FootstepOptionEnv):
        super().__init__()
        self._actor = actor
        self._env = env
        self._cached_durations = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns logits and caches durations.
        
        Args:
            obs: Observations (num_envs, obs_dim)
            
        Returns:
            logits: Action selection logits (num_envs, num_options)
        """
        logits, durations = self._actor(obs)
        # Cache durations for use after action selection
        self._cached_durations = durations
        return logits
    
    def get_durations_for_actions(self, action_indices: torch.Tensor) -> torch.Tensor:
        """Get the durations for the selected action indices.
        
        Args:
            action_indices: Selected action indices (num_envs,) or (num_envs, 1)
            
        Returns:
            durations: Durations for selected actions (num_envs,)
        """
        if self._cached_durations is None:
            raise RuntimeError("No cached durations available. Call forward() first.")
        
        # Flatten action_indices if it has shape (num_envs, 1)
        if action_indices.dim() > 1:
            action_indices = action_indices.squeeze(-1)
        
        batch_size = action_indices.shape[0]
        batch_indices = torch.arange(batch_size, device=action_indices.device)
        
        # Gather durations for the selected actions
        selected_durations = self._cached_durations[batch_indices, action_indices]
        
        return selected_durations
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped actor."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._actor, name)


class GaitnetActorCritic(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor: GaitnetActorWrapper,
        critic: GaitnetCritic,
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
    ):
        nn.Module.__init__(self)
        logger.info("GaitnetActorCritic initializing")
        logger.info(
            "Note: num_actor_obs, num_critic_obs, and num_actions are not used in making the actor and critic."
        )
        logger.debug(
            f"num_actor_obs: {num_actor_obs}, num_critic_obs: {num_critic_obs}, num_actions: {num_actions}"
        )

        self.actor = actor
        self.critic = critic
        
        # Store the number of options (should match num_actions from environment)
        self.num_options = const.gait_net.num_footstep_options * const.robot.num_legs + 1  # +1 for no-op

        # Action distribution (populated in update_distribution)
        self.distribution: Categorical | None = None
        
        # Cache for storing selected action indices
        self._last_action_indices: torch.Tensor | None = None

    def reset(self, dones=None):
        """Reset recurrent states (no-op for non-recurrent policy)."""
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """Return the mean of the distribution (for logging).
        
        For Categorical distribution, returns the mode (most likely action).
        Returns shape (num_envs, 1) to match action shape.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution first.")
        # Get the mode (argmax of logits/probs) and reshape to (num_envs, 1)
        return torch.argmax(self.distribution.logits, dim=-1, keepdim=True).float()
    
    @property
    def action_std(self):
        """Return dummy std for logging compatibility with PPO.
        
        Returns shape (num_envs, 1) to match action shape.
        Categorical distribution doesn't have a traditional std, so we return ones.
        """
        if self.distribution is None:
            return torch.ones(1)
        # Return ones with shape (num_envs, 1)
        num_envs = self.distribution.logits.shape[0]
        return torch.ones(num_envs, 1, device=self.distribution.logits.device)
    
    @property
    def entropy(self):
        """Return the entropy of the distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution first.")
        return self.distribution.entropy()
    
    def update_distribution(self, observations):
        """Update the action distribution based on observations.
        
        Args:
            observations: Observations (num_envs, obs_dim)
        """
        # Get logits from actor
        logits = self.actor(observations)  # (num_envs, num_options=17)
        
        # Create Categorical distribution
        self.distribution = Categorical(logits=logits)
    
    def act(self, observations, **kwargs):
        """Sample actions from the policy.
        
        Args:
            observations: Observations (num_envs, obs_dim)
            
        Returns:
            action_index: Sampled action index (num_envs,) with integer values 0-16
        """
        self.update_distribution(observations)
        action_index = self.distribution.sample().unsqueeze(-1)  # (num_envs, 1)
        
        # Cache the action index for duration lookup
        self._last_action_indices = action_index
        
        return action_index
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution.
        
        Args:
            actions: Action indices (num_envs,) or (num_envs, 1)
            
        Returns:
            log_probs: Log probabilities (num_envs,)
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution first.")
        
        # Flatten actions if it has shape (num_envs, 1)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        
        return self.distribution.log_prob(actions)
    
    def act_inference(self, observations):
        """Deterministic action selection for inference.
        
        Args:
            observations: Observations (num_envs, obs_dim)
            
        Returns:
            action_index: Most likely action index (num_envs,)
        """
        logits = self.actor(observations)
        # Select action with highest logit (deterministic)
        action_index = torch.argmax(logits, dim=-1)
        return action_index.unsqueeze(-1)  # (num_envs, 1)
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate the value function.
        
        Args:
            critic_observations: Observations (num_envs, obs_dim)
            
        Returns:
            values: State values (num_envs, 1)
        """
        return self.critic(critic_observations)
