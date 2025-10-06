import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gaitnet.actions.footstep_action import Sequence
from rsl_rl.modules import ActorCritic
from src import get_logger
from src.gaitnet.env_cfg.footstep_options_env import FootstepOptionEnv
from torch.distributions import Normal, Distribution, constraints
import src.constants as const

logger = get_logger()


class TopKCategoricalDistribution(Distribution):
    """
    A custom distribution for selecting top-k actions from a set of options.
    
    This distribution:
    - Samples the top-k indices based on logits (non-differentiable sampling)
    - Computes log_prob differentiably w.r.t. the logits
    - Maintains compatibility with PPO's training loop
    """
    
    arg_constraints = {}
    support = constraints.integer_interval(0, float('inf'))
    has_rsample = False
    
    def __init__(self, logits: torch.Tensor, k: int = 2, validate_args=None):
        """
        Args:
            logits: Tensor of shape (batch_size, num_options) with unnormalized log probabilities
            k: Number of top actions to select
            validate_args: Whether to validate arguments (for PyTorch distribution interface)
        """
        self.k = k
        self.logits = logits
        self.num_options = logits.shape[-1]
        
        # Convert logits to probabilities for log_prob computation
        self.probs = F.softmax(logits, dim=-1)
        
        # Store log probabilities for numerical stability
        self.log_probs = F.log_softmax(logits, dim=-1)
        
        batch_shape = logits.shape[:-1]
        event_shape = torch.Size([k])
        
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
    
    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        """
        Sample top-k indices based on logits with Gumbel noise for exploration.
        
        Uses the Gumbel-Max trick to add stochasticity while biasing toward higher logits.
        
        Returns:
            Tensor of shape (*sample_shape, *batch_shape, k) containing selected indices
        """
        if sample_shape != torch.Size():
            raise NotImplementedError("TopKCategoricalDistribution does not support sample_shape != ()")
        
        # Add Gumbel noise for stochastic sampling
        # Gumbel(0, 1) = -log(-log(Uniform(0, 1)))
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.logits) + 1e-10) + 1e-10)
        
        # Add noise to logits (Gumbel-Max trick)
        perturbed_logits = self.logits + gumbel_noise
        
        # Select top-k from perturbed logits
        _, top_indices = torch.topk(perturbed_logits, k=self.k, dim=-1, sorted=True)
        
        return top_indices  # Shape: (batch_size, k)
    
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of selecting the given top-k actions.
        
        This uses a differentiable approximation: we treat the selection as independent
        draws from the categorical distribution over options, which allows gradients
        to flow back through the logits.
        
        Args:
            actions: Tensor of shape (batch_size, k) containing the selected action indices
        
        Returns:
            Tensor of shape (batch_size,) containing log probabilities
        """
        batch_size = actions.shape[0]
        
        # Ensure actions are long tensor for indexing
        actions = actions.long()
        
        # Gather log probabilities for the selected actions
        # actions shape: (batch_size, k)
        # self.log_probs shape: (batch_size, num_options)
        
        # Expand batch indices for gather operation
        batch_indices = torch.arange(batch_size, device=actions.device).unsqueeze(1).expand(-1, self.k)
        
        # Gather the log probabilities of selected actions
        selected_log_probs = self.log_probs[batch_indices, actions]  # (batch_size, k)
        
        # Sum log probabilities across the k selections
        # This treats selections as independent (approximation for differentiability)
        total_log_prob = selected_log_probs.sum(dim=-1)  # (batch_size,)
        
        return total_log_prob
    
    def entropy(self) -> torch.Tensor:
        """
        Compute the entropy of the categorical distribution over options.
        
        Since we're selecting top-k, we compute the entropy of the underlying
        categorical distribution that the selections are based on.
        
        Returns:
            Tensor of shape (batch_size,) containing entropy values
        """
        # Entropy of categorical distribution: -sum(p * log(p))
        entropy = -(self.probs * self.log_probs).sum(dim=-1)  # (batch_size,)
        
        return entropy
    
    @property
    def mean(self) -> torch.Tensor:
        """
        Return the expected action indices (for logging purposes).
        This returns the top-k indices, which is what would be sampled.
        """
        _, top_indices = torch.topk(self.logits, k=self.k, dim=-1, sorted=True)
        return top_indices.float()  # (batch_size, k)


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
        # in theory this part should learn how we are using the output of the actor (topk 2)
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
            action_indices: Selected action indices (num_envs, k)
            
        Returns:
            durations: Durations for selected actions (num_envs, k)
        """
        if self._cached_durations is None:
            raise RuntimeError("No cached durations available. Call forward() first.")
        
        batch_size, k = action_indices.shape
        batch_indices = torch.arange(batch_size, device=action_indices.device).unsqueeze(1).expand(-1, k)
        
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
        
        # Number of top actions to select
        self.k = 2
        
        # Store the number of options (should match action_dim from FSCActionTerm)
        self.num_options = const.gait_net.num_footstep_options * const.robot.num_legs + 1  # +1 for no-op

        # Action distribution (populated in update_distribution)
        self.distribution: TopKCategoricalDistribution | None = None
        
        # Cache for storing selected action indices
        self._last_action_indices: torch.Tensor | None = None

    def reset(self, dones=None):
        """Reset recurrent states (no-op for non-recurrent policy)."""
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        """Return the mean of the distribution (for logging)."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution first.")
        return self.distribution.mean
    
    @property
    def action_std(self):
        """Return dummy std for logging compatibility with PPO."""
        # TopK distribution doesn't have a traditional std, return ones for logging
        if self.distribution is None:
            return torch.ones(self.num_options)
        return torch.ones_like(self.distribution.logits[:, :1])
    
    @property
    def entropy(self):
        """Return the entropy of the distribution."""
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution first.")
        return self.distribution.entropy()
    
    def update_distribution(self, obs):
        """Update the action distribution based on observations.
        
        Args:
            obs: Observations (num_envs, obs_dim)
        """
        # Get logits from actor
        logits = self.actor(obs)  # (num_envs, num_options)
        
        # Create TopK distribution
        self.distribution = TopKCategoricalDistribution(logits, k=self.k)
    
    def act(self, obs, **kwargs):
        """Sample actions from the policy.
        
        Args:
            obs: Observations (num_envs, obs_dim)
            
        Returns:
            action_indices: Sampled action indices (num_envs, k)
        """
        self.update_distribution(obs)
        action_indices = self.distribution.sample()  # (num_envs, k)
        
        # Cache the action indices for duration lookup
        self._last_action_indices = action_indices
        
        return action_indices
    
    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution.
        
        Args:
            actions: Action indices (num_envs, k)
            
        Returns:
            log_probs: Log probabilities (num_envs,)
        """
        if self.distribution is None:
            raise RuntimeError("Distribution not initialized. Call update_distribution first.")
        return self.distribution.log_prob(actions)
    
    def act_inference(self, obs):
        """Deterministic action selection for inference.
        
        Args:
            obs: Observations (num_envs, obs_dim)
            
        Returns:
            action_indices: Top-k action indices (num_envs, k)
        """
        logits = self.actor(obs)
        _, top_indices = torch.topk(logits, k=self.k, dim=-1, sorted=True)
        return top_indices
    
    def evaluate(self, obs, **kwargs):
        """Evaluate the value function.
        
        Args:
            obs: Observations (num_envs, obs_dim)
            
        Returns:
            values: State values (num_envs, 1)
        """
        return self.critic(obs)
