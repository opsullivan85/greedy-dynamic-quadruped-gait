import torch
import torch.nn as nn

from src.gaitnet.actions.footstep_action import Sequence
from rsl_rl.modules import ActorCritic
from src import get_logger
from src.gaitnet.env_cfg.footstep_options_env import FootstepOptionEnv
from torch.distributions import Normal

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
            output_activation=nn.Tanh,
        )
        logger.info(f"duration_head: {self.duration_head}")

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
        trunk_output = self.trunk(trunk_input)

        value = self.value_head(trunk_output).squeeze(-1)
        duration = self.duration_head(trunk_output).squeeze(-1)

        return value, duration


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
    """Wrap a GaitnetActor so that the duration values are used"""

    def __init__(self, actor: GaitnetActor, env: FootstepOptionEnv):
        super().__init__()
        self._actor = actor
        self._env = env

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        value, duration = self._actor(obs)
        self._env.observation_manager.set_footstep_actions(duration)
        return value


class GaitnetActorCritic(ActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor: GaitnetActor,
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

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

        # Action distribution (populated in update_distribution)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)
