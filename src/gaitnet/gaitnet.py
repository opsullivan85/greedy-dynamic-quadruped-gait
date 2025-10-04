from typing import Callable
from rsl_rl.modules import ActorCritic
import torch
import torch.nn as nn

from isaaclab.envs import ManagerBasedRLEnv
from src.gaitnet.actions.footstep_action import NO_STEP
from src import contactnet, get_logger
from src.contactnet.contactnet import CostMapGenerator
from src.gaitnet.env_cfg.observations import get_terrain_mask
from src.simulation.cfg.footstep_scanner_constants import idx_to_xy

from src.contactnet.debug import view_footstep_cost_map
from src.util.data_logging import save_fig, save_img
import src.constants as const
from src.util.math import seeded_uniform_noise

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

        # require minimum number leg to be in contact
        # by setting all costs to inf if not enough legs in contact
        min_legs_in_contact = 2
        num_legs_in_contact = contact_states.sum(dim=1)  # (num_envs,)
        contact_limit = num_legs_in_contact <= min_legs_in_contact  # (num_envs,)
        masked_cost_maps = torch.where(
            contact_limit.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
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
        # technically there is no need to sort here, but
        # it will guarantee determinism
        topk_values, topk_flat_indices = torch.topk(
            flat_cost_map, num_options, largest=False, sorted=True
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
        contactnet_obs = obs[:, :18]
        with torch.inference_mode():
            cost_maps = self.cost_map_generator.predict(
                contactnet_obs
            )  # (num_envs, 4, H, W)

        # switch from (FL, FR, RL, RR) to (FR, FL, RR, RL)
        cost_maps = cost_maps[:, [1, 0, 3, 2], :, :]

        # save_img(
        #     cost_maps[0, 0].cpu().numpy(),
        #     name="raw_cost_map_FR",
        #     cmap_limits=(-1, 1),
        # )

        # interpolate cost map
        cost_maps = nn.functional.interpolate(
            cost_maps,  # (num_envs, 4, h, w) 4 is being used as the channel dimension here
            size=const.footstep_scanner.grid_size.tolist(),  # (H, W)
            mode="bilinear",
            align_corners=True,
        ).squeeze(
            1
        )  # (num_envs, 4, H, W)

        # apply determanistic noise to costmap to slightly spread apart the best options
        # it is important that this is reproducable for the RL algorithm
        noise = seeded_uniform_noise(cost_maps, cost_maps.shape[1:])
        noise = (
            noise * 2 * const.footstep_scanner.upscale_costmap_noise
            - const.footstep_scanner.upscale_costmap_noise
        )
        cost_maps += noise

        # save_img(
        #     cost_maps[0, 0].cpu().numpy(),
        #     name="interpolated_cost_map_FR",
        # )
        # save_img(
        #     cost_maps[0, 1].cpu().numpy(),
        #     name="interpolated_cost_map_FL",
        # )
        # save_img(
        #     cost_maps[0, 2].cpu().numpy(),
        #     name="interpolated_cost_map_RL",
        # )
        # save_img(
        #     cost_maps[0, 3].cpu().numpy(),
        #     name="interpolated_cost_map_RR",
        # )

        masked_cost_maps = self._filter_cost_map(cost_maps, obs)  # (num_envs, 4, H, W)

        # save_img(
        #     cost_maps[0, 0].cpu().numpy(),
        #     name="masked_cost_map_FR",
        #     cmap_limits=(-1, 1),
        # )

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

        if torch.isnan(all_values).any():
            # lord help you...
            logger.error("NaN in all_values!")

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
        **kwargs,
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
            **kwargs,
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
        self._cached_selected_actions: torch.Tensor = None  # type: ignore
        self._cached_selected_log_probs: torch.Tensor = None  # type: ignore

    def _ensure_forward_pass(self, observations: torch.Tensor):
        """
        Ensure forward pass has been computed for given observations.
        Uses caching to avoid redundant computation.
        """
        # Check if we need to compute or observations have changed
        if self._cached_observations is None or not torch.equal(
            self._cached_observations, observations
        ):

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
        Content-based action selection that's robust to option re-ordering.

        Args:
            observations: (batch, num_obs) tensor of observations (robot state)

        Returns:
            actions: (batch, 4) tensor with [leg_idx, dx, dy, swing_duration]
                    For no-action: [NO_STEP, 0, 0, 0]
        """
        self._ensure_forward_pass(observations)

        batch_size = observations.shape[0]

        # Prepare actions with durations
        actions_with_durations = torch.cat(
            [self._cached_footstep_options, self._cached_swing_durations.unsqueeze(-1)],
            dim=-1,
        )  # (batch, num_options + 1, 4)

        if kwargs.get("deterministic", False):
            # Vectorized deterministic selection
            best_indices = self._cached_action_values.argmax(dim=1)  # (batch,)
        else:
            # Vectorized stochastic sampling
            best_indices = torch.multinomial(self._cached_action_probs, 1).squeeze(
                1
            )  # (batch,)

        # Gather selected actions: (batch, 4)
        selected_actions = actions_with_durations[
            torch.arange(batch_size, device=observations.device), best_indices
        ]

        # Compute log probs: (batch,)
        selected_log_probs = torch.log(
            self._cached_action_probs[
                torch.arange(batch_size, device=observations.device), best_indices
            ]
            + 1e-8
        )

        # Cache for efficiency
        self._cached_selected_actions = selected_actions.clone()
        self._cached_selected_log_probs = selected_log_probs

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
        Compute log probabilities based on action content, not indices.

        NOTE: PPO calls act() with the mini-batch observations BEFORE calling this method,
        so the cache should be valid for the current mini-batch.

        Args:
            actions: (batch, 4) tensor of actions [leg_idx, dx, dy, swing_duration]

        Returns:
            log_probs: (batch,) tensor of log probabilities
        """
        # Quick path: if actions match cached selected actions exactly
        if (
            self._cached_selected_actions is not None
            and actions.shape == self._cached_selected_actions.shape
            and actions.device == self._cached_selected_actions.device
            and torch.allclose(
                actions, self._cached_selected_actions, rtol=1e-5, atol=1e-8
            )
        ):
            return self._cached_selected_log_probs

        # Vectorized matching on footstep content only (ignore duration)
        batch_size = actions.shape[0]
        device = actions.device

        # Extract footstep part: [leg_idx, dx, dy]
        actions_footstep = actions[:, :3]  # (batch, 3)
        options_footstep = self._cached_footstep_options  # (batch, num_options+1, 3)

        # Compute pairwise differences: (batch, num_options+1, 3)
        # Expand dims for broadcasting: (batch, 1, 3) vs (batch, num_options+1, 3)
        differences = torch.abs(
            actions_footstep.unsqueeze(1) - options_footstep
        )  # (batch, num_options+1, 3)

        # Sum across feature dimension to get total difference per option
        total_diff = differences.sum(dim=-1)  # (batch, num_options+1)

        # Find best matching option for each batch element
        best_match_indices = torch.argmin(total_diff, dim=-1)  # (batch,)

        # Optional: Check if matching failed (for debugging)
        min_diffs = total_diff[
            torch.arange(batch_size, device=device), best_match_indices
        ]
        if torch.any(min_diffs > 1e-4):
            num_failed = (min_diffs > 1e-4).sum().item()
            max_diff = min_diffs.max().item()
            logger.warning(
                f"Action matching failed for {num_failed}/{batch_size} samples! "
                f"Max difference: {max_diff:.6f}"
            )

        # Gather log probabilities using matched indices
        log_probs = torch.log(
            self._cached_action_probs[
                torch.arange(batch_size, device=device), best_match_indices
            ]
            + 1e-8
        )

        return log_probs

    @property
    def action_mean(self):
        """this model is not compatiable with adaptive schedules"""
        return torch.zeros(1)

    @property
    def action_std(self):
        """this model is not compatiable with adaptive schedules"""
        return torch.zeros(1)

    @property
    def entropy(self):
        """
        Entropy of the action distribution.
        """
        if self._cached_action_probs is None:
            raise RuntimeError(
                "Action probabilities not computed. Call _ensure_forward_pass() first."
            )

        # Compute entropy directly from cached probabilities
        # entropy = -sum(p * log(p))
        log_probs = torch.log(
            self._cached_action_probs + 1e-8
        )  # Add small epsilon for numerical stability
        entropy = -(self._cached_action_probs * log_probs).sum(dim=-1)  # (batch,)
        return entropy

    def reset(self, dones=None):
        """
        Reset the cached values. Called between episodes.
        """
        self._cached_observations = None  # type: ignore
        self._cached_footstep_options = None  # type: ignore
        self._cached_action_values = None  # type: ignore
        self._cached_swing_durations = None  # type: ignore
        self._cached_action_probs = None  # type: ignore
        self._cached_selected_actions = None  # type: ignore
        self._cached_selected_log_probs = None  # type: ignore
        self.distribution = None  # type: ignore
