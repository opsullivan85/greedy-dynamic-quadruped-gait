"""Observation term to generate footstep options as an isaaclab observation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from isaaclab.managers import (
    ObservationManager,
)

import src.constants as const
from src.contactnet.contactnet import CostMapGenerator
from src.gaitnet.actions.footstep_action import NO_STEP
from src.gaitnet.actions.mpc_action import ManagerBasedEnv
from src.gaitnet.env_cfg.observations import get_terrain_mask
from src.simulation.cfg.footstep_scanner_constants import idx_to_xy
from src.util.math import seeded_uniform_noise
from src.contactnet.debug import view_footstep_cost_map
from src import get_logger

logger = get_logger()


_debug_footstep_cost_map = False
_debug_footstep_cost_map_all = False


class FootstepOptionGenerator:
    def __init__(self, env: ManagerBasedEnv, options_per_leg: int):
        self.env = env
        self.cost_map_generator = CostMapGenerator(device=env.device)
        self.options_per_leg = options_per_leg

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
        terrain_mask = get_terrain_mask(
            const.gait_net.valid_height_range, obs
        )  # (num_envs, 4, H, W)
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
        cost_map: torch.Tensor, options_per_leg: int
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
            flat_cost_map, options_per_leg, largest=False, sorted=True
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
        cost_map: torch.Tensor, options_per_leg: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the best footstep options for each leg.

        includes a NO_STEP option at the end.

        Args:
            cost_map: (num_envs, 4, H, W) filtered cost maps for each leg

        Returns:
            best_options: (num_envs, num_options, 3) best options as (leg_idx, dx, dy)
            best_values: (num_envs, num_options) corresponding costs
        """

        best_options = []
        best_values = []
        for leg in range(cost_map.shape[1]):  # iterate over legs
            leg_cost_map = cost_map[:, leg, :, :].unsqueeze(1)  # (num_envs, 1, H, W)
            topk_pos, topk_values = FootstepOptionGenerator._overall_best_options(
                leg_cost_map, options_per_leg
            )
            # manually set leg index
            topk_pos[:, :, 0] = leg
            best_options.append(topk_pos)
            best_values.append(topk_values)

        # add in the NO_STEP option
        num_envs = cost_map.shape[0]
        no_step_option = torch.zeros(
            (num_envs, 1, 3),
            dtype=torch.float32,
            device=cost_map.device,
        )
        no_step_option[:, 0, 0] = NO_STEP
        no_step_value = torch.zeros(
            (num_envs, 1),
            dtype=torch.float32,
            device=cost_map.device,
        )
        best_options.append(no_step_option)
        best_values.append(no_step_value)

        best_options = torch.cat(best_options, dim=1)  # (num_envs, num_options, 3)
        best_values = torch.cat(best_values, dim=1)  # (num_envs, num_options)

        return best_options, best_values

    def _apply_options_to_cost_map(
        self, cost_map: torch.Tensor, options: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply the selected footstep options to the cost map.

        Args:
            cost_map: (num_envs, 4, H, W) filtered cost maps for each leg
            options: (num_envs, num_options, 3) footstep options as (leg_idx, dx, dy)

        Returns:
            torch.Tensor: Updated cost map with applied footstep options.
        """
        zero = idx_to_xy(torch.tensor([0, 0], device=options.device))
        one = idx_to_xy(torch.tensor([1, 1], device=options.device))
        slope = (one[0] - zero[0], one[1] - zero[1])
        intercept = (zero[0], zero[1])
        # Apply each footstep option to the cost map
        for leg_idx, dx, dy in options[0, :-1]:  # exclude the NO_STEP option
            idx = torch.round((dx - intercept[0]) / slope[0]).long()
            idy = torch.round((dy - intercept[1]) / slope[1]).long()
            cost_map[0, int(leg_idx), idx, idy] = -0.3  # Set cost for selected options

        return cost_map

    def get_footstep_options(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Generate footstep options based on the current environment state.

        Returns:
            torch.Tensor: Footstep options of shape (num_envs, options_per_leg*4, 4)
                          Each option is represented as (leg_index, x_offset, y_offset, cost)
                          appends a NO_STEP option at the end of the list.
        """
        contactnet_obs = obs[:, :18]
        with torch.inference_mode():
            cost_maps = self.cost_map_generator.predict(
                contactnet_obs
            )  # (num_envs, 4, H, W)

        # switch from (FL, FR, RL, RR) to (FR, FL, RR, RL)
        cost_maps = cost_maps[:, [1, 0, 3, 2], :, :]

        if _debug_footstep_cost_map_all:
            view_footstep_cost_map(
                cost_map=cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Default Footstep Cost Map",
                save_figure=True,
                show_ticks=False,
            )

        # interpolate cost map
        cost_maps = nn.functional.interpolate(
            cost_maps,  # (num_envs, 4, h, w) 4 is being used as the channel dimension here
            size=const.footstep_scanner.grid_size.tolist(),  # (H, W)
            mode="bilinear",
            align_corners=True,
        ).squeeze(
            1
        )  # (num_envs, 4, H, W)

        if _debug_footstep_cost_map_all:
            view_footstep_cost_map(
                cost_map=cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Scaled Footstep Cost Map",
                save_figure=True,
                show_ticks=False,
            )

        # apply determanistic noise to costmap to slightly spread apart the best options
        # it is important that this is reproducable for the RL algorithm
        noise = seeded_uniform_noise(cost_maps, cost_maps.shape[1:])
        noise = (
            noise * 2 * const.gait_net.upscale_costmap_noise
            - const.gait_net.upscale_costmap_noise
        )
        cost_maps += noise

        if _debug_footstep_cost_map_all:
            view_footstep_cost_map(
                cost_map=cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Noisy Footstep Cost Map",
                save_figure=True,
                show_ticks=False,
            )

        masked_cost_maps = self._filter_cost_map(cost_maps, obs)  # (num_envs, 4, H, W)

        if _debug_footstep_cost_map_all:
            view_footstep_cost_map(
                masked_cost_maps[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Masked Footstep Cost Map",
                save_figure=True,
                # show_ticks=False,
                tick_skip=4,
                tick_rotations=(90, 0),
            )

        # best_options, best_values = self._overall_best_options(masked_cost_maps, self.num_options)
        best_options, best_values = self._best_options_per_leg(
            masked_cost_maps, self.options_per_leg
        )

        if _debug_footstep_cost_map:
            applied_map = self._apply_options_to_cost_map(
                masked_cost_maps, best_options
            )
            view_footstep_cost_map(
                applied_map[0][[1, 0, 3, 2]].cpu().numpy(),
                title="Applied Footstep Cost Map",
                save_figure=True,
                # show_ticks=False,
                tick_skip=4,
                tick_rotations=(90, 0),
            )

        # replace any options with inf cost with NO_STEP option and 0 cost
        inf_mask = torch.isinf(best_values)  # (num_envs, num_options)
        if inf_mask.any():
            # Use unsqueeze to broadcast the mask to match best_options shape
            inf_mask_expanded = inf_mask.unsqueeze(-1)  # (num_envs, num_options, 1)
            best_options[:, :, 0] = torch.where(
                inf_mask,
                torch.tensor(
                    NO_STEP, device=best_options.device, dtype=best_options.dtype
                ),
                best_options[:, :, 0],
            )
            best_options[:, :, 1:3] = torch.where(
                inf_mask_expanded.expand(-1, -1, 2),
                torch.tensor(0.0, device=best_options.device, dtype=best_options.dtype),
                best_options[:, :, 1:3],
            )
            best_values[inf_mask] = 0.0

        result = torch.cat([best_options, best_values.unsqueeze(-1)], dim=-1)
        # (num_envs, num_options, 4) where each option is (leg, dx, dy, cost)

        # shuffle the options to prevent any bias from ordering
        # this is important because the policy might learn to ignore certain indices
        # shuffle_indices = torch.randperm(result.shape[1], device=result.device)
        # result = result[:, shuffle_indices, :]
        return result


class FootstepObservationManager(ObservationManager):
    """Assumes the 4 footstep scanner values are at the end of the observation.

    Replaces the footstep scanner values with footstep options generated from the cost map.
    """

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        self.logged_update_history_warning = False
        self.footstep_option_generator = FootstepOptionGenerator(env, options_per_leg=const.gait_net.num_footstep_options)
        super().__init__(cfg, env)
        self.footstep_options: torch.Tensor
        """Stores the latest footstep options generated by the manager (num_envs, options_per_leg*4, 4).
        Each option is represented as (leg_index, x_offset, y_offset, cost)."""
        self._footstep_actions: torch.Tensor
        self._overwrite_obs_dim()

    @property
    def footstep_actions(self) -> torch.Tensor:
        """Get the latest footstep actions with durations from the policy.

        Returns:
            torch.Tensor: Footstep actions of shape (num_envs, options_per_leg*4+1, 4)
                          Each action is represented as (leg_index, x_offset, y_offset, duration).
        """
        # If durations have been set by the policy, use them
        if hasattr(self, "_footstep_actions") and self._footstep_actions is not None:
            return self._footstep_actions

        # Otherwise, return options with default durations for initialization
        durations = torch.full(
            (self.footstep_options.shape[0], self.footstep_options.shape[1], 1),
            0.2,
            device=self.footstep_options.device,
        )
        _footstep_actions = self.footstep_options.clone()
        _footstep_actions[:, :, 3] = durations.squeeze(-1)
        return _footstep_actions

    def set_footstep_actions(
        self, action_indices: torch.Tensor, durations: torch.Tensor
    ) -> None:
        """Set the footstep actions for the selected action indices.

        This is called after action selection to attach durations to the selected options.

        Args:
            action_indices (torch.Tensor): Selected action indices of shape (num_envs,) or (num_envs, 1).
            durations (torch.Tensor): Footstep durations for selected actions, shape (num_envs,).
        """
        # Create full action tensor from options
        self._footstep_actions = self.footstep_options.clone()

        # Flatten action_indices if it has shape (num_envs, 1)
        if action_indices.dim() > 1:
            action_indices = action_indices.squeeze(-1)

        # Set durations for the selected options
        batch_size = action_indices.shape[0]
        batch_indices = torch.arange(batch_size, device=action_indices.device)

        # Update durations at the selected indices
        self._footstep_actions[batch_indices, action_indices, 3] = durations

    def _overwrite_obs_dim(self) -> None:
        """Overwrite the observation dimensions to account for footstep options."""
        self._group_obs_dim
        policy_obs_dim = self._group_obs_dim["policy"][0]
        # if this is a list throw an error
        if isinstance(policy_obs_dim, tuple):
            raise NotImplementedError(
                "FootstepObservationManager does not support list observation dimensions."
            )
        obs_dim: int = policy_obs_dim
        obs_dim -= const.footstep_scanner.total_robot_features

        # add in the footstep options
        # TODO: pull this from the footstep_controller action
        footstep_option_size = 8  # (leg_one_hot (5), dx, dy, cost)
        # +1 is for the NO_STEP option
        obs_dim += (
            const.gait_net.num_footstep_options * const.robot.num_legs + 1
        ) * footstep_option_size
        self._group_obs_dim["policy"] = (obs_dim,)

    @staticmethod
    def footstep_options_to_one_hot(options: torch.Tensor) -> torch.Tensor:
        """Convert footstep options to one-hot encoding.

        Args:
            options (torch.Tensor): Footstep options of shape (num_envs, options_per_leg*4, 4)
                                    Each option is represented as (leg_index, x_offset, y_offset, cost).

        Returns:
            torch.Tensor: One-hot encoded footstep options of shape (num_envs, options_per_leg*4, embedding_dim)
                where the no-op option is the first index in the one-hot encoding.
        """
        num_envs, num_options, _ = options.shape
        embedding_dim = const.robot.num_legs + 1
        # +1 for no-step option to remap everything from -1-3 to 0-4
        one_hot = F.one_hot(options[:, :, 0].long() + 1, num_classes=embedding_dim)
        # replace leg index with one-hot encoding
        options_one_hot = torch.cat(
            [one_hot.float(), options[:, :, 1:]], dim=-1
        )  # (num_envs, num_options, 8)
        return options_one_hot

    def _modify_obs(self, obs: torch.Tensor) -> torch.Tensor:
        self.footstep_options = self.footstep_option_generator.get_footstep_options(obs)
        # (num_envs, options_per_leg*4, 4) where each option is (leg_index, x_offset, y_offset, cost)

        one_hot_options = self.footstep_options_to_one_hot(self.footstep_options)

        # Safety check: ensure no inf/nan values in observations
        if torch.any(torch.isinf(self.footstep_options)) or torch.any(
            torch.isnan(self.footstep_options)
        ):
            logger.error(
                "Found inf/nan in footstep options! This will cause training instability."
            )
            logger.error(
                f"Inf count: {torch.isinf(self.footstep_options).sum().item()}"
            )
            logger.error(
                f"NaN count: {torch.isnan(self.footstep_options).sum().item()}"
            )

        # replace the footstep scanner values at the end of the observation with the flattened footstep options
        obs = obs[:, : -const.footstep_scanner.total_robot_features]
        obs = torch.cat([obs, one_hot_options.flatten(start_dim=1)], dim=1)
        return obs

    def compute_group(
        self, group_name: str, update_history: bool = False
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        obs = super().compute_group(group_name, update_history)
        # only modify the policy observation group
        if group_name != "policy":
            return obs

        if isinstance(obs, dict):
            raise NotImplementedError(
                "FootstepObservationManager does not support dict observations."
            )

        # the history will not be in the same shape as the observations
        if update_history:
            if not self.logged_update_history_warning:
                logger.warning(
                    "FootstepObservationManager history might not work as expected."
                )
                self.logged_update_history_warning = True

        obs = self._modify_obs(obs)
        return obs
