from isaaclab.envs import ManagerBasedEnv
import torch
from src.sim2real import SimInterface
from src.util import VectorPool
from nptyping import NDArray, Shape, Bool


def controller_real_swing_error(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Error based on the difference between controller swing time and actual swing time.

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: The error in seconds between controller swing time and actual swing time.
    """
    incomplete_swing_multiplier = 2.0  # penalize incomplete swings more

    controllers: VectorPool[SimInterface] = env.cfg.controllers  # type: ignore
    controller_swing_durations = controllers.call(
        SimInterface.get_swing_durations,
        mask=None,
    )
    controller_swing_durations = torch.tensor(
        controller_swing_durations, device=env.device
    )
    # (num_envs, 4, 1) -> (num_envs, 4)
    controller_swing_durations = controller_swing_durations.squeeze(2)

    contact_forces = env.scene["contact_forces"].data.net_forces_w
    # robot is done if all feet have contact force above threshold
    foot_contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    swing_complete = torch.all(foot_contacts, dim=1)
    # if the swing is complete take the last air time as the real swing duration
    # else take the current air time as the real swing duration
    real_swing_durations = torch.where(
        swing_complete.unsqueeze(1),
        env.scene["contact_forces"].data.last_air_time,
        env.scene["contact_forces"].data.current_air_time,
    )

    error = torch.abs(controller_swing_durations - real_swing_durations)
    error[~swing_complete] *= incomplete_swing_multiplier
    return error.sum(dim=1)  # sum over legs


def triangle_area(
    vertex1: torch.Tensor, vertex2: torch.Tensor, vertex3: torch.Tensor
) -> torch.Tensor:
    """Calculate the area of triangles using the cross product method.

    This function mimics the interface of tensorflow_graphics.geometry.representation.triangle.area().

    Args:
        vertex1 (torch.Tensor): First vertices of triangles, shape (..., 3)
        vertex2 (torch.Tensor): Second vertices of triangles, shape (..., 3)
        vertex3 (torch.Tensor): Third vertices of triangles, shape (..., 3)

    Returns:
        torch.Tensor: Areas of the triangles, shape (...)
    """
    # Calculate vectors from vertex1 to vertex2 and vertex1 to vertex3
    v1 = vertex2 - vertex1  # (..., 3)
    v2 = vertex3 - vertex1  # (..., 3)

    # Calculate cross product
    cross_product = torch.cross(v1, v2, dim=-1)  # (..., 3)

    # Calculate magnitude of cross product
    cross_magnitude = torch.norm(cross_product, dim=-1)  # (...)

    # Area is half the magnitude of the cross product
    area = 0.5 * cross_magnitude

    return area


def support_polygon_area(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Calculate the area of the support polygon formed by the feet in contact with the ground.

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: The area of the support polygon for each environment.
    """
    # use triangel.area to calculate the area of the support polygon
    # first check for which feet are in contact
    # if less than 3 feet are in contact, area is 0
    # if 3 feet are in contact, area is the area of the triangle formed by the feet
    # if 4 feet are in contact, area is the sum of the areas of the two triangles formed by the feet
    contact_forces = env.scene["contact_forces"].data.net_forces_w
    foot_contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    foot_positions = env.scene["robot"].data.body_pos_w[:, env.cfg.foot_indices]  # type: ignore
    num_contacts = foot_contacts.sum(dim=1)

    areas = torch.zeros((env.num_envs,), device=env.device)
    # 3 contacts -> area = area of triangle
    three_contacts = num_contacts == 3
    if torch.any(three_contacts):
        foot_positions_3 = foot_positions[three_contacts]  # (N, 4, 3)
        foot_contacts_3 = foot_contacts[three_contacts]  # (N, 4)
        # reshape is needed because torch implicitly flattens for some reason
        foot_contact_positions = foot_positions_3[foot_contacts_3].reshape(
            -1, 3, 3
        )  # (N, 3, 3)
        areas[three_contacts] = triangle_area(
            foot_contact_positions[:, 0],
            foot_contact_positions[:, 1],
            foot_contact_positions[:, 2],
        )
    # 4 contacts -> area = area of two triangles
    four_contacts = num_contacts == 4
    if torch.any(four_contacts):
        foot_positions_4 = foot_positions[four_contacts]
        # note the important ordering of indices for the triangles here
        # foot ordering is [FL, FR, RL, RR],
        # so the triangles need to be (FL, FR, RL) and (FR, RL, RR)
        # as to not have intersecting triangles
        areas[four_contacts] = triangle_area(
            foot_positions_4[:, 0],
            foot_positions_4[:, 1],
            foot_positions_4[:, 2],
        ) + triangle_area(  # type: ignore
            foot_positions_4[:, 1],
            foot_positions_4[:, 2],
            foot_positions_4[:, 3],
        )

    return areas
