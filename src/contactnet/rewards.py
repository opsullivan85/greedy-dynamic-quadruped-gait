from unittest.util import three_way_cmp
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


def polygon_area_2d(vertices: torch.Tensor) -> torch.Tensor:
    """Calculate the area of 2D polygons using the Shoelace formula.
    
    Works for both convex and non-convex (simple) polygons.
    NOTE: this will not work for self-intersecting polygons.
    
    Args:
        vertices (torch.Tensor): Polygon vertices, shape (..., N, 2) where N is number of vertices
        
    Returns:
        torch.Tensor: Areas of the polygons, shape (...)
    """
    # Shift vertices to get pairs (x_i, y_i) and (x_{i+1}, y_{i+1})
    x = vertices[..., 0]  # (..., N)
    y = vertices[..., 1]  # (..., N)
    
    # Roll to get next vertices (cyclically)
    x_next = torch.roll(x, shifts=-1, dims=-1)
    y_next = torch.roll(y, shifts=-1, dims=-1)
    
    # Shoelace formula: 0.5 * |sum(x_i * y_{i+1} - x_{i+1} * y_i)|
    cross_terms = x * y_next - x_next * y
    area = 0.5 * torch.abs(torch.sum(cross_terms, dim=-1))
    
    return area


def distance_from_virtual_line(
    point: torch.Tensor, line_start: torch.Tensor, line_end: torch.Tensor
) -> torch.Tensor:
    """Calculate the shortest distance from a point to a line defined by two points in 2D.

    Args:
        point (torch.Tensor): The point to measure the distance from, shape (..., 2)
        line_start (torch.Tensor): One endpoint of the line segment, shape (..., 2)
        line_end (torch.Tensor): The other endpoint of the line segment, shape (..., 2)

    Returns:
        torch.Tensor: The shortest distance from the point to the line segment, shape (...)
    """
    p0 = point
    x0 = p0[:, 0]
    y0 = p0[:, 1]
    p1 = line_start
    x1 = p1[:, 0]
    y1 = p1[:, 1]
    p2 = line_end
    x2 = p2[:, 0]
    y2 = p2[:, 1]

    m_numerator = y2 - y1
    m_denominator = x2 - x1
    # Avoid division by zero, prevent branching
    m_denominator = torch.where(m_denominator == 0, torch.tensor(1e-8), m_denominator)
    m = m_numerator / m_denominator

    d_numerator = torch.abs(m * x0 - y0 + y2 - m * x2)
    d_denominator = torch.sqrt(m * m + 1)
    distance = d_numerator / d_denominator

    return distance


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
    # 4 contacts -> area = area of quadrilateral using Shoelace formula
    four_contacts = num_contacts == 4
    if torch.any(four_contacts):
        foot_positions_4 = foot_positions[four_contacts]  # (N, 4, 3)
        # Reorder feet to avoid self-intersecting polygon: [FL, FR, RL, RR] -> [FR, FL, RL, RR]
        foot_positions_4 = foot_positions_4[:, [1, 0, 2, 3]]  # (N, 4, 3)
        # Project to 2D (use x, y coordinates) for the Shoelace formula
        foot_positions_2d = foot_positions_4[..., :2]  # (N, 4, 2)
        areas[four_contacts] = polygon_area_2d(foot_positions_2d)

    return areas

def inscribed_circle_radius(
    env: ManagerBasedEnv,
) -> torch.Tensor:
    """Calculate the radius of the inscribed circle of the support polygon formed by the feet in contact with the ground.

    NOTE: this assumes that the COM is inside the support polygon. If the COM is outside the support polygon,
        the radius will be positive, and non-sensical.

    Args:
        env (ManagerBasedEnv): The environment.

    Returns:
        torch.Tensor: The radius of the inscribed circle of the support polygon for each environment.
    """
    contact_forces = env.scene["contact_forces"].data.net_forces_w
    foot_contacts = (
        contact_forces.norm(dim=2) > env.scene["contact_forces"].cfg.force_threshold
    )
    foot_positions = env.scene["robot"].data.body_pos_w[:, env.cfg.foot_indices]  # type: ignore
    com_position = env.scene["robot"].data.root_pos_w[:, :2]  # (num_envs, 2)
    num_contacts = foot_contacts.sum(dim=1)

    # Initialize radii to zero
    radii = torch.zeros((env.num_envs,), device=env.device)
    # use distance_from_virtual_line to calculate the radius of the inscribed circle
    three_contacts = num_contacts == 3
    if torch.any(three_contacts):
        foot_positions_3 = foot_positions[three_contacts]  # (N, 4, 3)
        foot_contacts_3 = foot_contacts[three_contacts]  # (N, 4)
        com_position_3 = com_position[three_contacts]  # (N, 2)
        # reshape is needed because torch implicitly flattens for some reason
        foot_contact_positions = foot_positions_3[foot_contacts_3].reshape(
            -1, 3, 3
        )  # (N, 3, 3)
        foot_contact_positions_2d = foot_contact_positions[:, :, :2]  # (N, 3, 2)
        distances = torch.stack([
            distance_from_virtual_line(com_position_3, foot_contact_positions_2d[:, 0], foot_contact_positions_2d[:, 1]),
            distance_from_virtual_line(com_position_3, foot_contact_positions_2d[:, 1], foot_contact_positions_2d[:, 2]),
            distance_from_virtual_line(com_position_3, foot_contact_positions_2d[:, 2], foot_contact_positions_2d[:, 0]),
        ], dim=1)  # (N, 3)
        radii[three_contacts] = torch.min(distances, dim=1).values
    
    four_contacts = num_contacts == 4
    if torch.any(four_contacts):
        foot_positions_4 = foot_positions[four_contacts]  # (N, 4, 3)
        com_position_4 = com_position[four_contacts]  # (N, 2)
        # Reorder feet to avoid self-intersecting polygon: [FL, FR, RL, RR] -> [FR, FL, RL, RR]
        foot_positions_4 = foot_positions_4[:, [1, 0, 2, 3]]  # (N, 4, 3)
        foot_positions_2d = foot_positions_4[..., :2]  # (N, 4, 2)
        distances = torch.stack([
            distance_from_virtual_line(com_position_4, foot_positions_2d[:, 0], foot_positions_2d[:, 1]),
            distance_from_virtual_line(com_position_4, foot_positions_2d[:, 1], foot_positions_2d[:, 2]),
            distance_from_virtual_line(com_position_4, foot_positions_2d[:, 2], foot_positions_2d[:, 3]),
            distance_from_virtual_line(com_position_4, foot_positions_2d[:, 3], foot_positions_2d[:, 0]),
        ], dim=1)  # (N, 4)
        radii[four_contacts] = torch.min(distances, dim=1).values

    return radii