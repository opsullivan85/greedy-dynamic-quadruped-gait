import numpy as np
import torch


# https://stackoverflow.com/a/56207565
def _quat_to_euler(w, x, y, z, backend) -> tuple:
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = backend.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = backend.clip(t2, -1.0, 1.0)
    Y = backend.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = backend.arctan2(t3, t4)

    return X, Y, Z


def quat_to_euler_np(
    w: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts quaternion (w, x, y, z) to euler angles (X, Y, Z)
    in radians

    Args:
        w: Scalar component of the quaternion.
        x: X component of the quaternion.
        y: Y component of the quaternion.
        z: Z component of the quaternion.

    Returns:
        A tuple of three numpy arrays representing the Euler angles (X, Y, Z) in radians.
    """
    return _quat_to_euler(w, x, y, z, backend=np)


def quat_to_euler_torch(
    w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Converts quaternion (w, x, y, z) to euler angles (X, Y, Z)
    in radians

    Args:
        w: Scalar component of the quaternion.
        x: X component of the quaternion.
        y: Y component of the quaternion.
        z: Z component of the quaternion.

    Returns:
        A tuple of three tensors representing the Euler angles (X, Y, Z) in radians.
    """
    return _quat_to_euler(w, x, y, z, backend=torch)


def seeded_uniform_noise(seed: torch.Tensor, shape: tuple) -> torch.Tensor:
    """Generate fast, deterministic pseudo-random noise for each seed row.

    Chat's implementation of a 32 bit linear congruential generator (LCG) seeded
    with a tensor.

    Args:
        seed (torch.Tensor): Input tensor of shape (num_envs, obs_dim).
        shape (tuple, optional): Desired output shape, without batch dimension.

    Returns:
        torch.Tensor: Pseudo-random noise tensor of shape (num_envs, *shape) on the same device as seed.
    """

    num_envs = seed.shape[0]
    flat_size = int(torch.prod(torch.tensor(shape)))

    seeds = (seed * 1e4).to(torch.int32).sum(dim=1)  # (num_envs,)
    seeds = torch.remainder(seeds, 2**31)  # ensure 32-bit

    a, c, m = 1664525, 1013904223, 2**32

    idx = torch.arange(flat_size, device=seed.device).unsqueeze(0)  # (1, flat_size)
    seeds = seeds.unsqueeze(1)  # (num_envs, 1)

    rand_ints = (a * (seeds + idx) + c) % m
    noise = rand_ints.float() / m  # normalize to [0,1)

    return noise.view((num_envs,) + shape)
