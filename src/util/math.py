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
