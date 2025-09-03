import logging
from abc import ABC, abstractmethod

import numpy as np
from nptyping import Float32, NDArray, Shape

logger = logging.getLogger(__name__)


class Sim2RealInterface(ABC):
    @abstractmethod
    def __init__(self, dt: float) -> None:
        pass

    @abstractmethod
    def get_torques(
        self,
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
        body_state: NDArray[Shape["13"], Float32],
        command: NDArray[Shape["3"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        """Compute and return the joint torques based on the current state and command.

        Args:
            joint_states (np.ndarray): (4,3,2) array of joint states:
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
                index 2: state (0: position, 1: velocity)
            body_state (np.ndarray): (13,) array of body state:
                position = [0:3]
                orientation (xyzw quaternion) = [3:7]
                velocity = [7:10]
                angular velocity = [10:13]
            command (np.ndarray): (3,) array of command:
                x velocity = [0]
                y velocity = [1]
                yaw rate = [2]

        Returns:
            np.ndarray: (4,3) array of joint torques
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
        """
        pass

    @abstractmethod
    def initiate_footstep(
        self,
        leg: int,
        location_hip: NDArray[Shape["2"], Float32],
        duration: float,
    ):
        """initiates a footstep for the specified leg at the specified location

        Args:
            leg (int): Index of the leg (0-3)
            location_hip (NDArray[Shape["2"], Float32]): Desired foot position in the respective hip frame (x, y)
                This position is relative to the hip of the specified leg.
                z will be projected down to zero.
            duration (float): Duration of the footstep
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Resets the robot
        """
        pass
