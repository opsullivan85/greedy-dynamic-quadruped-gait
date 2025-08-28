from abc import ABC, abstractmethod
from nptyping import NDArray, Float32, Shape
from typing import Type, Generic, TypeVar
import numpy as np
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed


class RobotInterface(ABC):
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
                # TODO: what frames are each of these things in?
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


T = TypeVar('T', bound=RobotInterface)

class RobotInterfaceVect(Generic[T]):
    def __init__(self, dt: float, instances: int, cls: Type[T], **kwargs) -> None:
        self.instances = instances
        self.interfaces: list[T] = [
            cls(dt=dt, **kwargs) for _ in range(instances)
        ]
        self.pool = ThreadPoolExecutor()
    
    def get_torques(
        self,
        joint_states: NDArray[Shape["N, 4, 3, 2"], Float32],
        body_states: NDArray[Shape["N, 13"], Float32],
        commands: NDArray[Shape["N, 3"], Float32],
    ) -> NDArray[Shape["N, 4, 3"], Float32]:
        """Compute and return the joint torques for multiple instances based on the current states and commands.

        Args:
            joint_states (np.ndarray): (N,4,3,2) array of joint states for N instances:
                index 0: instance index (0-(N-1))
                index 1: leg index (0-3)
                index 2: joint index (0-2) (hip, upper leg, lower leg)
                index 3: state (0: position, 1: velocity)
            body_states (np.ndarray): (N,13) array of body states for N instances:
                position = [0:3]
                orientation (xyzw quaternion) = [3:7]
                velocity = [7:10]
                angular velocity = [10:13]
            commands (np.ndarray): (N,3) array of commands for N instances:
                x velocity = [0]
                y velocity = [1]
                yaw rate = [2]

        Returns:
            np.ndarray: (N,4,3) array of joint torques for N instances:
                index 0: instance index (0-(N-1))
                index 1: leg index (0-3)
                index 2: joint index (0-2) (hip, upper leg, lower leg)
        """
        assert joint_states.shape[0] == self.instances
        assert body_states.shape[0] == self.instances
        assert commands.shape[0] == self.instances

        torques = []
        futures = []
        
        # Submit tasks to the thread pool
        for i in range(self.instances):
            future = self.pool.submit(
                RobotInterfaceVect._get_torques_single,
                self.interfaces[i],
                joint_states[i],
                body_states[i],
                commands[i]
            )
            futures.append(future)
        
        # Collect results in order
        torques = [future.result() for future in futures]
        
        return np.stack(torques, axis=0)

    @staticmethod
    def _get_torques_single(
        interface: T,
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
        body_state: NDArray[Shape["13"], Float32],
        command: NDArray[Shape["3"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        """Helper method to compute torques for a single instance. This method is intended to be run in a separate thread.

        Args:
            interface (RobotInterface): The robot interface instance.
            joint_states (NDArray[Shape["4, 3, 2"], Float32]): Joint states for the instance.
            body_state (NDArray[Shape["13"], Float32]): Body state for the instance.
            command (NDArray[Shape["3"], Float32]): Command for the instance.

        Returns:
            NDArray[Shape["4, 3"], Float32]: Computed joint torques for the instance.
        """
        return interface.get_torques(joint_states=joint_states, body_state=body_state, command=command)
