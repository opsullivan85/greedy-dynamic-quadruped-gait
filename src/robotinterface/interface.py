from abc import ABC, abstractmethod
from nptyping import NDArray, Float32, Shape
from typing import Any, Type, Generic, TypeVar
import numpy as np
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from concurrent.futures import ThreadPoolExecutor
import traceback
import enum


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


class Message:
    class _ManagerMessages(enum.Enum):
        PING = enum.auto()
        GET_TORQUES = enum.auto()
        SHUTDOWN = enum.auto()
        RESET = enum.auto()

    class _WorkerMessages(enum.Enum):
        PONG = enum.auto()
        SUCCESS = enum.auto()
        EXCEPTION = enum.auto()

    Manager = _ManagerMessages
    Worker = _WorkerMessages


T = TypeVar("T", bound=RobotInterface)


class RobotInterfaceVect(Generic[T]):
    """Provides a vectorized wrapper around a RobotInterface using multiprocessing.

    Each RobotInterface instance runs in its own dedicated process to handle CPU-bound
    computations in parallel while maintaining state persistence across calls.

    Args:
        dt: Time step for robot interfaces
        instances: Number of robot interface instances to create
        cls: The RobotInterface class to instantiate
        **kwargs: Additional arguments passed to RobotInterface constructor
    """

    def __init__(self, dt: float, instances: int, cls: Type[T], **kwargs) -> None:
        self.instances = instances
        self.cls = cls
        self.dt = dt
        self.kwargs = kwargs

        self.workers: list[Process] = []
        self.pipes: list[Connection] = []


        # Verify all workers started successfully
        self._setup_workers(dt, instances, cls, **kwargs)

    def _setup_workers(self, dt, instances: int, cls: Type[T], **kwargs) -> None:
        """Verify all worker processes initialized successfully."""
        for i in range(instances):
            parent_conn, child_conn = Pipe()
            worker = Process(
                target=self._worker_loop, args=(child_conn, i, cls, dt, kwargs)
            )
            worker.start()

            self.workers.append(worker)
            self.pipes.append(parent_conn)

        for pipe in self.pipes:
            try:
                # Send initialization check
                pipe.send((Message.Manager.PING, None))
                response_type, result = pipe.recv()
                if response_type != Message.Worker.PONG:
                    raise RuntimeError(
                        f"Worker failed to initialize: {result}"
                    )
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to verify worker readiness: {e}")

    @staticmethod
    def _worker_loop(
        conn: Connection, worker_id: int, cls: Type[T], dt: float, kwargs: dict
    ) -> None:
        """Main loop for worker process. Creates and maintains a RobotInterface instance."""
        try:
            interface = cls(dt=dt, **kwargs)

            # Main processing loop
            while True:
                try:
                    command, data = conn.recv()

                    if command == Message.Manager.PING:
                        conn.send((Message.Worker.PONG, None))
                    elif command == Message.Manager.GET_TORQUES:
                        joint_states, body_state, command_input = data
                        torques = interface.get_torques(
                            joint_states=joint_states,
                            body_state=body_state,
                            command=command_input,
                        )
                        conn.send((Message.Worker.SUCCESS, torques))
                    elif command == Message.Manager.SHUTDOWN:
                        conn.send((Message.Worker.SUCCESS, None))
                        break
                    else:
                        conn.send((Message.Worker.EXCEPTION, f"Unknown command: {command}"))

                except Exception as e:
                    # Send error back to main process
                    error_msg = (
                        f"Worker {worker_id} error: {str(e)}\n{traceback.format_exc()}"
                    )
                    conn.send((Message.Worker.EXCEPTION, error_msg))

        except Exception as e:
            # Initialization failed
            error_msg = f"Worker {worker_id} initialization failed: {str(e)}\n{traceback.format_exc()}"
            conn.send((Message.Worker.EXCEPTION, error_msg))
        finally:
            conn.close()

    @staticmethod
    def _expect_success(pipe: Connection) -> Any:
        """Helper method that returns the value when a pipe returns sucess
        throws a runtime error otherwise

        Args:
            pipe (Connection): The pipe to communicate with

        Raises:
            RuntimeError: Whever something goes wrong

        Returns:
            Any: The result result of the pipe's communciation
        """
        try:
            response_type, result = pipe.recv()
            if response_type == Message.Worker.SUCCESS:
                return result
            elif response_type == Message.Worker.EXCEPTION:
                raise RuntimeError(f"Worker computation failed: {result}")
            else:
                raise RuntimeError(f"Worker sent unexpected response: {response_type}")
        except Exception as e:
            raise RuntimeError(f"Failed to receive result from worker: {e}")

    def get_torques(
        self,
        joint_states: NDArray[Shape["N, 4, 3, 2"], Float32],
        body_states: NDArray[Shape["N, 13"], Float32],
        commands: NDArray[Shape["N, 3"], Float32],
    ) -> NDArray[Shape["N, 4, 3"], Float32]:
        """Compute and return the joint torques for multiple instances based on the current states and commands.

        Args:
            joint_states: (N,4,3,2) array of joint states for N instances
            body_states: (N,13) array of body states for N instances
            commands: (N,3) array of commands for N instances

        Returns:
            (N,4,3) array of joint torques for N instances
        """
        # Validate input dimensions
        assert (
            joint_states.shape[0] == self.instances
        ), f"Expected {self.instances} instances, got {joint_states.shape[0]}"
        assert (
            body_states.shape[0] == self.instances
        ), f"Expected {self.instances} instances, got {body_states.shape[0]}"
        assert (
            commands.shape[0] == self.instances
        ), f"Expected {self.instances} instances, got {commands.shape[0]}"

        # Send computation requests to all workers
        for i in range(self.instances):
            try:
                self.pipes[i].send(
                    (
                        Message.Manager.GET_TORQUES,
                        (joint_states[i], body_states[i], commands[i]),
                    )
                )
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to send data to worker: {e}")

        # Collect results from all workers
        torques_list = []
        for i in range(self.instances):
            torques = RobotInterfaceVect._expect_success(self.pipes[i])
            torques_list.append(torques)

        return np.stack(torques_list, axis=0)

    def _cleanup(self) -> None:
        """Clean up worker processes and pipes."""
        # Send shutdown signal to all responsive workers
        for pipe in self.pipes:
            try:
                if pipe is not None:
                    pipe.send((Message.Manager.SHUTDOWN, None))
                    pipe.recv()
                    pipe.close()
            except:
                pass

        for worker in self.workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=0.1)
                if worker.is_alive():
                    worker.kill()  # Force kill if terminate didn't work

        self.workers.clear()
        self.pipes.clear()

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        self._cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self._cleanup()
