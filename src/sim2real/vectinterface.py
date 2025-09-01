import inspect
import logging
import traceback
from multiprocessing import Pipe, Process, cpu_count
from multiprocessing.connection import Connection
from typing import Any, Callable, Generic, Type, TypeVar

import numpy as np
from nptyping import Float32, NDArray, Shape, Bool, Number

from src.sim2real.abstractinterface import Sim2RealInterface

import enum

logger = logging.getLogger(__name__)


class Message:
    class _ManagerMessages(enum.Enum):
        PING = enum.auto()
        CALL_FUNCTION = enum.auto()
        SHUTDOWN = enum.auto()

    class _WorkerMessages(enum.Enum):
        PONG = enum.auto()
        SUCCESS = enum.auto()
        EXCEPTION = enum.auto()

    Manager = _ManagerMessages
    Worker = _WorkerMessages


T = TypeVar("T", bound=Sim2RealInterface)


class VectSim2Real(Generic[T]):
    """Provides a vectorized wrapper around a RobotInterface using multiprocessing.

    Robot interfaces are distributed across worker processes to handle CPU-bound
    computations in parallel while maintaining state persistence across calls.

    Function signatures are almost exactly the same as those from Sim2RealInterface.
    just look there for documentation
    """

    def __init__(
        self,
        dt: float,
        instances: int,
        cls: Type[T],
        num_workers: None | int = None,
        **kwargs,
    ) -> None:
        """

        Args:
            dt: Time step for robot interfaces
            instances: Number of robot interface instances to create
            cls: The RobotInterface class to instantiate
            num_workers: Number of worker processes (default: cpu_count())
                automatically set to the number of instances if that is lower
                than the desired number of workers
            **kwargs: Additional arguments passed to RobotInterface constructor
        """
        self.instances = instances
        self.num_workers = min(num_workers or cpu_count(), instances)

        self.workers: list[Process] = []
        self.pipes: list[Connection] = []

        # Setup workers
        self._setup_workers(dt, cls, **kwargs)

    def _batch_data(self, data: np.ndarray) -> list[np.ndarray]:
        """Batches data to be sent to the workers"""
        return np.array_split(data, self.num_workers)

    def _setup_workers(self, dt: float, cls: Type[T], **kwargs) -> None:
        """Setup worker processes and verify they initialized successfully."""
        robot_assignments = self._batch_data(np.arange(self.instances))
        # figure out how many robots each worker has
        worker_robots = [assignment.shape[0] for assignment in robot_assignments]

        for worker_id, num_robots in zip(range(self.num_workers), worker_robots):
            parent_conn, child_conn = Pipe()

            worker = Process(
                target=self._worker_loop,
                args=(child_conn, worker_id, cls, dt, num_robots, kwargs),
            )
            worker.start()

            self.workers.append(worker)
            self.pipes.append(parent_conn)

        # Verify all workers started successfully
        for pipe in self.pipes:
            try:
                pipe.send((Message.Manager.PING, None))
                response_type, result = pipe.recv()
                if response_type != Message.Worker.PONG:
                    raise RuntimeError(f"Worker failed to initialize: {result}")
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to verify worker readiness: {e}")

    def _verify_workers_ready(self) -> None:
        """Verify all worker processes initialized successfully."""
        for pipe in self.pipes:
            try:
                # Send initialization check
                pipe.send((Message.Manager.PING, None))
                response_type, result = pipe.recv()
                if response_type != Message.Worker.PONG:
                    raise RuntimeError(f"Worker failed to initialize: {result}")
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to verify worker readiness: {e}")

    @staticmethod
    def _worker_loop(
        conn: Connection,
        worker_id: int,
        cls: Type[T],
        dt: float,
        num_robots: int,
        kwargs: dict,
    ) -> None:
        """Main loop for worker process. Creates and maintains multiple RobotInterface instances."""
        try:
            # Initialize all robot interfaces for this worker
            interfaces = [cls(dt=dt, **kwargs) for _ in range(num_robots)]

            logger.debug(f"started worker id {worker_id} with {num_robots} robots")

            # Main processing loop
            while True:
                try:
                    command, data = conn.recv()

                    if command == Message.Manager.PING:
                        conn.send((Message.Worker.PONG, None))

                    elif command == Message.Manager.CALL_FUNCTION:
                        function_name, mask, batch_args = data

                        # Process each robot in this worker's batch
                        results_list = []
                        for i in range(len(interfaces)):
                            if not mask[i]:
                                # nan here still allows the output array to be typed
                                results_list.append(np.nan)
                            function = getattr(interfaces[i], function_name)
                            args = [batch_arg[i] for batch_arg in batch_args]
                            result = function(*args)
                            results_list.append(result)

                        # Stack results for this worker's batch
                        batch_results = np.stack(results_list, axis=0)
                        conn.send((Message.Worker.SUCCESS, batch_results))

                    elif command == Message.Manager.SHUTDOWN:
                        conn.send((Message.Worker.SUCCESS, None))
                        break

                    else:
                        conn.send(
                            (Message.Worker.EXCEPTION, f"Unknown command: {command}")
                        )

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

    def _call_function(
        self,
        function: Callable,
        mask: None | NDArray[Shape["*"], Bool],
        **kwargs: NDArray[Shape["*, ..."], Number],
    ) -> np.ndarray:
        """Calls a function on all of the underlying interfaces

        Args:
            function (Callable): function to call (taken from the Sim2RealInterface)
            kwargs (np.ndarray): passed into the function. Expected to match function signature
                except the type will be wrapped in a np.ndarray. kwargs verified at runtime.

        Returns:
            np.ndarray: a numpy array of results
        """
        # validate kwargs against the function signature
        sig = inspect.signature(function)
        try:
            sig.bind_partial(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Invalid arguments for Sim2RealInterface::{function.__name__}: {e}"
            )

        # validate kwarg shapes
        for kw, arg in kwargs.items():
            assert (
                arg.shape[0] == self.instances
            ), f"Expected {self.instances} rows for {kw}, got {arg.shape[0]}"
        if mask is not None:
            assert (
                mask.shape[0] == self.instances
            ), f"Expected {self.instances} rows for mask, got {mask.shape[0]}"
        else:
            mask = np.full((self.instances,), True, dtype=bool)

        all_batched_args = [self._batch_data(arg) for arg in kwargs.values()]
        all_batched_mask = self._batch_data(mask)
        function_name = function.__name__

        # send function calls
        for pipe, batched_mask, batched_args in zip(
            self.pipes, all_batched_mask, zip(*all_batched_args)
        ):
            try:
                pipe_args = (function_name, batched_mask, batched_args)
                pipe.send((Message.Manager.CALL_FUNCTION, pipe_args))
            except Exception as e:
                self._cleanup()
                raise RuntimeError(f"Failed to send data to worker: {e}")

        # gather results
        results = []
        for pipe in self.pipes:
            batched_torques = VectSim2Real._expect_success(pipe)
            results.append(batched_torques)

        return np.concatenate(results)

    def get_torques(
        self,
        joint_states: NDArray[Shape["*, 4, 3, 2"], Float32],
        body_state: NDArray[Shape["*, 13"], Float32],
        command: NDArray[Shape["*, 3"], Float32],
        mask: None | NDArray[Shape["*"], Bool] = None,  # type: ignore
    ) -> NDArray[Shape["*, 4, 3"], Float32]:
        return self._call_function(
            function=Sim2RealInterface.get_torques,
            mask=mask,
            joint_states=joint_states,
            body_state=body_state,
            command=command,
        )

    def reset(
        self,
        mask: None | NDArray[Shape["*"], Bool] = None,
    ) -> NDArray[Shape["*"], Number]:
        return self._call_function(
            function=Sim2RealInterface.reset,
            mask=mask,
        )

    def _cleanup(self) -> None:
        """Clean up worker processes and pipes."""
        # Send shutdown signal to all responsive workers
        for pipe in self.pipes:
            try:
                if pipe is not None:
                    pipe.send((Message.Manager.SHUTDOWN, None))
                    pipe.recv()  # Wait for acknowledgment
                    pipe.close()
            except:
                pass  # Worker may already be dead

        # Terminate any remaining worker processes
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
