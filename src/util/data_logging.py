import logging
from pathlib import Path
from typing import Any, TypeAlias
import numpy as np
import torch
from src import timestamp, PROJECT_ROOT
import atexit
from time import time

logger = logging.getLogger(__name__)

DataType: TypeAlias = dict[str, torch.Tensor]
"""Data format to be logged.

Assumed to be {feature_name: tensor} where tensor shape is (num_robots, feature_dim).
feature_dim can be any shape, but num_robots must be the same for all features.

The logger will log data for each robot separately, resulting in a final shape of
(num_steps * num_robots, feature_dim) for each feature.
"""


class _DataLogger:
    def __init__(self, path: Path, save_interval: int = 1):
        self.path = path
        self.save_interval = save_interval
        self.step = 0
        self.data: dict[str, list[torch.Tensor]] = {}
        self.start_time = time()
        self.batches_logged = 0
        self.metadata: dict[str, Any] = {}
        self._sim_dt = None
        self._control_dt = None
        self._mpc_dt = None

    def log(self, data: DataType):
        """Log data for each robot.

        Args:
            data (DataType): Data to log.
        """
        if not self.data:
            self._initial_setup(data)

        for key, value in data.items():
            if key not in self.data:
                logger.warning(f"New key detected: {key}. Initializing storage.")
                self.metadata["feature_keys"].append(key)
                self.data[key] = []

            # Assume value shape is (num_robots, feature_dim)
            # We log data for each robot separately
            for robot in value.cpu():
                self.data[key].append(robot)

        self.batches_logged += 1
        self._try_save()

    def _try_save(self):
        self.step += 1
        if self.step % self.save_interval == 0:
            self.flush()

    def _initial_setup(self, data: DataType):
        self.start_time = time()
        self.data = {key: [] for key in data.keys()}
        self.metadata = {
            "timestamp": timestamp,
            "save_interval": self.save_interval,
            "path": str(self.path),
            "num_robots": data[next(iter(data))].shape[0],  # type: ignore
            "feature_keys": list(data.keys()),
            "elapsed_time": time() - self.start_time,
            "batches_logged": self.batches_logged,
            "sim_dt": self._sim_dt,  # to be filled in later if known
            "control_dt": self._control_dt,  # to be filled in later if known
            "mpc_dt": self._mpc_dt,  # to be filled in later if known
        }
    
    def set_dt(self, sim_dt: float, control_dt: float, mpc_dt: float):
        """Set the time step metadata.

        Args:
            sim_dt (float): Simulation time step.
            control_dt (float): Control time step.
            mpc_dt (float): MPC time step.
        """
        self._sim_dt = sim_dt
        self._control_dt = control_dt
        self._mpc_dt = mpc_dt
        if self.metadata:
            self.metadata["sim_dt"] = sim_dt
            self.metadata["control_dt"] = control_dt
            self.metadata["mpc_dt"] = mpc_dt

    def flush(self):
        """Flush any remaining data to disk."""
        if self.step == 0:
            # avoid double saving
            return
        self._save()

    def _update_metadata(self):
        self.metadata["elapsed_time"] = time() - self.start_time
        self.metadata["batches_logged"] = self.batches_logged

    def _save(self):
        self.step = 0
        if not self.data:
            logger.warning("No data to save.")
            return

        self._update_metadata()

        # Stack lists into tensors
        compressed_data: dict[str, Any] = {}
        for key in self.data:
            value_list = self.data[key]
            if value_list:
                compressed_data[key] = torch.stack(value_list)
            else:
                logger.warning(f"No data recorded for key: {key}")
        
        compressed_data["metadata"] = self.metadata

        torch.save(compressed_data, self.path)
        logger.debug(f"Data saved to {self.path}")

    def __del__(self):
        self.flush()


data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(exist_ok=True)

data_logger = _DataLogger(data_dir / f"{timestamp}.pt", save_interval=10)

atexit.register(data_logger.flush)
