import logging
from pathlib import Path
from typing import Any, TypeAlias
import numpy as np
import torch
from src import timestamp, PROJECT_ROOT

logger = logging.getLogger(__name__)

DataType: TypeAlias = dict[str, torch.Tensor]

class _DataLogger:
    def __init__(self, path: Path, save_interval: int = 1):
        self.path = path
        self.save_interval = save_interval
        self.step = 0
        self.data: dict[str, list[torch.Tensor]] = {}

    def log(self, data: DataType):
        if not self.data:
            self._initial_setup(data)

        self._try_save()

    def _try_save(self):
        self.step += 1
        if self.step % self.save_interval == 0:
            self.save()

    def _initial_setup(self, data: DataType):
        self.data = {key: [] for key in data.keys()}

    def flush(self):
        self.save()

    def save(self):
        self.step = 0
        if not self.data:
            logger.warning("No data to save.")
            return
        
        # Stack lists into tensors
        compressed_data: dict[str, torch.Tensor] = {}
        for key, value_list in self.data.items():
            if value_list:
                compressed_data[key] = torch.stack(value_list)
            else:
                logger.warning(f"No data recorded for key: {key}")

        torch.save(compressed_data, self.path)
        logger.info(f"Data saved to {self.path}")


data_dir = PROJECT_ROOT / "data"
data_dir.mkdir(exist_ok=True)

data_logger = _DataLogger(data_dir / f"{timestamp}.pt", save_interval=10)