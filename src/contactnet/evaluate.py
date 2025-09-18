import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from src.contactnet.train import QuadrupedDataset, QuadrupedModel
from src.contactnet.tree import IsaacStateCPU, StepNode
from src import PROJECT_ROOT
from src.contactnet.debug import view_footstep_cost_map
import logging
logger = logging.getLogger(__name__)

def compare_model_output(model: nn.Module, dataset: QuadrupedDataset, device: torch.device, num_samples: int = 5):
    """
    Visualize model outputs for a few samples from the dataset.

    Args:
        model (nn.Module): The trained ContactNet model.
        dataset (Dataset): The dataset to sample from.
        device (torch.device): The device to run the model on.
        num_samples (int): Number of samples to visualize.
    """
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        state, expected_costmap = dataset[idx]
        # move state to device
        state = state.to(device)

        with torch.no_grad():
            calculated_costmap = model(state)

        calculated_costmap = calculated_costmap.cpu().numpy().reshape(4, 5, 5)  # Reshape back to (4, 5, 5)
        expected_costmap = expected_costmap.cpu().numpy().reshape(4, 5, 5)  # Reshape back to (4, 5, 5)

        view_footstep_cost_map(calculated_costmap, title=f"Calculated Cost Map Sample {idx}", save_figure=True)
        view_footstep_cost_map(expected_costmap, title=f"Expected Cost Map Sample {idx}", save_figure=True)

def main():
    # load the most recent model
    checkpoint_dir = PROJECT_ROOT / "training" / "checkpoints"
    model_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.is_dir()], key=lambda d: d.stat().st_mtime, reverse=True)
    if not model_dirs:
        logger.warning("No model directories found in checkpoints.")
        return
    checkpoint_path = model_dirs[0] / "best_model.pt"
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load dataset
    dataset = QuadrupedDataset([
        PROJECT_ROOT / "data" / "2025-09-18T11-38-21.pkl",
        PROJECT_ROOT / "data" / "2025-09-18T11-45-07.pkl",
    ])

    # Load model
    model = QuadrupedModel(input_dim=dataset.input_dim, output_dim_per_foot=dataset.output_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    compare_model_output(model, dataset, device=device, num_samples=5)

if __name__ == "__main__":
    main()