import torch
import torch.nn as nn
import numpy as np
import logging
from src.contactnet.train import QuadrupedDataset, QuadrupedModel
from src.contactnet.debug import view_footstep_cost_map
import logging
import argparse

from src.contactnet.util import get_checkpoint_path, get_dataset_paths

logger = logging.getLogger(__name__)

def compare_model_output(
    model: nn.Module,
    dataset: QuadrupedDataset,
    device: torch.device,
    num_samples: int = 5,
):
    """
    Visualize model outputs for a few samples from the dataset.

    Args:
        model (nn.Module): The trained ContactNet model.
        dataset (Dataset): The dataset to sample from.
        device (torch.device): The device to run the model on.
        num_samples (int): Number of samples to visualize.
    """
    model.eval()
    # indices = np.random.choice(len(dataset), num_samples, replace=False)
    indices = np.linspace(0, len(dataset)-1, num_samples, dtype=int)[:5]

    for idx in indices:
        state, expected_costmap = dataset[idx]
        # move state to device
        state = state.to(device)

        with torch.no_grad():
            calculated_costmap = model(state)

        calculated_costmap = (
            calculated_costmap.cpu().numpy().reshape(4, 5, 5)
        )  # Reshape back to (4, 5, 5)
        expected_costmap = (
            expected_costmap.cpu().numpy().reshape(4, 5, 5)
        )  # Reshape back to (4, 5, 5)

        view_footstep_cost_map(
            calculated_costmap,
            title=f"Calculated Cost Map Sample {idx}",
            save_figure=True,
        )
        view_footstep_cost_map(
            expected_costmap, title=f"Expected Cost Map Sample {idx}", save_figure=True
        )


def main():

    # Load dataset
    dataset = QuadrupedDataset(get_dataset_paths())

    # Load model
    model = QuadrupedModel(
        input_dim=dataset.input_dim, output_dim_per_foot=dataset.output_dim
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(get_checkpoint_path(), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    compare_model_output(model, dataset, device=device, num_samples=5)


if __name__ == "__main__":
    from src.util import log_exceptions
    with log_exceptions(logger):
        main()
