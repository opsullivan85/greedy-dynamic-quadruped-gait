import argparse
import pickle

import numpy as np
from src import PROJECT_ROOT, logger
from src.contactnet.debug import view_footstep_cost_map
from src.contactnet.tree import StepNode
import src.simulation.cfg.footstep_scanner_constants as fs


parser = argparse.ArgumentParser(description="Data information")
parser.add_argument(
    "--costmaps",
    action="store_true",
    help="Generate costmaps for the dataset"
)
args, unused_args = parser.parse_known_args()

def log_data_info():
    """Logs information about the data files in the data directory."""
    # load most recent data from
    data_dir = PROJECT_ROOT / "data" / "datasets"
    data_files = sorted(data_dir.glob("*.pkl"))
    total_points = 0
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    for data_file in data_files:
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        msg = f"Found data file: {data_file}\n"
        msg += f"\t- training_data: {len(data['training_data'])} points\n"
        total_points += len(data['training_data'])
        metadata_msg = "\n\t\t- " + "\n\t\t- ".join(f"{k}: {v}" for k, v in data['metadata'].items())
        msg += f"\t- metadata: {metadata_msg}\n"
        logger.info(msg)
        
    logger.info(f"Total training data points across {len(data_files)} files: {total_points}")

def generate_all_costmaps():
    """Generates costmaps for all data files in the data directory."""
    data_dir = PROJECT_ROOT / "data" / "datasets"
    data_files = sorted(data_dir.glob("*.pkl"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    for data_file in data_files:
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Generating costmaps for data file: {data_file}")

        for idx, instance in enumerate(data['training_data']):
            instance: StepNode
            if instance.cost_map is None:
                logger.warning(f"No cost map found for instance {idx} in {data_file}, skipping.")
                continue
            cost_map: np.ndarray = instance.cost_map
            # reshape cost map to 4,n,m
            cost_map = cost_map.reshape((4, *fs.grid_size))

            title = f"{data_file.name} - {idx}"
            view_footstep_cost_map(
                cost_map=cost_map,
                title=title,
                save_figure=True,
            )

def main():
    log_data_info()
    if args.costmaps:
        generate_all_costmaps()