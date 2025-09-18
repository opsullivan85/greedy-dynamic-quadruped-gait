import pickle
from src import PROJECT_ROOT, logger

def log_data_info():
    """Logs information about the data files in the data directory."""
    # load most recent data from
    data_dir = PROJECT_ROOT / "data"
    data_files = sorted(data_dir.glob("*.pkl"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")
    for data_file in data_files:
        with open(data_file, "rb") as f:
            data = pickle.load(f)

        msg = f"Found data file: {data_file}\n"
        msg += f"\t- training_data: {len(data['training_data'])} points\n"
        metadata_msg = "\n\t\t- " + "\n\t\t- ".join(f"{k}: {v}" for k, v in data['metadata'].items())
        msg += f"\t- metadata: {metadata_msg}\n"
        logger.info(msg)