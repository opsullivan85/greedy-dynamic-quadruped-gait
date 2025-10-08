
from typing import Any
from tensorflow.python import checkpoint
import torch
from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Train an RL agent with Stable-Baselines3."
)
parser.add_argument(
    "--num_envs", type=int, default=100, help="Number of environments to simulate."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations", type=int, default=10000, help="RL Policy training iterations."
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume training from.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import multiprocessing
import signal
import datetime
import os
from src.gaitnet.util import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
from src.gaitnet.env_cfg.footstep_options_env import FootstepOptionEnv
import src.simulation.cfg.footstep_scanner_constants as fs
from rsl_rl.runners import on_policy_runner
import rsl_rl.modules
from src.gaitnet.env_cfg.gaitnet_env import get_env
from src.util import log_exceptions
from src.gaitnet import gaitnet
import re
from pathlib import Path
import src.constants as const
from src.util.timer import Timer
from src import get_logger

logger = get_logger()

def load_model(checkpoint_path: Path, device: torch.device) -> gaitnet.GaitnetActor:
    model = gaitnet.GaitnetActor(
        shared_state_dim=const.gait_net.robot_state_dim,
        shared_layer_sizes=[128, 128, 128],
        unique_state_dim=const.gait_net.footstep_option_dim,
        unique_layer_sizes=[64, 64],
        trunk_layer_sizes=[128, 128, 128],
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {re.sub(r"^actor\.", "", k): v for k, v in state_dict.items() if k.startswith("actor.")}
    model.load_state_dict(new_state_dict)
    model.to(device)
    return model

def main():
    args_cli.device = "cpu"
    args_cli.num_envs = 1
    device = torch.device(args_cli.device)
    model = load_model(get_checkpoint_path(), device)
    model.eval()

    env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        manager_class=FootstepOptionEnv,
    )
    obs_, info = env.reset()
    obs: torch.Tensor = obs_["policy"]  # type: ignore

    with torch.inference_mode():
        while True:
            with Timer(logger, "model inference"):
                actions = gaitnet.GaitnetActor.act_inference(model, obs)
            obs_, rew, terminated, truncated, info = env.step(actions)
            # coerce type system
            obs = obs_["policy"]  # type: ignore


if __name__ == "__main__":
    with log_exceptions(logger):
        main()