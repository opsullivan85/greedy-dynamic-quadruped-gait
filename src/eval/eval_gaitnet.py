from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Evaluate Gaitnet"
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from src.eval.evaluator import Evaluator
from src.gaitnet.util import get_checkpoint_path
from src.gaitnet.components.gaitnet_env import GaitNetEnv
from src.gaitnet.env_cfg.gaitnet_env_cfg import get_env
from src.util import log_exceptions
from src.gaitnet import gaitnet
import re
from pathlib import Path
import src.constants as const
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
    agent = model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    state_dict = {re.sub(r"^actor\.", "", k): v for k, v in state_dict.items() if k.startswith("actor.")}
    agent.load_state_dict(state_dict)
    agent.to(device)
    return agent


def main():
    args_cli.device = "cpu"
    args_cli.num_envs = 1
    device = torch.device(args_cli.device)
    model = load_model(get_checkpoint_path(), device)
    model.eval()

    env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        manager_class=GaitNetEnv,
    )
    observations, info = env.reset()
    obs: torch.Tensor = observations["policy"]  # type: ignore

    evaluator = Evaluator(env, observations, trials=1)

    with torch.inference_mode():
        while not evaluator.done:
            actions = gaitnet.GaitnetActor.act_inference(model, obs)
            env_step_info = env.step(actions)
            observations, rew, terminated, truncated, info = env_step_info
            obs = observations["policy"]  # type: ignore
            evaluator.process(env_step_info)


if __name__ == "__main__":
    # with log_exceptions(logger):
    main()