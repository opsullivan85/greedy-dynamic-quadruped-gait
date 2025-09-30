from skrl.memories.torch import RandomMemory
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
from src.gaitnet.gaitnet import GaitNetActorCritic, FootstepOptionGenerator
import src.simulation.cfg.footstep_scanner_constants as fs
from src.gaitnet.env_cfg.gaitnet_env import make_env, make_env_cfg
from src.util import log_exceptions
from src import get_logger, timestamp, PROJECT_ROOT
from isaaclab.utils.io import dump_pickle, dump_yaml

from skrl.trainers.torch import SequentialTrainer
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch.base import Wrapper
from skrl.envs.wrappers.torch.isaaclab_envs import IsaacLabWrapper

logger = get_logger()

# # Global flag for graceful shutdown
# shutdown_requested = False


# def signal_handler(sig, frame):
#     global shutdown_requested
#     if multiprocessing.current_process().name == "MainProcess":
#         signal_name = signal.Signals(sig).name
#         logger.info(f"signal {signal_name} received in main process, shutting down...")
#     shutdown_requested = True


# # Set up the signal handler for SIGINT (Ctrl+C)
# signal.signal(signal.SIGINT, signal_handler)

###############################################################################
###############################################################################


def main():
    env_cfg = make_env_cfg(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    # instantiate the agent's models
    models = {}
    models["q_network"] = ...
    models["target_q_network"] = ...  # only required during training

    # adjust some configuration if necessary
    cfg_agent = DDQN_DEFAULT_CONFIG.copy()
    cfg_agent["<KEY>"] = ...

    # setup logging directories
    log_dir = PROJECT_ROOT / "training" / "gaitnet" / "runs" / timestamp
    save_dir = PROJECT_ROOT / "training" / "gaitnet" / "checkpoints" / timestamp
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # set the IO descriptors output directory
    env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.io_descriptors_output_dir = str(log_dir)

    # setup env
    env = make_env(env_cfg)  # type: ignore
    env: IsaacLabWrapper = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # type: ignore

    memory_size = 24  # TODO: this should match the agent rollout length
    memory = RandomMemory(memory_size, device=env.device)  # only required during training

    # instantiate the agent
    agent = DDQN(models=models,
                memory=memory,  # only required during training
                cfg=cfg_agent,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)

    # load checkpoint (if specified)
    resume_path = args_cli.checkpoint if args_cli.checkpoint else None
    if resume_path:
        logger.info(f"loading model checkpoint from: {resume_path}")
        agent.load(resume_path)


    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    # runner = Runner(env, agent_cfg)

    # Sequential trainer
    # https://skrl.readthedocs.io/en/latest/api/trainers/sequential.html
    trainer = SequentialTrainer(
        env=env,
        agents=[agent],
    )

    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
