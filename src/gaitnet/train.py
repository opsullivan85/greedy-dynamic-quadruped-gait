import torch
from isaaclab.app import AppLauncher
import argparse

from src.simulation.cfg import terrain

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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
from src.gaitnet.gaitnet import GaitNetActorCritic
from rsl_rl.runners import on_policy_runner
import rsl_rl.modules
from src.gaitnet.env_cfg.gaitnet_env import get_env
from src.util import log_exceptions
from src import get_logger

logger = get_logger()

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    if multiprocessing.current_process().name == "MainProcess":
        signal_name = signal.Signals(sig).name
        logger.info(f"signal {signal_name} received in main process, shutting down...")
    shutdown_requested = True


# Set up the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

###############################################################################
###############################################################################


def main():
    env_cfg, env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    # wrap for RL training
    env = RslRlVecEnvWrapper(env)

    obs_space = env.observation_space["policy"].shape[1]
    # action_space = env.action_space.shape[1]
    num_footstep_candidates = 5

    def get_footstep_options_from_env(env):
        return torch.tensor(
            (
                (0, 0, 0),
                (1, 0, 0),
                (2, 0, 0),
                (3, 0, 0),
                (0, 2, 2),
            ),
            device=args_cli.device, dtype=torch.int
        )
    get_footstep_options = lambda: get_footstep_options_from_env(env)


    # hack to get OnPolicyRunner to be able to initiate a GaitNetActorCritic
    on_policy_runner.__dict__["GaitNetActorCritic"] = GaitNetActorCritic  # type: ignore

    # Prepare config dict for OnPolicyRunner
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "gamma": 0.99,
            "lam": 0.95,
        },
        "policy": {
            "class_name": "GaitNetActorCritic",
            "robot_state_dim": obs_space,
            "num_footstep_options": num_footstep_candidates,
            "hidden_dims": [256, 256],
            "get_footstep_options": get_footstep_options,
        },
        "save_dir": "./logs",
        "experiment_name": "gaitnet",
        "log_dir": "./logs",
        "num_steps_per_env": 24,
        "max_iterations": args_cli.max_iterations,
        "save_interval": 100,
        "empirical_normalization": False,
    }

    runner = on_policy_runner.OnPolicyRunner(
        env=env,
        train_cfg=train_cfg,
        log_dir=train_cfg["log_dir"],
        device=args_cli.device,
    )

    # Resume from checkpoint if specified
    if args_cli.resume is not None:
        logger.info(f"Loading checkpoint from {args_cli.resume}")
        runner.load(args_cli.resume)

    logger.info("Starting training...")
    iteration = 0
    while iteration < args_cli.max_iterations and not shutdown_requested:
        runner.learn(num_learning_iterations=1)
        iteration += 1
        if iteration % train_cfg["save_interval"] == 0:
            logger.info(f"Saving checkpoint at iteration {iteration}")
            runner.save(f"./logs/checkpoint_{iteration}.pt")

    # Final save at the end of training
    if not shutdown_requested:
        logger.info("Training completed successfully")
        runner.save(f"./logs/checkpoint_{args_cli.max_iterations}.pt")
    else:
        logger.info("Training interrupted, saving checkpoint")
        runner.save(f"./logs/checkpoint_{iteration}.pt")

    logger.info("Closing environments")
    env.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
