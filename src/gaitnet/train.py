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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # type: ignore
from src.gaitnet.gaitnet import GaitNetActorCritic
from src.gaitnet.env_cfg.footstep_options_env import FootstepOptionEnv
import src.simulation.cfg.footstep_scanner_constants as fs
from rsl_rl.runners import on_policy_runner
import rsl_rl.modules
from src.gaitnet.env_cfg.gaitnet_env import get_env
from src.util import log_exceptions
from src import get_logger

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
    env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        manager_class=FootstepOptionEnv
    )

    # wrap for RL training
    env = RslRlVecEnvWrapper(env)

    # hack to get OnPolicyRunner to be able to initiate a GaitNetActorCritic
    actor_critic_class = GaitNetActorCritic
    on_policy_runner.__dict__[actor_critic_class.__name__] = actor_critic_class  # type: ignore

    # Create unique experiment name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gaitnet_{timestamp}"
    
    # Create unique log directory
    log_dir = f"./training/gaitnet/runs/{experiment_name}"
    save_dir = f"./training/gaitnet/checkpoints/{experiment_name}"
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Prepare config dict for OnPolicyRunner
    train_cfg = {
        "algorithm": {
            "class_name": "PPO",
            "schedule": "fixed",  # we don't support adaptive
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
            # "class_name": actor_critic_class.__name__,
            "class_name": "ActorCritic",
            "hidden_dims": [128, 128, 128],
            "footstep_observation_manager": env.unwrapped.observation_manager,  # type: ignore
        },
        "log_dir": log_dir,
        "num_steps_per_env": 1000,  # ~2 episodes per batch (episode = 10s = 500 iterations)
        "save_interval": 5,
        "empirical_normalization": False,
        "logger": "tensorboard",  # Explicitly set TensorBoard as logger
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
    
    # Let the runner handle the entire training loop
    # This ensures proper TensorBoard logging continuity
    try:
        runner.learn(num_learning_iterations=args_cli.max_iterations)
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Closing environments")
    env.close()


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
