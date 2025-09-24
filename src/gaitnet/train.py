from isaaclab.app import AppLauncher
import argparse

from src.gaitnet.gaitnet import GaitNetActorCritic

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=10000, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import multiprocessing
import signal
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # type: ignore
from rsl_rl.algorithms import PPO
from rsl_rl.runners import OnPolicyRunner
from src.gaitnet.env_cfg import GaitNetEnvCfg
from src.gaitnet.env_cfg.gaitnet_env import get_gaitnet_env_cfg
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
    env_cfg: GaitNetEnvCfg = get_gaitnet_env_cfg(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        iterations_between_mpc=5,  # 50 Hz MPC
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # wrap for RL training
    env = RslRlVecEnvWrapper(env)

    # Create actor-critic network
    actor_critic = GaitNetActorCritic(
        num_obs=1,  # TODO: how do I get this from the observer?
        num_actions=6,  # 5 footstep options + 1 no-action
        hidden_dims=[256, 256],
    )
    
    # Configure PPO
    ppo_cfg = {
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "clip_param": 0.2,
        "max_grad_norm": 1.0,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 3e-4,
        "schedule": "adaptive",
        "desired_kl": 0.01,
    }
    
    # Create PPO algorithm
    ppo = PPO(actor_critic, **ppo_cfg)
    
    # Create runner
    runner = OnPolicyRunner(
        env=env,
        train_cfg={
            "algorithm": ppo,
            "policy": actor_critic,
            "num_steps_per_env": 25,  # Roughly 10 second
            "max_iterations": 10000,
            "save_interval": 100,
            "experiment_name": "footstep_selection",
            "run_name": "value_based",
        },
        log_dir="./logs",
        device="cuda",
    )
    runner.add_git_repo_to_log(__file__)

    if args_cli.resume is not None:
        runner.load(args_cli.resume)
    
    runner.learn(
        num_learning_iterations=args_cli.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()

if __name__ == "__main__":
    with log_exceptions(logger):
        main()
