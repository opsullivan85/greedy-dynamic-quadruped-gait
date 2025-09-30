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
import torch
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # type: ignore
from skrl.utils.runner.torch import Runner  # type: ignore
from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG  # type: ignore
from skrl.memories.torch import RandomMemory  # type: ignore
from src.gaitnet.gaitnet import GaitNetDDQN, FootstepOptionGenerator
from src.gaitnet.actions.footstep_action import NO_STEP
import src.simulation.cfg.footstep_scanner_constants as fs
from src.gaitnet.env_cfg.gaitnet_env import get_env
from src.util import log_exceptions
from src import get_logger


logger = get_logger()


def main():
    env_cfg, env = get_env(
        num_envs=args_cli.num_envs,
        device=args_cli.device,
    )

    # wrap for RL training with skrl
    env = SkrlVecEnvWrapper(env)

    obs_space = env.observation_space.shape[0]
    robot_state_dim = obs_space - (4*fs.grid_size[0]*fs.grid_size[1])  # subtract height scan
    # action_space = env.action_space.shape[1]
    # 2 per leg
    num_footstep_candidates = 12

    # Number of discrete actions (footstep options + no-action)
    num_footstep_candidates = 12
    num_actions = num_footstep_candidates + 1  # +1 for no-action option

    footstep_option_generator = FootstepOptionGenerator(
        env=env.unwrapped,
        num_options=num_footstep_candidates,
    )

    # Create unique experiment name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"gaitnet_ddqn_{timestamp}"

    # Create unique log directory
    log_dir = f"./training/gaitnet/runs/{experiment_name}"
    save_dir = f"./training/gaitnet/checkpoints/{experiment_name}"

    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Instantiate the Q-network (and target network)
    device = torch.device(args_cli.device)

    # Create environment wrapper that handles discrete-to-continuous action conversion
    class GaitNetEnvWrapper:
        """Environment wrapper that converts DDQN discrete actions to continuous footstep actions."""
        
        def __init__(self, env, footstep_generator, robot_state_dim):
            self._env = env
            self._footstep_generator = footstep_generator
            self._robot_state_dim = robot_state_dim
            # Override action space to be discrete
            import gymnasium as gym
            self._action_space = gym.spaces.Discrete(num_actions)
            
        @property
        def action_space(self):
            return self._action_space
            
        def step(self, actions):
            """Convert discrete actions to continuous and step environment."""
            continuous_actions = self._convert_discrete_to_continuous(actions)
            return self._env.step(continuous_actions)
            
        def _convert_discrete_to_continuous(self, discrete_actions):
            """Convert discrete action indices to continuous footstep actions."""
            batch_size = discrete_actions.shape[0]
            device = discrete_actions.device
            continuous_actions = torch.zeros((batch_size, 13), device=device)  # 4*3 footsteps + 1 duration
            
            for b in range(batch_size):
                action_idx = discrete_actions[b].long().item()
                
                if action_idx < num_footstep_candidates:
                    # Get robot state for this batch item
                    robot_state = self._env.observation_buffer[b, :self._robot_state_dim].cpu().numpy()
                    
                    # Generate footstep options for current state
                    options = self._footstep_generator.get_footstep_options(robot_state)
                    footstep_options = options['footstep_options']  # [num_actions, 4, 3]
                    swing_durations = options['swing_durations']    # [num_actions]
                    
                    # Select the chosen footstep option
                    selected_footstep = footstep_options[action_idx]  # [4, 3]  
                    selected_duration = swing_durations[action_idx]
                    
                    # Flatten and combine: [4*3] + [1] = [13]
                    continuous_actions[b, :12] = torch.tensor(selected_footstep.flatten(), device=device)
                    continuous_actions[b, 12] = torch.tensor(selected_duration, device=device)
                # else: action_idx == num_footstep_candidates (no-action), keep zeros
                    
            return continuous_actions
            
        def __getattr__(self, name):
            # Delegate all other attributes to the wrapped environment
            return getattr(self._env, name)

    # Wrap the environment
    wrapped_env = GaitNetEnvWrapper(env, footstep_option_generator, robot_state_dim)

    # Configure DDQN
    cfg_ddqn = DDQN_DEFAULT_CONFIG.copy()
    cfg_ddqn.update(
        {
            "gradient_steps": 1,
            "batch_size": 64,
            "discount_factor": 0.99,
            "polyak": 0.005,
            "learning_rate": 1e-3,
            "learning_starts": 1000,  # Start learning after collecting some experience
            "update_interval": 4,  # Update every 4 steps
            "target_update_interval": 1000,  # Update target network every 1000 steps
            "exploration": {
                "initial_epsilon": 1.0,
                "final_epsilon": 0.01,
                "timesteps": 100000,  # Decay epsilon over more timesteps
            },
            "experiment": {
                "directory": log_dir,
                "experiment_name": experiment_name,
                "write_interval": 1000,
                "checkpoint_interval": 5000,
                "store_separately": False,
                "wandb": False,
            },
        }
    )

    # Configure skrl Runner for DDQN training
    runner_cfg = {
        "seed": args_cli.seed,
        "models": {
            "separate": True,
            "q_network": {
                "class": "DeterministicMixin",
                "network": [
                    {
                        "name": "net",
                        "input": "STATES",
                        "layers": [256, 128, 64],
                        "activations": "relu"
                    }
                ],
                "output": "ACTIONS"
            },
            "target_q_network": {
                "class": "DeterministicMixin",
                "network": [
                    {
                        "name": "net", 
                        "input": "STATES",
                        "layers": [256, 128, 64],
                        "activations": "relu"
                    }
                ],
                "output": "ACTIONS"
            }
        },
        "memory": {
            "class": "RandomMemory",
            "memory_size": 100000,
        },  
        "agent": {
            "class": "DDQN",
            **cfg_ddqn
        },
        "trainer": {
            "class": "SequentialTrainer",
            "timesteps": args_cli.max_iterations * env.num_envs,
            "headless": True,
        }
    }

    # Create runner with wrapped environment
    runner = Runner(env=wrapped_env, cfg=runner_cfg)  # type: ignore

    # Resume from checkpoint if specified
    if args_cli.resume is not None:
        logger.info(f"Loading checkpoint from {args_cli.resume}")
        runner.agent.load(args_cli.resume)

    logger.info("Starting DDQN training...")

    try:
        runner.run()
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
