"""
Complete Isaac Lab integration with RL-enhanced quadruped controller

Note: Isaac sim needs to be launched before even importing this file, e.g.,
```
from omni.isaac.lab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app
```
"""
import argparse


from isaaclab.app import AppLauncher # pyright: ignore[reportMissingImports]


# add argparse arguments
parser = argparse.ArgumentParser(description="Simulation for Greedy Dynamic Quadruped Gait.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch
import numpy as np

import isaaclab.envs.mdp as mdp # pyright: ignore[reportMissingImports]
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg # pyright: ignore[reportMissingImports]
from isaaclab.managers import EventTermCfg as EventTerm # pyright: ignore[reportMissingImports]
from isaaclab.managers import ObservationGroupCfg as ObsGroup # pyright: ignore[reportMissingImports]
from isaaclab.managers import ObservationTermCfg as ObsTerm # pyright: ignore[reportMissingImports]
from isaaclab.managers import SceneEntityCfg # pyright: ignore[reportMissingImports]
from isaaclab.utils import configclass # pyright: ignore[reportMissingImports]

# import torch
# import numpy as np
# from isaaclab.app import AppLauncher

# # Launch Isaac Sim
# app_launcher = AppLauncher(headless=False)
# simulation_app = app_launcher.app

# import isaaclab.sim as sim_utils
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg
# from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.utils import configclass
# import isaaclab.assets.unitree as unitree_assets

from src.simulation.quad_env_cfg import QuadrupedEnvCfg # pyright: ignore[reportAttributeAccessIssue]
from src.simulation.quad_env import QuadrupedEnv
from src.simulation.controller import Controller





class RLTrainer:
    """PPO trainer for the RL policy"""
    
    def __init__(self, env: QuadrupedEnv, learning_rate: float = 3e-4):
        self.env = env
        self.device = env.device
        
        # PPO hyperparameters
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        
        # Training parameters
        self.buffer_size = 2048
        self.batch_size = 64
        self.n_epochs = 10
        
        # Initialize optimizers
        self.optimizer = torch.optim.Adam(
            self.env.controller.rl_policy.parameters(), 
            lr=learning_rate
        )
        
        # Value network (critic)
        obs_dim = self.env.controller.obs_dim
        self.value_network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256), 
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to(self.device)
        
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Training buffers
        self.reset_buffers()
        
    def reset_buffers(self):
        """Reset training buffers"""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []
        
    def collect_experience(self, obs: torch.Tensor, actions: torch.Tensor, 
                          rewards: torch.Tensor, dones: torch.Tensor):
        """Collect experience for training"""
        # Get action probabilities and values
        action_mean = self.env.controller.rl_policy(obs)
        
        # Gaussian policy for continuous actions
        action_std = torch.ones_like(action_mean) * 0.3
        dist = torch.distributions.Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        values = self.value_network(obs).squeeze()
        
        # Store in buffers
        self.observations.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)
        self.log_probs.append(log_probs)
        
    def compute_advantages(self):
        """Compute GAE advantages"""
        rewards = torch.stack(self.rewards)
        values = torch.stack(self.values)
        dones = torch.stack(self.dones)
        
        advantages = []
        advantage = torch.zeros_like(rewards[0])
        
        for t in reversed(range(len(rewards))):
            next_value = torch.zeros_like(values[t]) if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t].float()) * advantage
            advantages.insert(0, advantage.clone())
            
        advantages = torch.stack(advantages)
        returns = advantages + values
        
        return advantages.flatten(), returns.flatten()
        
    def update_policy(self):
        """Update policy using PPO"""
        if len(self.observations) < self.buffer_size:
            return {}
            
        # Compute advantages
        advantages, returns = self.compute_advantages()
        
        # Flatten tensors
        obs = torch.cat(self.observations, dim=0)
        actions = torch.cat(self.actions, dim=0)
        old_log_probs = torch.cat(self.log_probs, dim=0)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        
        # PPO training loop
        for epoch in range(self.n_epochs):
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Forward pass
                action_mean = self.env.controller.rl_policy(batch_obs)
                action_std = torch.ones_like(action_mean) * 0.3
                dist = torch.distributions.Normal(action_mean, action_std)
                
                log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                values = self.value_network(batch_obs).squeeze()
                
                # PPO losses
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = torch.nn.functional.mse_loss(values, batch_returns)
                
                total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                
                # Update networks
                self.optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.env.controller.rl_policy.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.value_optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())
                
        # Reset buffers
        self.reset_buffers()
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses), 
            'entropy': np.mean(entropies)
        }


def train_rl_controller():
    """Training script for RL-enhanced controller"""
    # Create training environment
    cfg = QuadrupedEnvCfg()
    cfg.num_envs = 4096  # Many parallel environments
    cfg.episode_length_s = 10.0  # Shorter episodes
    
    env = QuadrupedEnv(cfg)
    trainer = RLTrainer(env)
    
    # Training parameters
    max_iterations = 1000
    update_frequency = 2048
    save_frequency = 50
    
    print("Starting RL training...")
    print(f"Training with {env.num_envs} parallel environments")
    print(f"Observation dim: {env.controller.obs_dim}")
    print(f"Action dim: {env.controller.action_dim}")
    
    # Reset environment
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    
    step_count = 0
    
    for iteration in range(max_iterations):
        epoch_rewards = []
        
        # Collect experience
        for step in range(update_frequency // env.num_envs):
            # Sample actions from current policy
            with torch.no_grad():
                actions = env.controller.rl_policy(obs)
                # Add exploration noise
                actions += torch.randn_like(actions) * 0.1
                actions = torch.clamp(actions, -1, 1)
            
            # Step environment
            next_obs_dict, rewards, terminated, truncated, info = env.step(actions)
            next_obs = next_obs_dict["policy"]
            
            dones = terminated | truncated
            
            # Collect experience
            trainer.collect_experience(obs, actions, rewards, dones)
            
            obs = next_obs
            step_count += env.num_envs
            epoch_rewards.append(rewards.mean().item())
            
            # Reset if needed
            if dones.any():
                obs_dict, _ = env.reset()
                obs = obs_dict["policy"]
                
        # Update policy
        training_stats = trainer.update_policy()
        
        # Logging
        avg_reward = np.mean(epoch_rewards)
        print(f"Iteration {iteration:4d} | Steps: {step_count:8d} | "
              f"Reward: {avg_reward:7.3f} | Policy Loss: {training_stats.get('policy_loss', 0):.4f}")
              
        # Save checkpoint
        if iteration % save_frequency == 0:
            checkpoint_path = f"quadruped_rl_policy_iter_{iteration}.pt"
            env.controller.save_policy(checkpoint_path, 
                                     trainer.optimizer.state_dict(), 
                                     iteration)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    env.close()
    print("Training completed!")


def main():
    """Choose between training and inference"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="test",
                       help="Training or testing mode")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to trained policy checkpoint")
    args = parser.parse_args()
    
    if args.mode == "train":
        train_rl_controller()
    else:
        # Test mode - run simulation
        cfg = QuadrupedEnvCfg()
        env = QuadrupedEnv(cfg)
        
        # Load trained policy if provided
        if args.checkpoint:
            env.controller.load_policy(args.checkpoint)
            env.controller.training_mode = False
            print(f"Loaded trained policy from {args.checkpoint}")
        else:
            print("Running with untrained policy (random actions)")
        
        # Reset environment
        env.reset()
        
        print("Starting simulation...")
        print("Your hybrid MPC+RL controller is running!")
        
        # Run simulation
        while simulation_app.is_running():
            # Step environment
            obs, rewards, terminated, truncated, info = env.step(
                torch.zeros(env.num_envs, env.cfg.num_actions, device=env.device)
            )
            
            # Reset if needed
            if terminated.any() or truncated.any():
                env.reset()
        
        env.close()
