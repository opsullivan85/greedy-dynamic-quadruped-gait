import torch
from isaaclab.envs import DirectRLEnv # pyright: ignore[reportMissingImports]

from src.simulation.quad_env_cfg import QuadrupedEnvCfg # pyright: ignore[reportAttributeAccessIssue]
from src.simulation.controller import Controller


class QuadrupedEnv(DirectRLEnv):
    """Custom quadruped environment with RL training support"""
    
    cfg: QuadrupedEnvCfg
    
    def __init__(self, cfg: QuadrupedEnvCfg, render_mode: str|None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Initialize hybrid controller
        self.controller = Controller(num_envs=self.num_envs, training_mode=True)
        
        # RL training state
        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        
    def _setup_scene(self):
        """Setup the scene with robot, terrain, and sensors"""
        # Add robot
        self._robot = self.scene.articulations["robot"]
        
        # Add sensors
        self._height_scanner = self.scene.sensors["height_scanner"]
        self._contact_sensor = self.scene.sensors["contact_sensor"]
        
        # Clone environments
        self.scene.clone_environments(copy_from_source=False)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called before each physics step"""
        # Get current robot state
        robot_state = self._get_robot_state()
        
        # Get terrain heightmap
        heightmap = self._get_terrain_heightmap()
        
        # Compute torques using hybrid controller
        torques = self.controller.compute_torques(heightmap, robot_state)
        
        # Apply torques to robot
        self._robot.set_joint_effort_target(torques)
        
    def _get_robot_state(self) -> dict:
        """Extract robot state for the controller"""
        return {
            'joint_pos': self._robot.data.joint_pos[:, :12],
            'joint_vel': self._robot.data.joint_vel[:, :12],
            'base_pos': self._robot.data.root_pos_w,
            'base_quat': self._robot.data.root_quat_w,
            'base_lin_vel': self._robot.data.root_lin_vel_b,
            'base_ang_vel': self._robot.data.root_ang_vel_b,
            'contact_forces': self._contact_sensor.data.net_forces_w.view(self.num_envs, 4, 3),
        }
        
    def _get_terrain_heightmap(self) -> torch.Tensor:
        """Get local terrain heightmap around robot"""
        # Get ray-casting data
        ray_hits = self._height_scanner.data.ray_hits_w[..., 2]  # Z coordinates
        base_pos = self._robot.data.root_pos_w[..., 2:3]  # Robot base height
        
        # Convert to relative heights
        heightmap = ray_hits - base_pos
        
        # Reshape to grid (20x20 for 2m x 2m at 0.1m resolution)
        heightmap = heightmap.view(self.num_envs, 20, 20)
        
        return heightmap
        
    def _get_observations(self) -> dict:
        """Get observations for RL training"""
        robot_state = self._get_robot_state()
        heightmap = self._get_terrain_heightmap()
        
        # Use controller's observation preparation
        rl_obs = self.controller._prepare_rl_observations(heightmap, robot_state)
        
        return {"policy": rl_obs}
        
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for RL training"""
        # Forward velocity reward
        base_lin_vel = self._robot.data.root_lin_vel_b
        forward_vel = base_lin_vel[:, 0]  # X-axis velocity
        target_vel = 1.0  # m/s target forward velocity
        vel_reward = torch.exp(-torch.abs(forward_vel - target_vel))
        
        # Stability reward (penalize excessive roll/pitch)
        base_quat = self._robot.data.root_quat_w
        roll_pitch = self.controller._quat_to_euler(base_quat)[:, :2]
        stability_reward = torch.exp(-torch.norm(roll_pitch, dim=1))
        
        # Energy efficiency (penalize high torques)
        joint_torques = self._robot.data.applied_torque[:, :12]
        energy_penalty = -0.01 * torch.sum(torch.abs(joint_torques), dim=1)
        
        # Contact stability reward
        contact_forces = self._contact_sensor.data.net_forces_w
        foot_contacts = torch.norm(contact_forces, dim=-1) > 1.0
        num_contacts = torch.sum(foot_contacts, dim=1).float()
        contact_reward = torch.exp(-torch.abs(num_contacts - 2.0))  # Prefer 2 feet in contact
        
        # Joint limit penalty
        joint_pos = self._robot.data.joint_pos[:, :12]
        joint_limits_lower = torch.tensor([-0.8, -0.8, -2.7] * 4).to(self.device)
        joint_limits_upper = torch.tensor([0.8, 4.5, -0.9] * 4).to(self.device)
        
        joint_violations = torch.sum(
            torch.clamp(joint_pos - joint_limits_upper, min=0) + 
            torch.clamp(joint_limits_lower - joint_pos, min=0), 
            dim=1
        )
        joint_penalty = -10.0 * joint_violations
        
        # Combine rewards
        total_reward = (
            2.0 * vel_reward + 
            1.0 * stability_reward + 
            energy_penalty + 
            0.5 * contact_reward + 
            joint_penalty
        )
        
        self.episode_rewards += total_reward
        return total_reward
        
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes are done"""
        # Termination conditions
        base_pos = self._robot.data.root_pos_w
        base_height = base_pos[:, 2]
        
        # Terminate if robot falls
        fallen = base_height < 0.2
        
        # Terminate if robot flips over
        base_quat = self._robot.data.root_quat_w
        euler = self.controller._quat_to_euler(base_quat)
        flipped = (torch.abs(euler[:, 0]) > 1.57) | (torch.abs(euler[:, 1]) > 1.57)
        
        terminated = fallen | flipped
        
        # Truncation (max episode length)
        self.episode_lengths += 1
        max_episode_length = int(self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
        truncated = self.episode_lengths >= max_episode_length
        
        return terminated, truncated
        
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # Reset episode tracking
        self.episode_rewards[env_ids] = 0.0
        self.episode_lengths[env_ids] = 0
        
        # Reset robot to default state with small randomization
        default_joint_pos = torch.tensor([
            0.0, 0.9, -1.8,  # FL
            0.0, 0.9, -1.8,  # FR
            0.0, 0.9, -1.8,  # RL
            0.0, 0.9, -1.8   # RR
        ]).repeat(len(env_ids), 1).to(self.device)
        
        # Add noise to initial joint positions
        joint_pos = default_joint_pos + torch.randn_like(default_joint_pos) * 0.1
        joint_vel = torch.zeros_like(joint_pos)
        
        # Random base position
        base_pos = torch.zeros(len(env_ids), 3, device=self.device)
        base_pos[:, 2] = 0.4  # Start 40cm above ground
        base_pos[:, :2] += torch.randn(len(env_ids), 2, device=self.device) * 0.1
        
        base_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(len(env_ids), 1).to(self.device)
        base_vel = torch.zeros(len(env_ids), 6, device=self.device)
        
        # Apply reset
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.write_root_state_to_sim(
            torch.cat([base_pos, base_quat, base_vel], dim=-1), env_ids=env_ids
        )
