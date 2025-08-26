import torch


class RLPolicy(torch.nn.Module):
    """Neural network policy for RL component"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
            torch.nn.Tanh(),  # Actions typically normalized to [-1, 1]
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class Controller:
    """Hybrid controller with MPC + RL components"""

    def __init__(self, num_envs: int = 1, training_mode: bool = True):
        self.num_envs = num_envs
        self.num_joints = 12  # A1 has 12 actuated joints
        self.training_mode = training_mode

        # RL Policy dimensions
        self.heightmap_size = 20 * 20  # 400 height values
        self.robot_state_dim = (
            12 + 12 + 3 + 3 + 3 + 3
        )  # joint_pos + joint_vel + base_pos + base_vel + base_ang_vel + base_orient
        self.obs_dim = self.heightmap_size + self.robot_state_dim
        self.action_dim = 12  # Could be joint torques, or higher-level commands

        # Initialize RL policy
        self.rl_policy = RLPolicy(self.obs_dim, self.action_dim)

        # MPC/Classical controller parameters
        self.kp = 20.0
        self.kd = 1.0

    def load_policy(self, checkpoint_path: str):
        """Load trained RL policy"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.rl_policy.load_state_dict(checkpoint["policy_state_dict"])
        self.rl_policy.eval()
        print(f"Loaded RL policy from {checkpoint_path}")

    def save_policy(
        self, checkpoint_path: str, optimizer_state_dict=None, episode=None
    ):
        """Save RL policy checkpoint"""
        checkpoint = {
            "policy_state_dict": self.rl_policy.state_dict(),
            "episode": episode,
        }
        if optimizer_state_dict is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state_dict
        torch.save(checkpoint, checkpoint_path)

    def _prepare_rl_observations(
        self, heightmap: torch.Tensor, robot_state: dict
    ) -> torch.Tensor:
        """Convert heightmap and robot state to RL observation vector"""
        # Flatten heightmap
        heightmap_flat = heightmap.view(self.num_envs, -1)  # (num_envs, 400)

        # Extract robot state components
        joint_pos = robot_state["joint_pos"]  # (num_envs, 12)
        joint_vel = robot_state["joint_vel"]  # (num_envs, 12)
        base_pos = robot_state["base_pos"]  # (num_envs, 3)
        base_lin_vel = robot_state["base_lin_vel"]  # (num_envs, 3)
        base_ang_vel = robot_state["base_ang_vel"]  # (num_envs, 3)

        # Convert quaternion to euler for easier learning
        base_euler = self._quat_to_euler(robot_state["base_quat"])  # (num_envs, 3)

        # Concatenate all observations
        obs = torch.cat(
            [
                heightmap_flat,
                joint_pos,
                joint_vel,
                base_pos,
                base_lin_vel,
                base_ang_vel,
                base_euler,
            ],
            dim=-1,
        )

        return obs

    def _quat_to_euler(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to euler angles"""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll (x-axis rotation)
        roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        # Pitch (y-axis rotation)
        sin_pitch = 2 * (w * y - z * x)
        sin_pitch = torch.clamp(sin_pitch, -1, 1)
        pitch = torch.asin(sin_pitch)

        # Yaw (z-axis rotation)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return torch.stack([roll, pitch, yaw], dim=-1)

    def compute_torques(
        self, heightmap: torch.Tensor, robot_state: dict
    ) -> torch.Tensor:
        """
        Hybrid controller: combines MPC with RL policy

        Returns:
            torch.Tensor: (num_envs, 12) desired joint torques
        """
        # Prepare observations for RL policy
        rl_obs = self._prepare_rl_observations(heightmap, robot_state)

        if self.training_mode:
            # During training, store observations for later RL updates
            self.last_observations = rl_obs.clone()

        # Get RL policy output
        with torch.no_grad() if not self.training_mode else torch.enable_grad():
            rl_actions = self.rl_policy(rl_obs)

        # Classical MPC baseline (replace this with your actual MPC)
        target_pos = (
            torch.tensor(
                [
                    0.0,
                    0.9,
                    -1.8,  # FL hip, thigh, calf
                    0.0,
                    0.9,
                    -1.8,  # FR hip, thigh, calf
                    0.0,
                    0.9,
                    -1.8,  # RL hip, thigh, calf
                    0.0,
                    0.9,
                    -1.8,  # RR hip, thigh, calf
                ]
            )
            .repeat(self.num_envs, 1)
            .to(robot_state["joint_pos"].device)
        )

        mpc_torques = (
            self.kp * (target_pos - robot_state["joint_pos"])
            - self.kd * robot_state["joint_vel"]
        )

        # Combine MPC + RL: RL directly outputs torques
        final_torques = rl_actions * 33.5  # Scale to A1 torque limits

        # Alternative fusion strategies:
        # final_torques = mpc_torques + rl_actions * 10.0  # RL as corrections
        # blend_factor = 0.7
        # final_torques = blend_factor * final_torques + (1 - blend_factor) * mpc_torques

        return torch.clamp(final_torques, -33.5, 33.5)

    def get_last_observations(self) -> torch.Tensor | None:
        """Get the last RL observations (for training)"""
        return getattr(self, "last_observations", None)
