import torch
from isaaclab.assets import ArticulationData
from isaaclab.envs import ManagerBasedRLEnv, VecEnvObs, VecEnvStepReturn
from src import PROJECT_ROOT

data_folder = PROJECT_ROOT / "data" / "evaluations"
data_folder.mkdir(parents=True, exist_ok=True)


class Evaluator:
    def __init__(self, env: ManagerBasedRLEnv, observations: VecEnvObs, trials: int, name: str) -> None:
        self.env = env
        self._reset_from_obs(observations)
        self.remaining_trials = trials
        self.data_file = data_folder / name
        self.data_file.unlink(missing_ok=True)

    def _reset(self):
        data_str = ",".join(self.terminal_distances.cpu().numpy()) + ","
        with open(self.data_file, "a") as f:
            f.write(data_str)

        self.remaining_trials -= 1
        observations, _ = self.env.reset()
        self._reset_from_obs(observations)

    def _reset_from_obs(self, observations: VecEnvObs):
        self.initial_obs: torch.Tensor = observations["policy"]  # type: ignore
        self.dones = torch.zeros(self.env.num_envs, dtype=torch.bool)
        self.terminal_distances = torch.zeros(self.env.num_envs, dtype=torch.float)

    @property
    def done(self) -> bool:
        return self.remaining_trials == 0
    
    def process(self, data: VecEnvStepReturn) -> None:
        observations, rew, terminated, truncated, info = data
        obs: torch.Tensor = observations["policy"]  # type: ignore
        dones = torch.logical_and(truncated, terminated)
        new_dones = torch.logical_xor(self.dones, dones)

        robot_data: ArticulationData = self.env.scene["robot"].data

        x_positions = robot_data.root_link_pos_w[:, 0]
        self.terminal_distances[new_dones] = x_positions[new_dones]

        self.dones = torch.logical_and(self.dones, terminated)
        self.dones = torch.logical_and(self.dones, truncated)
        if torch.all(self.dones):
            self._reset()


