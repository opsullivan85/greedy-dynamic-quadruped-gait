from src.control import RobotRunnerMin, RobotType
from src.robotinterface import interface
from nptyping import NDArray, Float32, Shape


class SimInterface(interface.RobotInterface):
    def init(self, dt: float):
        self.robot_runner = RobotRunnerMin()
        self.robot_runner.init(RobotType.GO1, dt)

    def get_torques(
        self,
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
        body_state: NDArray[Shape["13"], Float32],
        command: NDArray[Shape["3"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        joint_states = SimInterface._convert_joint_states(joint_states=joint_states)
        torques = self.robot_runner.run(dof_states=joint_states, body_states=body_state, commands=command)
        return SimInterface._convert_torques(torques=torques)

    def reset(self) -> None:
        self.robot_runner.reset()

    @staticmethod
    def _convert_joint_states(
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
    ) -> NDArray[Shape["12, 2"], Float32]:
        return joint_states.reshape((12, 2))

    @staticmethod
    def _convert_torques(
        torques: NDArray[Shape["12"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        return torques.reshape((4, 3))
