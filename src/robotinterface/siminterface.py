from src.control import RobotRunnerMin, RobotType
from src.robotinterface import interface
from nptyping import NDArray, Float32, Shape
import logging

logger = logging.getLogger(__file__)


class SimInterface(interface.RobotInterface):
    id: int = 1

    def __init__(self, dt: float, debug_logging: bool = False) -> None:
        self.id = SimInterface.id
        SimInterface.id += 1
        self.logger: None | logging.Logger = None
        if debug_logging:
            self.logger = logger
            self.logger.setLevel(logging.DEBUG)
            self.logger.debug(f"Creating SimInterface instance {self.id}")

        self.robot_runner = RobotRunnerMin()
        self.robot_runner.init(RobotType.GO1, dt)

    def get_torques(
        self,
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
        body_state: NDArray[Shape["13"], Float32],
        command: NDArray[Shape["3"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        joint_states_converted = SimInterface._convert_joint_states(
            joint_states=joint_states
        )
        torques = self.robot_runner.run(
            dof_states=joint_states_converted, body_states=body_state, commands=command
        )
        torques_converted = SimInterface._convert_torques(torques=torques)

        if self.logger is not None:

            self.logger.debug(
                f"got torques"
                "\t- joint_states: {joint_states}"
                "\t- body_state: {body_state}"
                "\t- command: {command}"
                "\t- torques: {torques}"
            )

        return torques_converted

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
