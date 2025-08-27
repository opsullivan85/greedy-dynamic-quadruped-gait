from src.control import RobotRunnerMin, RobotType
from src.robotinterface import interface
from nptyping import NDArray, Float32, Shape
from src.robotinterface import logger
import logging
import numpy as np


class SimInterface(interface.RobotInterface):
    def __init__(self, dt: float, debug_logging: bool = False) -> None:
        self.logger: None | logging.Logger = None
        if debug_logging:
            self.logger = logger

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
            # make each array print on the next line, with a tab indent
            formatter = "\n\t\t"
            with np.printoptions(precision=5, suppress=True):
                joint_states_str = formatter + formatter.join(str(joint_states).split("\n"))
                body_state_str = formatter + formatter.join(str(body_state).split("\n"))
                command_str = formatter + formatter.join(str(command).split("\n"))
                torques_str = formatter + formatter.join(str(torques).split("\n"))
            self.logger.debug(
                f"got torques\n"
                f"\t- joint_states:{joint_states_str}\n"
                f"\t- body_state:{body_state_str}\n"
                f"\t- command:{command_str}\n"
                f"\t- torques:{torques_str}"
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
