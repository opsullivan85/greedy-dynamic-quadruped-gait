import logging

import numpy as np
from nptyping import Float32, NDArray, Shape

from src.control import RobotRunnerFSM, RobotType
from src.sim2real.abstractinterface import Sim2RealInterface

logger = logging.getLogger(__name__)


class SimInterface(Sim2RealInterface):
    def __init__(self, dt: float, debug_logging: bool = False) -> None:
        self.logger: None | logging.Logger = None
        if debug_logging:
            self.logger = logger

        self.robot_runner = RobotRunnerFSM()
        self.robot_runner.init(RobotType.GO1, dt)

    def get_torques(
        self,
        joint_states: NDArray[Shape["4, 3, 2"], Float32],
        body_state: NDArray[Shape["13"], Float32],
        command: NDArray[Shape["3"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        joint_states_converted = SimInterface._convert_joint_states(
            joint_states_interface=joint_states
        )
        torques = self.robot_runner.run(
            dof_states=joint_states_converted, body_states=body_state, commands=command
        )
        torques_converted = SimInterface._convert_torques(torques_control=torques)

        if self.logger is not None:
            # make each array print on the next line, with a tab indent
            formatter = "\n\t\t"
            with np.printoptions(precision=5, suppress=True):
                joint_states_str = formatter + formatter.join(
                    str(joint_states).split("\n")
                )
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
        joint_states_interface: NDArray[Shape["4, 3, 2"], Float32],
    ) -> NDArray[Shape["12, 2"], Float32]:
        """_summary_

        Args:
            joint_states_interface (np.ndarray): (4, 3, 2) joint states in interface order
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
                index 2: state (0: position, 1: velocity)

        Returns:
            joint_states_control (np.ndarray): (12, 2) joint states in control order
                [
                    [FL_hip_pos, FL_hip_vel]
                    [FL_knee_pos, FL_knee_vel]
                    [FL_ankle_pos, FL_ankle_vel]
                    [FR_hip_pos, FR_hip_vel]
                    ...
                ]
        """
        return joint_states_interface.reshape((12, 2))

    @staticmethod
    def _convert_torques(
        torques_control: NDArray[Shape["12"], Float32],
    ) -> NDArray[Shape["4, 3"], Float32]:
        """_summary_

        Args:
            torques_control (np.ndarray): (12,) torques in control order
                [
                    FL_hip_torque
                    FL_knee_torque
                    FL_ankle_torque
                    FR_hip_torque
                    ...
                ]

        Returns:
            torques_interface (np.ndarray): (4, 3) torques in interface order
                index 0: leg index (0-3)
                index 1: joint index (0-2) (hip, upper leg, lower leg)
        """
        return torques_control.reshape((4, 3))
