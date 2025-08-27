from src.util import log_exceptions
import logging
logger = logging.getLogger(__file__)

@log_exceptions(logger)
def main():
    print("Hello, RobotInterface!")
    from src.robotinterface.siminterface import SimInterface
    import numpy as np
    sim_interface = SimInterface(dt=0.01, debug_logging=True)

    joint_states = np.zeros(shape=(4, 3, 2), dtype=np.float32)
    body_state = np.zeros(shape=(13,), dtype=np.float32)
    command = np.zeros(shape=(3,), dtype=np.float32)
    sim_interface.get_torques(joint_states=joint_states, body_state=body_state, command=command)
    sim_interface.get_torques(joint_states=joint_states, body_state=body_state, command=command)
    sim_interface.get_torques(joint_states=joint_states, body_state=body_state, command=command)

if __name__ == "__main__":
    main()