from src.util import log_exceptions
from src.gaitnet import logger

@log_exceptions(logger)
def main():
    print("Hello, RobotInterface!")
    from src.robotinterface.siminterface import SimInterface
    sim_interface = SimInterface()
    sim_interface.init(dt=0.01)
    # sim_interface.get_torques("test")

if __name__ == "__main__":
    main()