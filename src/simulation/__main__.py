from src.util import log_exceptions
from src.simulation import logger
from src.simulation.simulation import main

if __name__ == "__main__":
    with log_exceptions(logger):
        main()