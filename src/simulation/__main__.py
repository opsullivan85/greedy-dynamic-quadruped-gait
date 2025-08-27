from src.util import log_exceptions
import logging
logger = logging.getLogger(__file__)
from src.simulation.simulation import main

if __name__ == "__main__":
    with log_exceptions(logger):
        main()