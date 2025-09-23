from src.util import log_exceptions
from src import get_logger

logger = get_logger()

def main(): ...


if __name__ == "__main__":
    with log_exceptions(logger):
        main()
