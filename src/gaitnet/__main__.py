from src.util import log_exceptions
from src import get_logger
logger = get_logger()

@log_exceptions(logger)
def main():
    print("Hello, GaitNet!")

if __name__ == "__main__":
    main()